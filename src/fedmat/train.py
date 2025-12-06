"""FedMAT train experiments that packs local client training onto a GPU w streams."""

from __future__ import annotations

import contextlib
import copy
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from dataclasses import asdict

import hydra
import polars as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor, ViTForImageClassification

from fedmat import install_exception_handlers, install_global_log_context, set_log_context
from fedmat.data import Batch, build_dataloaders, load_named_dataset_subsets
from fedmat.evaluate import evaluate
from fedmat.train_utils import StateDict, aggregate_models, log_run_artifacts, WandbQuiver
from fedmat.utils import create_vit_classifier, default_device, get_amp_settings, set_seed

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fedmat.config import TrainConfig

logger = logging.getLogger(__name__)


class MetricRow(TypedDict):
    """Flat, typed metric record for federated serial training."""

    round: int
    client: int
    step: int  # local step index within round for this client
    global_step: int
    loss: float

def _build_optimizer(optimizer: Literal["sgd", "adam", "adagrad"], *args, **kwargs) -> torch.optim.Optimizer:
    if optimizer == "sgd":
        target = torch.optim.SGD
    elif optimizers == "adam":
        target = torch.optim.Adam
    elif optimizer == "adagrad":
        target = torch.optim.Adam
        kwargs["adagrad"] = True
    else:
        raise ValueError(f"'{optimizer}' is not a valid optimizer")

    return target(*args, **kwargs)

def _compute_gradient_norm(m: torch.nn.Module) -> Tensor:
    return torch.nn.utils.get_total_norm(p.grad for p in m.parameters() if p.grad is not None)

def _models_delta_norm(am: torch.mm.Module, bm: torch.mm.Module) -> Tensor:
    return torch.nn.utils.get_total_norm(ap - bm.get_parameter(name) for name, ap in am.named_parameters())

def _run_fed_training(train_config: TrainConfig, quiver: WandbQuiver | None = None) -> float | None:
    """Execute the federated training loop with per-step multi-client scheduling."""
    device = default_device()
    logger.info("Using device: %s", device)

    train_ds, eval_ds = load_named_dataset_subsets(
        dataset_name=train_config.dataset,
        max_train_samples=train_config.max_train_samples,
        max_eval_samples=train_config.max_eval_samples,
    )
    image_processor = AutoImageProcessor.from_pretrained(train_config.model_name)

    client_dataloaders, eval_dataloader = build_dataloaders(
        train_ds=train_ds,
        homogeneity=train_config.homogeneity,
        num_clients=train_config.num_clients,
        eval_ds=eval_ds,
        image_processor=image_processor,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        prefetch_factor=train_config.prefetch_factor,
        device=device,
    )
    client_batches = [len(dl) for dl in client_dataloaders]
    local_steps = train_config.local_steps or max(client_batches)
    # TODO configure this with an actual bespoke variable, don't tie everything to local_steps
    if train_config.local_steps is None:
        client_weights = torch.as_tensor(client_batches).to(device) / sum(client_batches)
    else:
        client_weights = None

    logger.info("Client batches: %s", client_batches)
    logger.info("Client weights: %s", client_weights)
    if quiver is not None:
        quiver.server_run.summary["client_batches"] = client_batches
        quiver.server_run.summary["client_weights"] = client_weights

    model: ViTForImageClassification = create_vit_classifier(
        model_name=train_config.model_name,
        num_labels=train_config.num_labels,
        use_pretrained=train_config.use_pretrained,
    )
    _ = model.to(device=device)  # type: ignore
    if train_config.use_torch_compile:
        model.compile()

    global_state: StateDict = copy.deepcopy(model.state_dict())  # type: ignore
    global_step = 0
    all_train_metrics: list[MetricRow] = []
    final_accuracy: float | None = None
    final_confmat: torch.Tensor | None = None
    best_server_accuracy = float("-inf")

    # set up streams if on cuda
    streams: list[torch.cuda.Stream | None] = [
        torch.cuda.Stream(device=device) if device.type == "cuda" else None for _ in range(train_config.num_clients)
    ]

    use_autocast, amp_dtype = get_amp_settings(device, train_config.use_bf16)

    for round_idx in range(train_config.num_rounds):
        logger.info(
            "Starting round %d/%d",
            round_idx + 1,
            train_config.num_rounds,
        )

        # Per-round client models, optimizers, streams, and dataloader iterators
        client_models: list[ViTForImageClassification] = []
        client_optimizers: list[torch.optim.Optimizer] = []
        client_iters: list[Iterator[Batch]] = []

        for client_idx, dataloader in enumerate(client_dataloaders):
            client_model = copy.deepcopy(model)  # init from current server arch
            client_model.load_state_dict(global_state)
            _ = client_model.to(device=device)  # type: ignore

            optimizer = _build_optimizer(
                train_config.optimizer,
                client_model.parameters(),
                lr=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            )

            client_models.append(client_model)
            client_optimizers.append(optimizer)
            client_iters.append(iter(dataloader))

        # Per-client running loss on device + logging counters
        running_loss_gpu: list[torch.Tensor | None] = [None] * train_config.num_clients
        steps_since_log: list[int] = [0] * train_config.num_clients

        # Timing start
        if train_config.enable_timing:
            if device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            else:
                start_time = time.perf_counter()

        # Local training: step-major, client-minor loop
        for local_step in range(local_steps):
            for client_idx in range(train_config.num_clients):
                stream = streams[client_idx]
                client_model = client_models[client_idx]
                optimizer = client_optimizers[client_idx]
                it = client_iters[client_idx]
                batch_metrics = {
                    "round": round_idx,
                    "client": client_idx,
                    "step": local_step,
                    "global_step": global_step,
                }

                try:
                    batch = next(it)
                except StopIteration:
                    if train_config.local_steps is None:
                        continue
                    it = iter(client_dataloaders[client_idx])
                    client_iters[client_idx] = it
                    batch = next(it)

                client_model.train()

                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                ctx = (
                    torch.cuda.stream(stream)
                    if (stream is not None and device.type == "cuda")
                    else contextlib.nullcontext()
                )
                with ctx:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                        outputs = client_model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    # Log gradient norm
                    batch_metrics["gradient_norm"] = float(_compute_gradient_norm(client_model))

                    if train_config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            client_model.parameters(),
                            train_config.max_grad_norm,
                        )
                        # Log gradient norm after clipping
                        batch_metrics["clipped_gradient_norm"] = float(_compute_gradient_norm(client_model))

                    optimizer.step()

                    loss_det = loss.detach()

                    # Metrics and logging bookkeeping
                    steps_since_log[client_idx] += 1

                    # Accumulate running loss on device
                    if running_loss_gpu[client_idx] is None:
                        running_loss_gpu[client_idx] = loss_det
                    else:
                        running_loss_gpu[client_idx] = running_loss_gpu[client_idx] + loss_det

                    batch_metrics["loss"] = float(loss_det.item())  # sync

                    # Log metrics for this batch
                    all_train_metrics.append(batch_metrics)
                    if quiver is not None:
                        quiver.client_runs[client_idx].log(batch_metrics)

                    # Logging - interval or round end
                    # do_log = (
                    #     steps_since_log[client_idx] >= train_config.log_every_n_steps
                    #     or local_step == train_config.local_steps - 1
                    # )
                    do_log = ((local_step + 1) % train_config.log_every_n_steps == 0) or (
                        local_step == local_steps - 1
                    )
                    if do_log and running_loss_gpu[client_idx] is not None:
                        avg_loss_gpu = running_loss_gpu[client_idx] / steps_since_log[client_idx]  # type: ignore
                        avg_loss = float(avg_loss_gpu.item())  # sync
                        logger.info(
                            "Round %d client %d local_step %d global_step %d avg_loss=%.4f",
                            round_idx + 1,
                            client_idx + 1,
                            local_step,
                            global_step,
                            avg_loss,
                        )
                        running_loss_gpu[client_idx] = None
                        steps_since_log[client_idx] = 0

            global_step += 1 # the X-axis should increment only per local step, not between client switches

        # Barrier
        for s in streams:
            if s is not None:
                s.synchronize()

        # Timing end for local training.
        if train_config.enable_timing:
            if device.type == "cuda":
                end_evt.record()  # type: ignore
                torch.cuda.synchronize()
                elapsed_s = start_evt.elapsed_time(end_evt) / 1000  # pyright: ignore[reportPossiblyUnboundVariable]
            else:
                elapsed_s = time.perf_counter() - start_time  # pyright: ignore[reportPossiblyUnboundVariable]
            logger.info(
                "Round %d local training time: %.3f s",
                round_idx + 1,
                elapsed_s,
            )

        round_metrics = { "round": round_idx }

        if quiver is not None:
            delta_norm_sum = 0
            for client_idx, client_model in enumerate(client_models):
                delta_norm = float(_models_delta_norm(client_model, model))
                round_metrics[f"client_{client_idx}_delta_norm"] = delta_norm
                delta_norm_sum += delta_norm
            round_metrics[f"client_delta_norm_sum"] = delta_norm_sum

        # Aggregate client models into a new global state
        if train_config.enable_timing:
            start_time = time.perf_counter()

        aggregated_state = aggregate_models(
            client_models,
            matcher=train_config.matcher,
            client_weights=client_weights,
        )
        old_model = copy.deepcopy(model)
        model.load_state_dict(aggregated_state)

        global_state = aggregated_state
        if train_config.enable_timing:
            elapsed_s = time.perf_counter() - start_time  # pyright: ignore[reportPossiblyUnboundVariable]
            logger.info("Round %d aggregation time: %.3f s", round_idx + 1, elapsed_s)

        if quiver is not None:
            round_metrics["aggregate_delta_norm"] = float(_models_delta_norm(old_model, model))

        accuracy, confmat = evaluate(
            model,
            eval_dataloader,
            device,
            enable_bf16=train_config.use_bf16,
        )
        final_accuracy = float(accuracy)
        final_confmat = confmat
        round_metrics["accuracy"] = final_accuracy
        round_metrics["confmat"] = confmat
        round_metrics["global_step"] = global_step

        logger.info(
            "Server eval after round %d: accuracy=%.4f",
            round_idx + 1,
            final_accuracy,
        )

        if quiver is not None:
            quiver.server_run.log(round_metrics)

        # Best-server checkpoint on disk in run dir
        if final_accuracy > best_server_accuracy:
            best_server_accuracy = final_accuracy
            best_ckpt_path = train_config.output_dir / "best-server.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                "New best server model (acc=%.4f) saved to %s",
                final_accuracy,
                best_ckpt_path,
            )

    metrics_df = pl.DataFrame(
        all_train_metrics,
        schema={
            "round": pl.Int64,
            "client": pl.Int64,
            "step": pl.Int64,
            "global_step": pl.Int64,
            "loss": pl.Float64,
        },
    )

    # Log all artifacts
    artifacts_dir = log_run_artifacts(
        train_config=train_config,
        model=model,
        metrics_df=metrics_df,
        final_confmat=final_confmat,
        final_accuracy=final_accuracy,
        run=None if quiver is None else quiver.server_run,
    )

    logger.info("Saved run artifacts to %s", artifacts_dir)
    logger.info("Final evaluation accuracy: %s", final_accuracy)
    return final_accuracy


@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for serial federated training."""
    install_exception_handlers(threading_=True, asyncio_=True)
    OmegaConf.resolve(cfg)

    install_global_log_context()
    set_log_context(rank="main", world_size=cfg.num_clients, backend="thread")

    config: TrainConfig = instantiate(cfg)
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    if config.dry:
        logger.warning("Dry run requested; exiting before training.")
        return

    result: float | None = None
    if config.use_wandb:
        if config.run_name is None:
            group = input("Please enter a run name: ")
        else:
            group = config.run_name
        group = group.strip()
        if group == "":
            logger.error("No valid group name supplied")
            return
        quiver = WandbQuiver(
            group=group,
            num_clients=config.num_clients,

            entity="fedmat-team",
            project="fedmat-project",
            config=asdict(config),
            mode="online",
            dir=str(config.output_dir),
        )
    else:
        quiver = None
    try:
        result = _run_fed_training(config, quiver)
    finally:
        logger.info(
            "Training finished with accuracy %.4f",
            result or 0.0,
        )
        if quiver is not None:
            quiver.finish()

if __name__ == "__main__":
    main()
