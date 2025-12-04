"""FedMAT train experiments that packs local client training onto a GPU w streams."""

from __future__ import annotations

import contextlib
import copy
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypedDict

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor, ViTForImageClassification

from fedmat import install_exception_handlers, install_global_log_context, set_log_context
from fedmat.data import Batch, build_dataloaders, load_named_dataset_subsets
from fedmat.evaluate import evaluate
from fedmat.train_utils import ModelReshaper, aggregate_models, clone_state_dict, log_run_artifacts
from fedmat.utils import create_vit_classifier, get_amp_settings, set_seed

if TYPE_CHECKING:
    from fedmat.config import TrainConfig

logger = logging.getLogger(__name__)


class MetricRow(TypedDict):
    """Flat, typed metric record for federated serial training."""

    round: int
    client: int
    step: int  # local step index within round for this client
    global_step: int
    loss: float


def _select_device() -> torch.device:
    """Select the best available device for serial training."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_fed_training(train_config: TrainConfig) -> float | None:
    """Execute the federated training loop with per-step multi-client scheduling."""
    device = _select_device()
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

    model: ViTForImageClassification = create_vit_classifier(
        model_name=train_config.model_name,
        num_labels=train_config.num_labels,
        use_pretrained=train_config.use_pretrained,
    )
    model.to(device)
    if train_config.use_torch_compile:
        model.compile()

    # Flatten metadata from the server model once
    # metadata = build_flat_metadata(model.state_dict())
    reshaper = ModelReshaper()

    global_state = copy.deepcopy(model.state_dict())
    global_step = 0
    all_train_metrics: list[MetricRow] = []
    final_accuracy: float | None = None
    final_confmat: torch.Tensor | None = None
    best_server_accuracy = float("-inf")

    # set up streams if on cuda.
    streams: list[torch.cuda.Stream | None] = [
        torch.cuda.Stream(device=device) if device.type == "cuda" else None for _ in range(train_config.num_clients)
    ]

    use_autocast, amp_dtype = get_amp_settings(device, train_config.use_bf16)
    logger.info("Dataset lens: %s", [len(dl) for dl in client_dataloaders])

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
            client_model.to(device)

            optimizer = torch.optim.SGD(
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
        for local_step in range(train_config.local_steps):
            for client_idx in range(train_config.num_clients):
                stream = streams[client_idx]
                client_model = client_models[client_idx]
                optimizer = client_optimizers[client_idx]
                it = client_iters[client_idx]

                client_model.train()

                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(client_dataloaders[client_idx])
                    client_iters[client_idx] = it
                    batch = next(it)

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
                    if train_config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            client_model.parameters(),
                            train_config.max_grad_norm,
                        )
                    optimizer.step()

                    loss_det = loss.detach()

                    # Metrics and logging bookkeeping
                    global_step += 1
                    steps_since_log[client_idx] += 1

                    # Accumulate running loss on device
                    if running_loss_gpu[client_idx] is None:
                        running_loss_gpu[client_idx] = loss_det
                    else:
                        running_loss_gpu[client_idx] = running_loss_gpu[client_idx] + loss_det

                    all_train_metrics.append({
                        "round": round_idx,
                        "client": client_idx,
                        "step": local_step,
                        "global_step": global_step,
                        "loss": float(loss_det.item()),  # sync
                    })

                    # Logging - interval or round end
                    # do_log = (
                    #     steps_since_log[client_idx] >= train_config.log_every_n_steps
                    #     or local_step == train_config.local_steps - 1
                    # )
                    do_log = ((local_step + 1) % train_config.log_every_n_steps == 0) or (
                        local_step == train_config.local_steps - 1
                    )

                    if do_log and running_loss_gpu[client_idx] is not None:
                        avg_loss_gpu = running_loss_gpu[client_idx] / steps_since_log[client_idx]
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

        # Barrier
        for s in streams:
            if s is not None:
                s.synchronize()

        # Timing end for local training.
        if train_config.enable_timing:
            if device.type == "cuda":
                end_evt.record()
                torch.cuda.synchronize()
                elapsed_s = start_evt.elapsed_time(end_evt) / 1000
            else:
                elapsed_s = time.perf_counter() - start_time
            logger.info(
                "Round %d local training time: %.3f s",
                round_idx + 1,
                elapsed_s,
            )


        # Aggregate client models into a new global state
        if train_config.enable_timing:
            start_time = time.perf_counter()
        client_states = [client_model.state_dict() for client_model in client_models]
        if train_config.enable_timing:
            elapsed_s = time.perf_counter() - start_time
            logger.info(
                "Round %d communication time: %.3f s",
                round_idx + 1,
                elapsed_s
            )
            start_time = time.perf_counter()
        aggregated_state = aggregate_models(
            client_states,
            model.config,
            device=device,
            matcher=train_config.matcher,
        )
        if train_config.enable_timing:
            elapsed_s = time.perf_counter() - start_time
            logger.info(
                "Round %d aggregation time: %.3f s",
                round_idx + 1,
                elapsed_s
            )

        # Canonicalize model weights via flat + unflat and load
        flat = reshaper.flatten(aggregated_state)
        reshaper.unflatten_model(model, flat)
        global_state = clone_state_dict(model.state_dict(), device=device)

        accuracy, confmat = evaluate(
            model,
            eval_dataloader,
            device,
            enable_bf16=train_config.use_bf16,
        )
        final_accuracy = float(accuracy)
        final_confmat = confmat

        logger.info(
            "Server eval after round %d: accuracy=%.4f",
            round_idx + 1,
            final_accuracy,
        )

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
    try:
        result = _run_fed_training(config)
    finally:
        logger.info(
            "Training finished with accuracy %.4f",
            result or 0.0,
        )


if __name__ == "__main__":
    main()
