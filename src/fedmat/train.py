"""Training utilities for vision transformer models on federated data."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

from . import install_exception_handlers, install_global_log_context, set_log_context, utils
from .data import Batch, build_dataloaders, load_cifar10_subsets
from .distributed_context import DistributedContext
from .evaluate import evaluate
from .permute import permute_vit_layer_heads
from .train_utils import ModelFlatMetadata, flatten_state_dict, unflatten_state_dict
from .utils import create_vit_classifier, get_amp_settings, set_seed

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader
    from transformers.models.vit.modeling_vit import ViTLayer

    from .configs import TrainConfig
    from .matching import Matcher

logger = logging.getLogger(__name__)


class MetricRecord(TypedDict):
    """Metric record for a training step."""

    epoch: int
    global_step: int
    step: int
    loss: float


Metrics = list[MetricRecord]


@dataclass
class TrainingMetadata:
    """Training metadata across epochs for logging purposes."""

    global_step: int = 0
    best_loss: float = float("inf")


def _resolve_config_paths(config: TrainConfig) -> TrainConfig:
    """Resolve all relative paths within the config."""
    raw_output_dir = Path(config.output_dir)
    output_dir = Path(to_absolute_path(raw_output_dir.as_posix()))
    return replace(config, output_dir=output_dir)


def _initialize_server_and_matcher(
    cfg: TrainConfig,
    model: ViTForImageClassification,
    ctx: DistributedContext,
) -> tuple[ViTForImageClassification | None, Matcher | None]:
    """Initialize server model (on rank 0) and matcher."""
    matcher = cfg.matcher
    server_model: ViTForImageClassification | None = None
    if ctx.is_main:
        server_model = create_vit_classifier(
            model_name=cfg.model_name,
            num_labels=cfg.num_labels,
            use_pretrained=cfg.use_pretrained,
        )
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        server_model.load_state_dict(cpu_state)
        server_model.to("cpu")
    return server_model, matcher


def _broadcast_initial_state(
    model: ViTForImageClassification,
    server_model: ViTForImageClassification | None,
    ctx: DistributedContext,
) -> torch.Tensor:
    """Broadcast initial global model from server (rank 0) to all clients using flattened weights."""
    flat = ctx.flatten_model(model)
    flat = ctx.broadcast_tensor(flat, src=0)
    ctx.unflatten_model(model, flat)
    metadata = ctx.metadata
    assert metadata is not None
    if server_model is not None:
        server_model.load_state_dict(unflatten_state_dict(flat, metadata))
    return flat


def _compute_mean_loss(metrics: Metrics) -> float | None:
    """Return the mean loss from a list of metric records.

    This is dumb.
    """
    if not metrics:
        return None
    losses = [rec["loss"] for rec in metrics if math.isfinite(rec["loss"])]
    if not losses:
        return None
    return float(sum(losses) / len(losses))


def _aggregate_and_evaluate_round(
    cfg: TrainConfig,
    server_model: ViTForImageClassification | None,
    matcher: Matcher | None,
    eval_dataloader: DataLoader[Batch],
    ctx: DistributedContext,
    gathered: list[torch.Tensor],
    metadata: ModelFlatMetadata,
    round_idx: int,
    server_best_accuracy: float,
    round_train_loss: float | None,
) -> tuple[torch.Tensor, float | None, torch.Tensor | None, float]:
    """Aggregate client models on rank 0, evaluate, and log artifacts."""
    aggregated_flat = torch.zeros_like(gathered[0])
    final_accuracy: float | None = None
    final_confmat: torch.Tensor | None = None

    if ctx.is_main:
        assert server_model is not None
        state_dicts = [unflatten_state_dict(vec, metadata) for vec in gathered]
        logger.info("Aggregating %d client models on rank 0 (round %d)", len(state_dicts), round_idx + 1)
        aggregated_state = aggregate_models(
            state_dicts,
            server_model.config,
            device=ctx.device,
            matcher=matcher,
        )
        aggregated_flat = flatten_state_dict(aggregated_state, metadata)
        server_model.load_state_dict(aggregated_state)

        server_model.to(ctx.device)
        acc, conf = evaluate(server_model, eval_dataloader, ctx.device, enable_bf16=cfg.use_bf16)
        server_model.to("cpu")

        final_accuracy = float(acc)
        final_confmat = conf

        now = datetime.now().isoformat(timespec="seconds")
        server_ckpt = cfg.output_dir / f"server_model-round{round_idx + 1}-{now}.pt"
        torch.save(server_model.state_dict(), server_ckpt)
        logger.info(
            "Server eval after round %d: accuracy=%.4f; checkpoint saved to %s",
            round_idx + 1,
            final_accuracy,
            server_ckpt,
        )

        if conf is not None:
            conf_path = cfg.output_dir / f"server_confusion-round{round_idx + 1}-{now}.pt"
            torch.save(conf.cpu(), conf_path)
            logger.info("Confusion matrix saved to %s", conf_path)

        if final_accuracy > server_best_accuracy:
            server_best_accuracy = final_accuracy
            best_ckpt_path = cfg.output_dir / "best-server.pt"
            torch.save(server_model.state_dict(), best_ckpt_path)
            logger.info("New best server model (acc=%.4f) saved to %s", server_best_accuracy, best_ckpt_path)
            ctx.wandb_log_artifact("best-server-model", best_ckpt_path)

        log_payload = {
            "round/number": round_idx + 1,
            "round/server_accuracy": final_accuracy,
            "round/server_best_accuracy": server_best_accuracy,
        }
        if round_train_loss is not None:
            log_payload["round/server_train_loss"] = round_train_loss

        ctx.wandb_log(log_payload, step=round_idx + 1)

    aggregated_flat = ctx.broadcast_tensor(aggregated_flat, src=0)
    return aggregated_flat, final_accuracy, final_confmat, server_best_accuracy


def _save_federated_artifacts(
    cfg: TrainConfig,
    model: ViTForImageClassification,
    final_confmat: torch.Tensor | None,
    metrics: dict,
) -> None:
    """Save metrics, confusion matrix, and final client model for federated runs."""
    now = datetime.now().isoformat(timespec="seconds")

    metrics_path = cfg.output_dir / f"metrics-{now}.json"
    logger.info("Saving metrics to %s", metrics_path)
    with metrics_path.open("w") as f:
        json.dump(metrics, fp=f, indent=2)

    if final_confmat is not None:
        conf_path = cfg.output_dir / f"confusion_matrix-{now}.pt"
        logger.info("Saving final confusion matrix to %s", conf_path)
        torch.save(final_confmat.cpu(), conf_path)

    ckpt_path = cfg.output_dir / f"model-{now}.pt"
    logger.info("Saving final local model checkpoint to %s", ckpt_path)
    torch.save(model.state_dict(), ckpt_path)


def _build_client_models(
    state_dicts: list[dict[str, torch.Tensor]],
    template_config: ViTConfig,
) -> list[ViTForImageClassification]:
    """Instantiate client models on CPU from state dicts."""
    client_models: list[ViTForImageClassification] = []
    for sd in state_dicts:
        model = ViTForImageClassification(template_config)
        model.load_state_dict(sd)
        model.to("cpu")
        client_models.append(model)
    return client_models


def _aggregate_encoder_layers(
    client_models: list[ViTForImageClassification],
    device: torch.device,
    matcher: Matcher | None,
) -> dict[str, torch.Tensor]:
    """Aggregate encoder layers with optional head matching."""
    aggregated_state: dict[str, torch.Tensor] = {}

    reference_model = client_models[0]
    num_layers = len(reference_model.vit.encoder.layer)

    with torch.no_grad():
        for layer_idx in range(num_layers):
            client_layers: list[ViTLayer] = [model.vit.encoder.layer[layer_idx] for model in client_models]

            for layer in client_layers:
                layer.to(device)

            if matcher is not None:
                perms = matcher.match(client_layers)
                for layer, perm in zip(client_layers, perms):
                    permute_vit_layer_heads(layer, perm)

            param_maps = [dict(layer.named_parameters()) for layer in client_layers]
            reference_param_map = param_maps[0]

            for name, reference_param in reference_param_map.items():
                full_name = f"vit.encoder.layer.{layer_idx}.{name}" if name else f"vit.encoder.layer.{layer_idx}"
                tensors = [param_map[name].data.float() for param_map in param_maps]
                stacked = torch.stack(tensors, dim=0)
                avg = stacked.mean(dim=0).to(reference_param.dtype).cpu()
                aggregated_state[full_name] = avg

            buffer_maps = [dict(layer.named_buffers()) for layer in client_layers]
            if buffer_maps:
                reference_buffer_map = buffer_maps[0]
                for name, reference_buffer in reference_buffer_map.items():
                    full_name = f"vit.encoder.layer.{layer_idx}.{name}" if name else f"vit.encoder.layer.{layer_idx}"
                    if torch.is_floating_point(reference_buffer):
                        tensors = [buffer_map[name].data.float() for buffer_map in buffer_maps]
                        stacked = torch.stack(tensors, dim=0)
                        avg = stacked.mean(dim=0).to(reference_buffer.dtype).cpu()
                        aggregated_state[full_name] = avg
                    else:
                        aggregated_state[full_name] = reference_buffer.detach().cpu().clone()

            for layer in client_layers:
                layer.to("cpu")

    return aggregated_state


def _aggregate_remaining_parameters(
    client_models: list[ViTForImageClassification],
    aggregated_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Aggregate all non-encoder parameters and buffers."""
    client_state_dicts = [model.state_dict() for model in client_models]
    reference_state_dict = client_state_dicts[0]

    with torch.no_grad():
        for name, reference_tensor in reference_state_dict.items():
            if name in aggregated_state:
                continue

            if torch.is_floating_point(reference_tensor):
                stacked = torch.stack(
                    [sd[name].to(device=device, dtype=torch.float32) for sd in client_state_dicts],
                    dim=0,
                )
                avg = stacked.mean(dim=0).to(reference_tensor.dtype).cpu()
                aggregated_state[name] = avg
            else:
                aggregated_state[name] = reference_tensor.detach().clone()

    return aggregated_state


def aggregate_models(
    state_dicts: list[dict[str, torch.Tensor]],
    template_config: ViTConfig,
    device: torch.device,
    matcher: Matcher | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate client ViT models with per-layer matched averaging."""
    client_models = _build_client_models(state_dicts, template_config)
    aggregated_state = _aggregate_encoder_layers(client_models, device, matcher)
    aggregated_state = _aggregate_remaining_parameters(client_models, aggregated_state, device)
    return aggregated_state


def train_epoch(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
    ctx: DistributedContext | None,
    training_metadata: TrainingMetadata,
    epoch_metadata: dict | None = None,
    epoch_name: str = "",
) -> tuple[Metrics, float]:
    """Train model for one epoch.

    Parameters
    ----------
    model : ViTForImageClassification
        Vision Transformer model to train
    dataloader : DataLoader[Batch]
        Training data loader
    device : torch.device
        Device to train on
    cfg : TrainConfig
        Training configuration
    training_metadata : TrainingMetadata
        Metadata tracked across epochs
    epoch_metadata : dict | None, optional
        Metadata for this epoch, by default None
    epoch_name : str, optional
        Name for progress bar, by default ""

    Returns
    -------
    tuple[Metrics, float]
        Tuple of (metrics_list, final_loss)
    """
    if epoch_metadata is None:
        epoch_metadata = {}
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_autocast, amp_dtype = get_amp_settings(device, cfg.use_bf16)

    n_batches = len(dataloader)

    metrics_loss_gpu: Tensor = torch.empty(
        n_batches,
        dtype=torch.float32,
        device=device,
    )

    metrics_meta: list[MetricRecord] = []  # cpu

    try:
        running_loss_gpu: Tensor = torch.zeros((), device=device)
        steps_since_log = 0

        progress = tqdm(dataloader, desc=epoch_name) if (ctx and ctx.is_main) else dataloader

        for step, batch in enumerate(progress):
            training_metadata.global_step += 1
            steps_since_log += 1

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            ):
                outputs = model(
                    pixel_values=pixel_values,  # (B, C, H, W)
                    labels=labels,  # (B,)
                )
                loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            optimizer.step()

            metrics_loss_gpu[step] = loss.detach()
            metrics_meta.append({
                **epoch_metadata,
                "step": step,
                "global_step": training_metadata.global_step,
                "loss": float("nan"),  # placeholder
            })

            running_loss_gpu += loss.detach()
            if training_metadata.global_step % cfg.log_every_n_steps == 0 or step == len(dataloader) - 1:
                avg_loss_gpu = running_loss_gpu / steps_since_log
                avg_loss = float(avg_loss_gpu.item())

                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(loss=f"{avg_loss:.4f}")

                # NOTE: ONLY MAIN rank logs metrics rn...
                # ctx should NEVER be done this is stupid mistake an against the design
                if ctx:
                    ctx.wandb_log(
                        {
                            **{f"train/{k}": v for k, v in epoch_metadata.items()},
                            "train/loss": avg_loss,
                            "train/best_loss": training_metadata.best_loss,
                            "train/step": step,
                            "train/global_step": training_metadata.global_step,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=training_metadata.global_step,
                    )

                # NOTE: ONLY MAIN rank checkpoints main writes best rn should instead put the rank in,
                #   the filename i guess and have them all log
                if avg_loss < training_metadata.best_loss:
                    training_metadata.best_loss = avg_loss

                    if ctx is None or ctx.is_main:
                        ckpt_path = cfg.output_dir / "best.pt"
                        torch.save(
                            {
                                **epoch_metadata,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "global_step": training_metadata.global_step,
                            },
                            ckpt_path,
                        )

                        if ctx is not None and getattr(cfg, "save_round_checkpoints", False):
                            ctx.wandb_log_artifact("best-model", ckpt_path)

                # reset local running stats
                running_loss_gpu.zero_()
                steps_since_log = 0

    finally:
        # sync and fill
        loss_list_cpu = metrics_loss_gpu.cpu().tolist()
        for rec in metrics_meta:
            rec["loss"] = loss_list_cpu[rec["step"]]

    return metrics_meta


def train(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
    ctx: DistributedContext | None,
) -> Metrics:
    """Train model for multiple epochs.

    Parameters
    ----------
    model : ViTForImageClassification
        Vision Transformer model to train
    dataloader : DataLoader[Batch]
        Training data loader
    device : torch.device
        Device to train on
    cfg : TrainConfig
        Training configuration

    Returns
    -------
    Metrics
        List of metric records for all training steps
    """
    epoch_name_padding = " " * len(str(cfg.epochs))
    metrics = []
    training_metadata = TrainingMetadata()
    for epoch in range(cfg.epochs):
        epoch_padded = epoch_name_padding + str(epoch)
        epoch_metrics = train_epoch(
            model,
            dataloader,
            device,
            cfg,
            ctx,
            training_metadata=training_metadata,
            epoch_metadata={"epoch": epoch},
            epoch_name=f"Epoch {epoch_padded}/{cfg.epochs}",
        )
        metrics.extend(epoch_metrics)

    if ctx is not None and ctx.is_main:
        ctx.wandb_update_summary("final/loss", metrics[-1]["loss"])
        ctx.wandb_update_summary("best/loss", training_metadata.best_loss)
    return metrics


def _main_local(cfg: TrainConfig, ctx: DistributedContext) -> float:
    """Run local (single-process) training."""
    device = ctx.device
    logger.info("Using device: %s", device)

    train_ds, eval_ds = load_cifar10_subsets(
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )

    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    model: ViTForImageClassification = create_vit_classifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        use_pretrained=cfg.use_pretrained,
    )

    logger.info("Model:\n%s", pformat(model))

    _ = model.to(device)

    if cfg.use_torch_compile:
        model.compile()

    client_dataloaders, eval_dataloader = build_dataloaders(
        train_ds=train_ds,
        homogeneity=cfg.homogeneity,
        num_clients=cfg.num_clients,
        eval_ds=eval_ds,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    train_dataloader = client_dataloaders[0]

    # Train
    metrics_train = train(model, train_dataloader, device, cfg, ctx)
    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(metrics_train)

    # Eval
    accuracy, confmat = evaluate(model, eval_dataloader, device, enable_bf16=cfg.use_bf16)
    metrics["eval"]["accuracy"] = accuracy

    # Save
    now = datetime.now().isoformat()

    fpath = cfg.output_dir.joinpath(f"metrics-{now}.json")
    logger.info("Saving metrics: %s", fpath)
    with fpath.open("w") as f:
        json.dump(metrics, fp=f, indent=2)

    fpath = cfg.output_dir / f"confusion_matrix-{now}.pt"
    logger.info("Saving confusion matrix data to %s", fpath)
    torch.save(confmat.cpu(), fpath)

    ckpt_path = cfg.output_dir / f"model-{now}.pt"
    logger.info("Saving checkpoint to %s", ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

    logger.info("Evaluation accuracy: %.4f", accuracy)
    if ctx.is_main:
        ctx.wandb_update_summary("eval/accuracy", accuracy)
        ctx.wandb_log_artifact("final-model", ckpt_path)

    return float(accuracy)


def _main_federated(cfg: TrainConfig, ctx: DistributedContext) -> float | None:
    """Run distributed FedAvg-style training using flattened tensor collectives."""
    device = ctx.device
    logger.info("Using device: %s (rank %d/%d)", device, ctx.rank, ctx.world_size)

    train_ds, eval_ds = load_cifar10_subsets(
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    if cfg.use_pretrained:
        logger.info("Using pretrained backbone '%s'", cfg.model_name)
    else:
        logger.info("Training from scratch")

    model: ViTForImageClassification = create_vit_classifier(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        use_pretrained=cfg.use_pretrained,
    )
    model.to(device)
    if cfg.use_torch_compile:
        model.compile()

    client_dataloaders, eval_dataloader = build_dataloaders(
        train_ds=train_ds,
        homogeneity=cfg.homogeneity,
        num_clients=cfg.num_clients,
        eval_ds=eval_ds,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )
    train_dataloader = client_dataloaders[ctx.rank]

    server_model, matcher = _initialize_server_and_matcher(cfg, model, ctx)
    metadata = ctx.ensure_metadata(model)
    _broadcast_initial_state(model, server_model, ctx)

    all_train_metrics: Metrics = []
    server_best_accuracy = float("-inf")
    final_accuracy: float | None = None
    final_confmat: torch.Tensor | None = None

    server_round_train_losses: list[float] = []

    for round_idx in range(cfg.num_rounds):
        logger.info("Starting communication round %d/%d", round_idx + 1, cfg.num_rounds)

        round_metrics = train(model, train_dataloader, device, cfg, ctx)
        all_train_metrics.extend(round_metrics)

        round_train_loss = _compute_mean_loss(round_metrics)
        aggregated_round_loss: float | None = None
        if round_train_loss is not None:
            loss_tensor = torch.tensor(round_train_loss, dtype=torch.float32, device=ctx.device)
        else:
            loss_tensor = torch.tensor(float("nan"), dtype=torch.float32, device=ctx.device)
        gathered_losses = ctx.all_gather_tensor(loss_tensor)
        finite_losses = [float(t.item()) for t in gathered_losses if math.isfinite(float(t.item()))]
        if finite_losses:
            aggregated_round_loss = sum(finite_losses) / len(finite_losses)
            server_round_train_losses.append(aggregated_round_loss)

        local_flat = ctx.flatten_model(model)
        gathered = ctx.all_gather_tensor(local_flat)

        aggregated_flat, final_accuracy, final_confmat, server_best_accuracy = _aggregate_and_evaluate_round(
            cfg=cfg,
            server_model=server_model,
            matcher=matcher,
            eval_dataloader=eval_dataloader,
            ctx=ctx,
            gathered=gathered,
            metadata=metadata,
            round_idx=round_idx,
            server_best_accuracy=server_best_accuracy,
            round_train_loss=aggregated_round_loss,
        )

        ctx.unflatten_model(model, aggregated_flat)

    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(all_train_metrics)
    if server_round_train_losses:
        metrics["server"]["train_loss"] = server_round_train_losses

    if ctx.is_main and final_accuracy is not None:
        metrics["eval"]["accuracy"] = final_accuracy
        metrics["server"]["eval_accuracy"] = final_accuracy
        _save_federated_artifacts(cfg, model, final_confmat, metrics)
        ctx.wandb_update_summary("eval/accuracy", final_accuracy)

    return final_accuracy


# def _driver(ctx: DistributedContext, cfg: TrainConfig) -> float | None:
#     """Driver function executed per rank."""
#     set_seed(cfg.seed)
#     ctx.wandb_init(cfg)
#     try:
#         if cfg.mode == "local":
#             return _main_local(cfg, ctx)
#         return _main_federated(cfg, ctx)
#     finally:
#         ctx.wandb_finish()


# @hydra.main(config_path="configs", config_name="experiment", version_base=None)
# def main(cfg: DictConfig) -> None:
#     """Entry point for local and federated ViT training on CIFAR-10."""
#     OmegaConf.resolve(cfg)
#     config: TrainConfig = instantiate(cfg)
#     resolved_config = _resolve_config_paths(config)
#     resolved_config.output_dir.mkdir(parents=True, exist_ok=True)

#     logger.info(pformat(resolved_config))

#     world_size = resolved_config.num_clients if resolved_config.mode == "federated" else 1
#     ctx = DistributedContext(world_size=world_size)
#     ctx.launch(_driver, resolved_config)

from hydra.core.hydra_config import HydraConfig


def _driver(ctx: DistributedContext, cfg: DictConfig) -> float | None:
    """Driver function executed per rank (and in the parent when world_size=1)."""
    # Build the runtime config here, after Hydra main, inside this process.
    config: TrainConfig = instantiate(cfg)
    resolved_config = _resolve_config_paths(config)
    resolved_config.output_dir.mkdir(parents=True, exist_ok=True)

    if ctx.rank != 0:
        for h in list(logging.getLogger().handlers):
            if isinstance(h, logging.FileHandler):
                logging.getLogger().removeHandler(h)

    set_seed(resolved_config.seed)

    ctx.wandb_init(resolved_config)
    try:
        if resolved_config.mode == "local":
            return _main_local(resolved_config, ctx)
        return _main_federated(resolved_config, ctx)
    finally:
        ctx.wandb_finish()


def _get_hydra_logging_cfg() -> dict:
    """Load Hydra's active job_logging dictConfig from the run directory."""
    hydra_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra" / "hydra.yaml"
    hydra_cfg = OmegaConf.load(hydra_cfg_path)
    job_logging_cfg = OmegaConf.to_container(hydra_cfg.hydra.job_logging, resolve=True)
    return job_logging_cfg


@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for local and federated ViT training on CIFAR-10."""
    install_exception_handlers(threading_=True, asyncio_=True)

    OmegaConf.resolve(cfg)

    # Install our global logging context in the hydra parent
    install_global_log_context()
    set_log_context(rank="main", world_size=1, backend="-")  # default until spawn workers override

    mode = cfg.mode
    num_clients = cfg.num_clients
    world_size = num_clients if mode == "federated" else 1
    log_cfg = _get_hydra_logging_cfg()

    ctx = DistributedContext(world_size=world_size)

    try:
        ctx.launch(_driver, cfg, log_cfg=log_cfg)
    except BaseException:
        logger.exception("Top-level training failure")
        raise


if __name__ == "__main__":
    main()
