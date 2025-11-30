"""Training utilities for vision transformer models on federated data."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

from . import utils
from .data import Batch, build_dataloaders, load_cifar10_subsets
from .evaluate import evaluate
from .permute import permute_vit_layer_heads
from .utils import create_vit_classifier, default_device, get_amp_settings, set_seed

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


def _try_init_distributed(cfg: TrainConfig) -> tuple[int, int]:
    """Initialize torch.distributed (env://) and return (rank, world_size).

    Requires launch under torchrun (or similar). Also checks that world_size
    matches cfg.num_clients.
    """
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != cfg.num_clients:
        msg = (
            f"Configured num_clients ({cfg.num_clients}) does not match "
            f"distributed world_size ({world_size})"
        )
        raise RuntimeError(msg)

    logger.info("Initialized distributed: rank %d / %d", rank, world_size)
    return rank, world_size


def _select_device(rank: int) -> torch.device:
    """Return the device for the given rank."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return default_device()


def _initialize_server_and_matcher(
    cfg: TrainConfig,
    model: ViTForImageClassification,
    rank: int,
    device: torch.device,
) -> tuple[ViTForImageClassification | None, Matcher | None]:
    """Initialize server model (on rank 0) and matcher."""
    matcher = cfg.matcher
    server_model: ViTForImageClassification | None = None
    if rank == 0:
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
    rank: int,
    device: torch.device,
) -> None:
    """Broadcast initial global model from server (rank 0) to all clients."""
    global_state_list: list[dict | None] = [None]
    if rank == 0:
        assert server_model is not None
        global_state_list[0] = {k: v.cpu() for k, v in server_model.state_dict().items()}
    dist.broadcast_object_list(global_state_list, src=0)
    initial_state = global_state_list[0]
    assert initial_state is not None
    model.load_state_dict(initial_state)
    model.to(device)


def _aggregate_and_evaluate_round(
    cfg: TrainConfig,
    model: ViTForImageClassification,
    server_model: ViTForImageClassification | None,
    matcher: Matcher | None,
    eval_dataloader: DataLoader[Batch],
    device: torch.device,
    rank: int,
    world_size: int,
    gathered: list[dict | None],
    round_idx: int,
    server_best_accuracy: float,
) -> tuple[float | None, torch.Tensor | None, float]:
    """Aggregate client models on rank 0, evaluate, and log artifacts."""
    final_accuracy: float | None = None
    final_confmat: torch.Tensor | None = None

    if rank == 0:
        assert server_model is not None
        logger.info("Aggregating %d client models on rank 0 (round %d)", len(gathered), round_idx + 1)
        aggregated_state = aggregate_models(
            [g for g in gathered if g is not None],
            server_model.config,
            device=device,
            matcher=matcher,
        )
        server_model.load_state_dict(aggregated_state)

        server_model.to(device)
        acc, conf = evaluate(server_model, eval_dataloader, device, enable_bf16=cfg.use_bf16)
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

            artifact = wandb.Artifact("best-server-model", type="model")
            artifact.add_file(best_ckpt_path)
            wandb.log_artifact(artifact)

        wandb.log(
            {
                "round/number": round_idx + 1,
                "round/server_accuracy": final_accuracy,
                "round/server_best_accuracy": server_best_accuracy,
            },
            step=round_idx + 1,
        )

    return final_accuracy, final_confmat, server_best_accuracy


def _broadcast_aggregated_state(
    model: ViTForImageClassification,
    server_model: ViTForImageClassification | None,
    rank: int,
    device: torch.device,
) -> None:
    """Broadcast the aggregated server state to all clients."""
    aggregated_state: dict | None = None
    if rank == 0:
        assert server_model is not None
        aggregated_state = {k: v.cpu() for k, v in server_model.state_dict().items()}

    state_list: list[dict | None] = [aggregated_state]
    dist.broadcast_object_list(state_list, src=0)
    global_state = state_list[0]
    assert global_state is not None
    model.load_state_dict(global_state)
    model.to(device)


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

        progress = tqdm(dataloader, desc=epoch_name)

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
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                wandb.log(
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

                if avg_loss < training_metadata.best_loss:  # new best
                    training_metadata.best_loss = avg_loss
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
                    artifact = wandb.Artifact("best-model", type="model")
                    artifact.add_file(ckpt_path.as_posix())
                    wandb.log_artifact(artifact)

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
            training_metadata=training_metadata,
            epoch_metadata={"epoch": epoch},
            epoch_name=f"Epoch {epoch_padded}/{cfg.epochs}",
        )
        metrics.extend(epoch_metrics)

    wandb.summary["final/loss"] = metrics[-1]["loss"]
    wandb.summary["best/loss"] = training_metadata.best_loss

    return metrics


def _main_local(cfg: TrainConfig) -> float:
    """Run local (single-process) training."""
    device = default_device()
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
    metrics_train = train(model, train_dataloader, device, cfg)
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
    wandb.summary["eval/accuracy"] = accuracy
    wandb.save(ckpt_path.as_posix())

    return float(accuracy)


def _main_federated(cfg: TrainConfig) -> float | None:
    """Run distributed FedAvg-style training using torch.distributed."""
    rank, world_size = _try_init_distributed(cfg)
    device = _select_device(rank)
    logger.info("Using device: %s (rank %d/%d)", device, rank, world_size)

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
    train_dataloader = client_dataloaders[rank]

    server_model, matcher = _initialize_server_and_matcher(cfg, model, rank, device)
    _broadcast_initial_state(model, server_model, rank, device)

    all_train_metrics: Metrics = []
    server_best_accuracy = float("-inf")
    final_accuracy: float | None = None
    final_confmat = None

    for round_idx in range(cfg.num_rounds):
        logger.info("Starting communication round %d/%d", round_idx + 1, cfg.num_rounds)
        round_metrics = train(model, train_dataloader, device, cfg)
        all_train_metrics.extend(round_metrics)

        local_state = {k: v.cpu() for k, v in model.state_dict().items()}
        gathered: list[dict | None] = [None] * world_size
        dist.all_gather_object(gathered, local_state)

        final_accuracy, final_confmat, server_best_accuracy = _aggregate_and_evaluate_round(
            cfg=cfg,
            model=model,
            server_model=server_model,
            matcher=matcher,
            eval_dataloader=eval_dataloader,
            device=device,
            rank=rank,
            world_size=world_size,
            gathered=gathered,
            round_idx=round_idx,
            server_best_accuracy=server_best_accuracy,
        )

        _broadcast_aggregated_state(model, server_model, rank, device)

    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(all_train_metrics)

    if rank == 0 and final_accuracy is not None:
        metrics["eval"]["accuracy"] = final_accuracy
        _save_federated_artifacts(cfg, model, final_confmat, metrics)
        wandb.summary["eval/accuracy"] = final_accuracy

    return final_accuracy


@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for local and federated ViT training on CIFAR-10."""
    OmegaConf.resolve(cfg)
    config: TrainConfig = instantiate(cfg)
    resolved_config = _resolve_config_paths(config)
    resolved_config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(resolved_config.seed)

    logger.info(pformat(resolved_config))

    run = wandb.init(
        entity="fedmat-team",
        project="fedmat-project",
        name=resolved_config.run_name,
        config=asdict(resolved_config),
        mode="online",
        dir=resolved_config.output_dir.as_posix(),
    )
    try:
        _ = _main_local(resolved_config) if resolved_config.mode == "local" else _main_federated(resolved_config)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
