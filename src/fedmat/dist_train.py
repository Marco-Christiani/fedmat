"""Training utilities for vision transformer models on federated data."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict
import os
import copy

import torch.distributed as dist

import torch
import wandb
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

from . import utils
from .data import Batch, build_dataloaders, load_cifar10_subsets
from .evaluate import evaluate
from .utils import default_device, get_amp_settings, set_seed

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Config for ViT fine-tuning on CIFAR-10."""

    use_wandb: bool = True
    run_name: str | None = None
    seed: int = 42

    model_name: str = "google/vit-base-patch16-224-in21k"
    use_pretrained: bool = True
    num_labels: int = 10

    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 5
    max_grad_norm: float | None = None
    log_every_n_steps: int = 50
    use_bf16: bool = True
    use_torch_compile: bool = False

    max_train_samples: int | None = 8_000
    max_eval_samples: int | None = 1_000
    num_workers: int = 4
    prefetch_factor: int = 4

    output_dir: Path = Path("outputs/vit_cifar10")

    homogeneity: float = 1.0
    num_clients: int = 1
    # Number of communication rounds. `epochs` is interpreted as epochs per round.
    num_rounds: int = 1


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
                if cfg.use_wandb:
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
                    if cfg.use_wandb:
                        artifact = wandb.Artifact("best-model", type="model")
                        artifact.add_file(ckpt_path)
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

    if cfg.use_wandb:
        wandb.summary["final/loss"] = metrics[-1]["loss"]
        wandb.summary["best/loss"] = training_metadata.best_loss
    return metrics


def _init_distributed_if_needed(cfg: TrainConfig) -> tuple[int, int]:
    """Initialize torch.distributed if running under torchrun/env and return (rank, world_size).

    Uses environment init ('env://'). If not running distributed, returns (0, 1).
    """
    if cfg.num_clients <= 1:
        return 0, 1

    if not dist.is_available():
        logger.warning("torch.distributed not available; falling back to single-process mode")
        return 0, 1

    # init via environment (torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception as e:
            logger.warning("Failed to init process group: %s. Running single-process.", e)
            return 0, 1

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info("Initialized distributed: rank %d / %d", rank, world_size)
    return rank, world_size


def aggregate_models_on_rank0(state_dicts: list[dict]) -> dict:
    if not state_dicts:
        raise ValueError("No state dicts to aggregate")

    # Copy first as template
    agg = {}
    keys = list(state_dicts[0].keys())
    with open("debug_agg.txt", "w") as f:
        f.write(f"Aggregating {len(state_dicts)} models with keys: {keys}\n")
    for k in keys:
        # stack tensors from all clients and average
        tensors = [sd[k].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)
        agg[k] = (stacked.mean(dim=0))

    return agg


def main() -> None:
    """Vanilla train ViT on CIFAR-10."""
    parser = ArgumentParser(
        prog="fedmat_train",
        description="Train a round of FedMAT",
        epilog="Copyright 2025 (TM) OC do not steal",
    )
    parser.add_argument("-e", "--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("-bs", "--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("-mts", "--max-train-samples", type=int)
    parser.add_argument("-mes", "--max-eval-samples", type=int)
    parser.add_argument("-nw", "--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("-pf", "--prefetch-factor", type=int, default=TrainConfig.prefetch_factor)
    parser.add_argument("-bf16", "--use-bf16", action="store_true")
    parser.add_argument("-tc", "--use-torch-compile", action="store_true")
    parser.add_argument("-o", "--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("-pre", "--use-pretrained", action="store_true")
    parser.add_argument("-rn", "--run-name", type=str, default=TrainConfig.run_name)
    parser.add_argument("-alpha", "--homogeneity", type=float, default=1.0)
    parser.add_argument("-c", "--num-clients", type=int, default=1)
    parser.add_argument("-r", "--num-rounds", type=int, default=TrainConfig.num_rounds,
                        help="Number of communication rounds; epochs are per-round")

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    logger.info(pformat(cfg))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    if cfg.use_wandb:
        import wandb

        run = wandb.init(
            entity="fedmat-team",
            project="fedmat-project",
            name=cfg.run_name,
            config=asdict(cfg),
            mode="online",
            dir=str(cfg.output_dir),  # wandb files in run dir for now
        )
    else:
        run = None

    # Initialize distributed (if running with torchrun and cfg.num_clients>1)
    rank, world_size = _init_distributed_if_needed(cfg)

    # Choose device per-process. If CUDA available and torchrun provided LOCAL_RANK, prefer that.
    if torch.cuda.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = default_device()

    logger.info(f"Using device: {device} (rank {rank}/{world_size})")

    train_ds, eval_ds = load_cifar10_subsets(
        max_train_samples=cfg.max_train_samples, max_eval_samples=cfg.max_eval_samples
    )

    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    if cfg.use_pretrained:
        logger.info("Using pretrained")
        model = ViTForImageClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    else:
        logger.info("Not using pretrained")
        model: ViTForImageClassification = ViTForImageClassification(
            ViTConfig(
                num_labels=cfg.num_labels,
            )
        )

    logger.info("Model:\n%s", pformat(model))

    model.config.id2label = {i: str(i) for i in range(cfg.num_labels)}
    model.config.label2id = {str(i): i for i in range(cfg.num_labels)}

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
    # Each process selects a client dataloader based on its rank (simulates federated clients).
    if world_size > 1:
        if rank < len(client_dataloaders):
            train_dataloader = client_dataloaders[rank]
        else:
            logger.warning(
                "Rank %d >= number of client_dataloaders (%d). Falling back to index 0.",
                rank,
                len(client_dataloaders),
            )
            train_dataloader = client_dataloaders[0]
    else:
        train_dataloader = client_dataloaders[0]

    # Keep a non-distributed server model copy on rank 0 for aggregation
    server_model = None
    if world_size > 1 and rank == 0:
        # create a fresh model instance for the server copy and load current params
        if cfg.use_pretrained:
            server_model = ViTForImageClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
        else:
            server_model = ViTForImageClassification(ViTConfig(num_labels=cfg.num_labels))
        # load state from local model (ensure on CPU for safe aggregation)
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        server_model.load_state_dict(cpu_state)
        server_model.to("cpu")

    # Communication rounds: clients train locally for `cfg.epochs` each round,
    # then rank 0 aggregates and broadcasts the global model back to clients.
    all_train_metrics: list[MetricRecord] = []
    all_round_eval_metrics: list[dict] = []
    for round_idx in range(cfg.num_rounds):
        logger.info("Starting communication round %d/%d", round_idx + 1, cfg.num_rounds)

        # Before local training, have clients initialize from the server model (rank 0)
        if world_size > 1 and dist.is_initialized():
            if rank == 0 and server_model is not None:
                server_state = {k: v.cpu() for k, v in server_model.state_dict().items()}
                if hasattr(dist, "broadcast_object_list"):
                    try:
                        dist.broadcast_object_list([server_state], src=0)
                    except Exception as e:
                        logger.warning("broadcast_object_list failed on rank 0: %s", e)
                else:
                    logger.warning("broadcast_object_list not available; clients keep their local weights for round %d", round_idx + 1)
            elif world_size > 1 and hasattr(dist, "broadcast_object_list"):
                try:
                    recv = [None]
                    dist.broadcast_object_list(recv, src=0)
                    if recv[0] is not None:
                        model.load_state_dict(recv[0])
                        model.to(device)
                except Exception as e:
                    logger.warning("broadcast_object_list receive failed on rank %d: %s", rank, e)

        # Local training (epochs per round)
        metrics_train = train(model, train_dataloader, device, cfg)
        if metrics_train:
            all_train_metrics.extend(metrics_train)

        # After local training, gather local states and aggregate on rank 0
        if world_size > 1 and dist.is_initialized():
            local_state = {k: v.cpu() for k, v in model.state_dict().items()}
            gathered = [None] * world_size
            if hasattr(dist, "all_gather_object"):
                try:
                    dist.all_gather_object(gathered, local_state)
                except Exception as e:
                    logger.warning("all_gather_object failed: %s", e)
                    gathered = None
            else:
                logger.warning("torch.distributed.all_gather_object not available; skipping aggregation for round %d", round_idx + 1)
                gathered = None

            if rank == 0 and gathered:
                logger.info("Aggregating %d client models on rank 0 (round %d)", len(gathered), round_idx + 1)
                aggregated_state = aggregate_models_on_rank0(gathered)
                server_model.load_state_dict(aggregated_state)
                server_model.to("cpu")
                server_ckpt = cfg.output_dir / f"server_model-round{round_idx + 1}-{datetime.now().isoformat()}.pt"
                torch.save(server_model.state_dict(), server_ckpt)
                logger.info("Saved server aggregated checkpoint to %s", server_ckpt)

                # Evaluate server model on held-out eval set (rank 0)
                try:
                    server_model.to(device)
                    serv_acc, serv_conf = evaluate(server_model, eval_dataloader, device, enable_bf16=cfg.use_bf16)
                    serv_conf_path = cfg.output_dir / f"server_confusion-round{round_idx + 1}-{datetime.now().isoformat()}.pt"
                    torch.save(serv_conf.cpu(), serv_conf_path)
                    logger.info(
                        "Server eval after round %d: accuracy=%.4f; confusion saved to %s",
                        round_idx + 1,
                        serv_acc,
                        serv_conf_path,
                    )
                    all_round_eval_metrics.append({
                        "round": round_idx + 1,
                        "accuracy": float(serv_acc),
                        "server_ckpt": str(server_ckpt),
                        "confusion_path": str(serv_conf_path),
                    })
                    # Log round metrics to Weights & Biases if enabled
                    if run and rank == 0:
                        try:
                            wandb.log(
                                {
                                    "round/number": round_idx + 1,
                                    "round/server_accuracy": float(serv_acc),
                                    "round/server_ckpt": str(server_ckpt),
                                },
                                step=round_idx + 1,
                            )
                        except Exception:
                            logger.exception("Failed to log round metrics to wandb for round %d", round_idx + 1)
                    # move server back to cpu to keep aggregation safe
                    server_model.to("cpu")
                except Exception as e:
                    logger.warning("Server evaluation failed on rank 0 after round %d: %s", round_idx + 1, e)

                # Broadcast aggregated model to clients for next round
                if hasattr(dist, "broadcast_object_list"):
                    try:
                        dist.broadcast_object_list([aggregated_state], src=0)
                    except Exception as e:
                        logger.warning("broadcast_object_list failed when distributing aggregated model: %s", e)
                else:
                    logger.warning("broadcast_object_list not available; aggregated model not sent to clients")
            elif world_size > 1 and gathered:
                # receive aggregated state from rank 0
                if hasattr(dist, "broadcast_object_list"):
                    try:
                        recv = [None]
                        dist.broadcast_object_list(recv, src=0)
                        if recv[0] is not None:
                            model.load_state_dict(recv[0])
                            model.to(device)
                    except Exception as e:
                        logger.warning("broadcast_object_list receive failed on rank %d: %s", rank, e)

    # Collect and reshape training metrics
    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(all_train_metrics)

    # Eval
    accuracy, confmat = evaluate(model, eval_dataloader, device, enable_bf16=cfg.use_bf16)
    metrics["eval"]["accuracy"] = accuracy

    # Save
    now = datetime.now().isoformat()

    fpath = cfg.output_dir.joinpath(f"metrics-{now}.json")
    logger.info(f"Saving metrics: {fpath!s}")
    with fpath.open("w") as f:
        json.dump(metrics, fp=f, indent=2)

    fpath = cfg.output_dir / f"confusion_matrix-{now}.pt"
    logger.info(f"Saving confusion matrix data to {fpath!s}")
    torch.save(confmat.cpu(), fpath)

    ckpt_path = cfg.output_dir / f"model-{now}.pt"
    logger.info(f"Saving checkpoint to {ckpt_path}")
    torch.save(model.state_dict(), ckpt_path)

    # If running distributed, gather model parameters and perform aggregation on rank 0.
    if world_size > 1 and dist.is_initialized():
        # move state to CPU and gather
        local_state = {k: v.cpu() for k, v in model.state_dict().items()}
        gathered = [None] * world_size
        if hasattr(dist, "all_gather_object"):
            try:
                dist.all_gather_object(gathered, local_state)
            except Exception as e:
                logger.warning("all_gather_object failed: %s", e)
                gathered = None
        else:
            logger.warning("torch.distributed.all_gather_object not available; skipping aggregation placeholder")
            gathered = None

        if rank == 0 and gathered:
            logger.info("Aggregating %d client models on rank 0 (placeholder)", len(gathered))
            aggregated_state = aggregate_models_on_rank0(gathered)
            server_model.load_state_dict(aggregated_state)
            server_ckpt = cfg.output_dir / f"server_model-{now}.pt"
            torch.save(server_model.state_dict(), server_ckpt)
            logger.info("Saved server aggregated checkpoint to %s", server_ckpt)

    logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    if run:
        wandb.summary["eval/accuracy"] = accuracy
        wandb.save(str(ckpt_path))
        run.finish()

    return accuracy


if __name__ == "__main__":
    main()
