"""Training utilities for vision transformer models on federated data (FedAvg)."""

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

import torch
import torch.distributed as dist
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
    """Config for ViT fine-tuning on CIFAR-10 with FedAvg."""

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


def train_epoch(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
    training_metadata: TrainingMetadata,
    epoch_metadata: dict | None = None,
    epoch_name: str = "",
) -> tuple[Metrics, float]:
    """Train model for one epoch."""
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
    metrics_meta: list[MetricRecord] = []

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
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        optimizer.step()

        metrics_loss_gpu[step] = loss.detach()
        metrics_meta.append(
            {
                **epoch_metadata,
                "step": step,
                "global_step": training_metadata.global_step,
                "loss": float("nan"),  # filled below
            }
        )

        running_loss_gpu += loss.detach()
        if training_metadata.global_step % cfg.log_every_n_steps == 0 or step == n_batches - 1:
            avg_loss_gpu = running_loss_gpu / steps_since_log
            progress.set_postfix(loss=f"{avg_loss_gpu.item():.4f}")
            running_loss_gpu.zero_()
            steps_since_log = 0

    # Move losses to CPU and fill metric records
    loss_list_cpu = metrics_loss_gpu.cpu().tolist()
    for rec in metrics_meta:
        rec["loss"] = loss_list_cpu[rec["step"]]

    final_loss = float(loss_list_cpu[-1]) if loss_list_cpu else float("nan")
    return metrics_meta, final_loss


def train(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
) -> Metrics:
    """Train model for multiple epochs and return concatenated per-step metrics."""
    epoch_name_padding = " " * len(str(cfg.epochs))
    metrics: Metrics = []
    training_metadata = TrainingMetadata()

    for epoch in range(cfg.epochs):
        epoch_padded = epoch_name_padding + str(epoch)
        epoch_metrics, _ = train_epoch(
            model,
            dataloader,
            device,
            cfg,
            training_metadata=training_metadata,
            epoch_metadata={"epoch": epoch},
            epoch_name=f"Epoch {epoch_padded}/{cfg.epochs}",
        )
        metrics.extend(epoch_metrics)

    return metrics


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
        raise RuntimeError(
            f"Configured num_clients ({cfg.num_clients}) does not match distributed world_size ({world_size})"
        )

    logger.info("Initialized distributed: rank %d / %d", rank, world_size)
    return rank, world_size


def aggregate_models(state_dicts: list[dict]) -> dict:
    """Average a list of (CPU) state_dicts parameter-wise (FedAvg)."""
    if not state_dicts:
        raise ValueError("No state dicts to aggregate")

    agg: dict[str, torch.Tensor] = {}
    keys = list(state_dicts[0].keys())

    for k in keys:
        tensors = [sd[k].float() for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)
        agg[k] = stacked.mean(dim=0)

    return agg


def main() -> None:
    """Train ViT on CIFAR-10 with simple FedAvg using torch.distributed."""
    parser = ArgumentParser(
        prog="fedmat_train",
        description="Train a round of FedMAT (FedAvg baseline)",
        epilog="Copyright 2025 (TM)",
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
    parser.add_argument("-r", "--num-rounds", type=int, default=TrainConfig.num_rounds)

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    logger.info("Config:\n%s", pformat(cfg))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    rank, world_size = _try_init_distributed(cfg)

    # Optional Weights & Biases (only on rank 0)
    run = None
    if cfg.use_wandb and rank == 0:
        import wandb
        run = wandb.init(
            entity="fedmat-team",
            project="fedmat-project",
            name=cfg.run_name,
            config=asdict(cfg),
            mode="online",
            dir=str(cfg.output_dir),  # wandb files in run dir for now
        )

    # Device setup
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = default_device()

    logger.info("Using device: %s (rank %d/%d)", device, rank, world_size)

    # Data
    train_ds, eval_ds = load_cifar10_subsets(
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    # Model (client copy on each rank)
    if cfg.use_pretrained:
        logger.info("Using pretrained backbone '%s'", cfg.model_name)
        model = ViTForImageClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
        )
    else:
        logger.info("Training from scratch")
        model = ViTForImageClassification(ViTConfig(num_labels=cfg.num_labels))

    model.config.id2label = {i: str(i) for i in range(cfg.num_labels)}
    model.config.label2id = {str(i): i for i in range(cfg.num_labels)}

    model.to(device)
    if cfg.use_torch_compile:
        model.compile()

    # Build per-client dataloaders and select the one for this rank
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

    # Rank 0 keeps a CPU "server" model for aggregation/eval
    server_model: ViTForImageClassification | None = None
    if rank == 0:
        if cfg.use_pretrained:
            server_model = ViTForImageClassification.from_pretrained(
                cfg.model_name,
                num_labels=cfg.num_labels,
            )
        else:
            server_model = ViTForImageClassification(ViTConfig(num_labels=cfg.num_labels))

        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        server_model.load_state_dict(cpu_state)
        server_model.to("cpu")

    # Broadcast initial global model from server (rank 0) to all clients
    global_state_list: list[dict | None] = [None]
    if rank == 0:
        assert server_model is not None
        global_state_list[0] = {k: v.cpu() for k, v in server_model.state_dict().items()}
    dist.broadcast_object_list(global_state_list, src=0)
    initial_state = global_state_list[0]
    assert initial_state is not None
    model.load_state_dict(initial_state)
    model.to(device)

    # FedAvg communication rounds
    all_train_metrics: Metrics = []
    server_best_accuracy = float("-inf")
    final_accuracy: float | None = None
    final_confmat = None
    best_ckpt_path: Path | None = None

    for round_idx in range(cfg.num_rounds):
        logger.info("Starting communication round %d/%d", round_idx + 1, cfg.num_rounds)

        # Local training on each client
        round_metrics = train(model, train_dataloader, device, cfg)
        all_train_metrics.extend(round_metrics)

        # Gather local state dicts (on CPU) from all clients
        local_state = {k: v.cpu() for k, v in model.state_dict().items()}
        gathered: list[dict | None] = [None] * world_size
        dist.all_gather_object(gathered, local_state)

        aggregated_state = None
        if rank == 0:
            assert server_model is not None
            logger.info("Aggregating %d client models on rank 0 (round %d)", len(gathered), round_idx + 1)
            aggregated_state = aggregate_models(gathered)  # type: ignore[arg-type]
            server_model.load_state_dict(aggregated_state)

            # Evaluate aggregated (server) model
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

            # Track best server model
            if final_accuracy > server_best_accuracy:
                server_best_accuracy = final_accuracy
                best_ckpt_path = cfg.output_dir / "best-server.pt"
                torch.save(server_model.state_dict(), best_ckpt_path)
                logger.info("New best server model (acc=%.4f) saved to %s", server_best_accuracy, best_ckpt_path)

                if run is not None:
                    artifact = wandb.Artifact("best-server-model", type="model")
                    artifact.add_file(best_ckpt_path)
                    wandb.log_artifact(artifact)

            # Per-round logging to wandb
            if run is not None:
                wandb.log(
                    {
                        "round/number": round_idx + 1,
                        "round/server_accuracy": final_accuracy,
                        "round/server_best_accuracy": server_best_accuracy,
                    },
                    step=round_idx + 1,
                )

        # Broadcast aggregated state to all clients and load into local model
        state_list: list[dict | None] = [aggregated_state]
        dist.broadcast_object_list(state_list, src=0)
        global_state = state_list[0]
        assert global_state is not None
        model.load_state_dict(global_state)
        model.to(device)

    # Collect metrics (only rank 0 writes to disk)
    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(all_train_metrics)

    if rank == 0 and final_accuracy is not None:
        metrics["eval"]["accuracy"] = final_accuracy

    if rank == 0:
        now = datetime.now().isoformat(timespec="seconds")

        metrics_path = cfg.output_dir / f"metrics-{now}.json"
        logger.info("Saving metrics to %s", metrics_path)
        with metrics_path.open("w") as f:
            json.dump(metrics, fp=f, indent=2)

        if final_confmat is not None:
            conf_path = cfg.output_dir / f"confusion_matrix-{now}.pt"
            logger.info("Saving final confusion matrix to %s", conf_path)
            torch.save(final_confmat.cpu(), conf_path)

        # Save final local (client) model checkpoint as reference
        ckpt_path = cfg.output_dir / f"model-{now}.pt"
        logger.info("Saving final local model checkpoint to %s", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)

    # Finish wandb run (only rank 0)
    if run is not None and rank == 0:
        try:
            if final_accuracy is not None:
                wandb.summary["eval/accuracy"] = final_accuracy
        finally:
            run.finish()


if __name__ == "__main__":
    main()
