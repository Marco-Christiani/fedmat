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

import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

import wandb

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


class MetricRecord(TypedDict):
    epoch: int
    global_step: int
    step: int
    loss: float


Metrics = list[MetricRecord]


def train(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
) -> Metrics:
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_autocast, amp_dtype = get_amp_settings(device, cfg.use_bf16)

    n_batches = len(dataloader)
    total_steps = cfg.epochs * n_batches

    metrics_loss_gpu: Tensor = torch.empty(
        total_steps,
        dtype=torch.float32,
        device=device,
    )

    metrics_meta: list[MetricRecord] = []  # cpu

    global_step = 0

    best_loss: float = float("inf")

    try:
        for epoch in range(cfg.epochs):
            running_loss_gpu: Tensor = torch.zeros((), device=device)
            steps_since_log = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

            for step, batch in enumerate(progress):
                global_step += 1
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

                metrics_loss_gpu[global_step - 1] = loss.detach()
                metrics_meta.append({
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": float("nan"),  # placeholder
                })

                if global_step % cfg.log_every_n_steps == 0:
                    running_loss_gpu += loss.detach()
                    avg_loss_gpu = running_loss_gpu / steps_since_log
                    avg_loss = float(avg_loss_gpu.item())
                    progress.set_postfix(loss=f"{avg_loss:.4f}")
                    if cfg.use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/epoch": epoch,
                                "train/step": step,
                                "train/global_step": global_step,
                                "train/lr": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

                    if avg_loss < best_loss:  # new best
                        best_loss = avg_loss
                        ckpt_path = cfg.output_dir / "best.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "global_step": global_step,
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
            rec["loss"] = loss_list_cpu[rec["global_step"] - 1]

        if cfg.use_wandb:
            wandb.summary["final/loss"] = loss_list_cpu[-1]
            wandb.summary["best/loss"] = best_loss

    return metrics_meta


def main():
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

    device = default_device()
    logger.info(f"Using device: {device}")

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
    logger.info(f"Saving metrics: {fpath!s}")
    with fpath.open("w") as f:
        json.dump(metrics, fp=f, indent=2)

    fpath = cfg.output_dir / f"confusion_matrix-{now}.pt"
    logger.info(f"Saving confusion matrix data to {fpath!s}")
    torch.save(confmat.cpu(), fpath)

    ckpt_path = cfg.output_dir / f"model-{now}.pt"
    logger.info(f"Saving checkpoint to {ckpt_path}")
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    if run:
        wandb.summary["eval/accuracy"] = accuracy
        wandb.save(str(ckpt_path))
        run.finish()

    return accuracy


if __name__ == "__main__":
    main()
