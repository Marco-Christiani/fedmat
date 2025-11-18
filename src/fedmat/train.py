from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict

import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

from fedmat import utils
from fedmat.data import build_dataloaders, load_cifar10_subsets
from fedmat.evaluate import evaluate
from fedmat.utils import default_device, get_amp_settings, set_seed

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    """Config for ViT fine-tuning on CIFAR-10."""

    model_name: str = "google/vit-base-patch16-224-in21k"
    num_labels: int = 10
    train_homogeneity: float = 1.0
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42

    max_train_samples: int | None = 8_000
    max_eval_samples: int | None = 1_000

    num_workers: int = 4
    prefetch_factor: int = 4
    log_every_n_steps: int = 50

    use_bf16: bool = True
    use_torch_compile: bool = False

    output_dir: Path = Path("outputs/vit_cifar10")
    use_pretrained: bool = True


class MetricRecord(TypedDict):
    epoch: int
    step: int
    loss: float

Metrics = list[MetricRecord]

def train(
    model: ViTForImageClassification,
    train_batches: Iterable[dict[str, Tensor]],
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
    global_step = 0
    metrics =[]
    for epoch in range(cfg.epochs):
        running_loss: float = 0.0
        step_count: int = 0

        # tqdm over a known-length iterable if available
        train_iter = train_batches
        if hasattr(train_batches, "__len__"):
            train_iter = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for step, batch in enumerate(train_iter):
            step_count += 1
            global_step += 1

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            ):
                outputs = model(
                    pixel_values=pixel_values,  # [B, C, H, W]
                    labels=labels,              # [B]
                )
                loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % cfg.log_every_n_steps == 0:
                loss_val = float(loss.detach().cpu())
                metrics.append({
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss": loss_val,
                })
                running_loss += loss_val
                avg_loss = running_loss / ((running_loss != 0 and (step + 1) / cfg.log_every_n_steps) or 1.0)
                if isinstance(train_iter, tqdm):
                    train_iter.set_postfix(loss=f"{avg_loss:.4f}")
    return metrics

def main():
    """Vanilla train ViT on CIFAR-10."""
    parser = ArgumentParser(
            prog='fedmat_train',
            description='Train a round of FedMAT',
            epilog='Copyright 2025 (TM) OC do not steal')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch-size', type=int, default=128)
    parser.add_argument('-mts', '--max-train-samples', type=int)
    parser.add_argument('-mes', '--max-eval-samples', type=int)
    parser.add_argument('-nw', '--num-workers', type=int, default=4)
    parser.add_argument('-pf', '--prefetch-factor', type=int, default=4)
    parser.add_argument('-bf16', '--use-bf16', action='store_true')
    parser.add_argument('-tc', '--use-torch-compile', action='store_true')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path("outputs/vit_cifar10"))
    parser.add_argument('-pre', '--use-pretrained', action='store_true')
    parser.add_argument('-alpha', '--train-homogeneity', type=float, default=1.0)
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    logger.info(pformat(cfg))

    if cfg is None:
        cfg = TrainConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = default_device()
    logger.info(f"Using device: {device}")

    train_ds, eval_ds  = load_cifar10_subsets(max_train_samples=cfg.max_train_samples, max_eval_samples=cfg.max_eval_samples)

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

    _ = model.to(device) # pyright: ignore[reportArgumentType] upstream typing issue

    if cfg.use_torch_compile:
        # model = torch.compile(model)
        model.compile()

    train_batches, eval_batches = build_dataloaders(
        train_ds=train_ds,
        train_homogeneity=cfg.train_homogeneity,
        eval_ds=eval_ds,
        image_processor=image_processor,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        device=device,
    )

    metrics_train = train(model, train_batches, device, cfg)
    metrics = defaultdict(dict)
    metrics["train"] = utils.aos_to_soa(metrics_train)

    accuracy, confmat = evaluate(model, eval_batches, device, enable_bf16=cfg.use_bf16)
    metrics["eval"]["accuracy"] = accuracy
    now = datetime.now().isoformat()
    logger.info(f"metrics:\n {metrics!s}")

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
    return accuracy


if __name__ == "__main__":
    main()

