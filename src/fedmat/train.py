from __future__ import annotations
from collections import defaultdict
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification, ViTImageProcessor

from fedmat import utils
from fedmat.data import build_dataloaders, load_cifar10_subsets
from fedmat.evaluate import evaluate
from fedmat.utils import get_amp_settings, default_device, set_seed

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    """Config for ViT fine-tuning on CIFAR-10."""

    model_name: str = "google/vit-base-patch16-224-in21k"
    num_labels: int = 10
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
                    "loss": loss_val,
                })
                running_loss += loss_val
                avg_loss = running_loss / ((running_loss != 0 and (step + 1) / cfg.log_every_n_steps) or 1.0)
                if isinstance(train_iter, tqdm):
                    train_iter.set_postfix(loss=f"{avg_loss:.4f}")
    return metrics

def main():
    """Vanilla train ViT on CIFAR-10."""
    cfg = TrainConfig(
        epochs=5,
        batch_size=64,
        max_train_samples=8_000,
        max_eval_samples=1_000,
        num_workers=4,
        prefetch_factor=4,
        use_bf16=True,
        use_torch_compile=False,
        output_dir=Path("outputs/vit_cifar10"),
    )
    logger.info(pformat(cfg))

    if cfg is None:
        cfg = TrainConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = default_device()
    logger.info(f"Using device: {device}")

    train_ds, eval_ds  = load_cifar10_subsets(max_train_samples=cfg.max_train_samples, max_eval_samples=cfg.max_eval_samples)

    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        # ignore_mismatched_sizes=True,  # re-init classification head
        # device_map=device,
    )

    model.config.id2label = {i: str(i) for i in range(cfg.num_labels)}
    model.config.label2id = {str(i): i for i in range(cfg.num_labels)}

    _ = model.to(device) # pyright: ignore[reportArgumentType] upstream typing issue

    if cfg.use_torch_compile:
        # model = torch.compile(model)
        model.compile()

    train_batches, eval_batches = build_dataloaders(
        train_ds=train_ds,
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

    accuracy = evaluate(model, eval_batches, device, enable_bf16=cfg.use_bf16)
    metrics["eval"]["accuracy"] = accuracy
    logger.info(f"Saving metrics: {metrics!s}")
    with cfg.output_dir.joinpath("metrics.json").open("w") as f:
        json.dump(metrics, fp=f, indent=2)

    logger.info(f"Evaluation accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    main()

