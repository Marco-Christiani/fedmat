from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, TypedDict, Literal

import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTConfig, ViTForImageClassification

import wandb

from . import utils
from .data import Batch, build_dataloaders, load_cifar10_subsets
from .evaluate import evaluate
from .utils import default_device, get_amp_settings, set_seed
from .fedutils import reduce, torch2reduce, replicate

if TYPE_CHECKING:
    from typing import Tuple

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
    aggregation: Literal["avg", "mat"] = "avg"
    local_iterations: int = 2


def train_epoch(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
    epoch_name: str = "",
):
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_autocast, amp_dtype = get_amp_settings(device, cfg.use_bf16)

    n_batches = len(dataloader)
    progress = tqdm(dataloader, desc=epoch_name)

    for step, batch in enumerate(progress):
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

def train(
    model: ViTForImageClassification,
    dataloader: DataLoader[Batch],
    device: torch.device,
    cfg: TrainConfig,
):
    for epoch in range(cfg.epochs):
        epoch_name_padding = " " * (len(str(cfg.epochs)) - len(str(epoch)))
        epoch_padded = epoch_name_padding + str(epoch)
        train_epoch(
            model, dataloader, device, cfg,
            epoch_name=f"Epoch {epoch_padded}/{cfg.epochs}"
        )

def train_fedavg(
    model: ViTForImageClassification,
    dataloaders: List[DataLoader[Batch]],
    device: torch.device,
    cfg: TrainConfig,
):
    C = len(dataloaders)

    client_models = None

    for epoch in range(cfg.epochs): # communication rounds
        epoch_name = str(epoch+1)
        epoch_name_padding = " " * (len(str(cfg.epochs)) - len(epoch_name))
        epoch_padded = epoch_name_padding + epoch_name

        client_models = replicate(model, client_models or C) # replicate server onto clients

        for client_id in range(C):
            client_name = str(client_id+1)
            client_name_padding = " " * (len(str(C)) - len(client_name))
            client_padded = client_name_padding + client_name

            client_model = client_models[client_id]
            client_dataloader = dataloaders[client_id]

            for local in range(cfg.local_iterations): # local iterations
                local_name = str(local+1)
                local_name_padding = " " * (len(str(cfg.local_iterations)) - len(local_name))
                local_padded = local_name_padding + local_name

                train_epoch(
                    client_model, client_dataloader, device, cfg,
                    epoch_name=(
                        f"Comm. Round {epoch_padded}/{cfg.epochs} | "
                        f"Client {client_padded}/{C} | "
                        f"Local Iter. {local_padded}/{cfg.local_iterations}"
                        )
                )

        reduce(client_models, torch2reduce(torch.mean), model)

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
    parser.add_argument("-c", "--num-clients", type=int, default=1) # 1 means regular SGD, no FL
    parser.add_argument("-agg", "--aggregation", default="avg", choices = ["avg", "mat"])
    parser.add_argument("-li", "--local-iterations", type=int, default=2)

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    logger.info(pformat(cfg))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

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

    # Train
    if cfg.num_clients == 1:
        train(model, client_dataloaders[0], device, cfg)
    elif cfg.aggregation == "avg":
        train_fedavg(model, client_dataloaders, device, cfg)
    elif cfg.aggregation == "mat":
        raise NotImplementedError("fedmat aggregation is still WIP")

    # Eval
    accuracy, confmat = evaluate(model, eval_dataloader, device, enable_bf16=cfg.use_bf16)

    # Save
    now = datetime.now().isoformat()

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
