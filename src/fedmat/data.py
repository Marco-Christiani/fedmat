from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from torch import Tensor
    from transformers import AutoImageProcessor


def load_cifar10_subsets(
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[Dataset, Dataset]:
    """Load CIFAR-10 and apply optional subset selection."""

    raw_ds = load_dataset("cifar10")

    train_ds: Dataset = raw_ds["train"]
    eval_ds: Dataset = raw_ds["test"]

    if max_train_samples is not None:
        train_ds = train_ds.select(range(max_train_samples), keep_in_memory=True)

    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(max_eval_samples), keep_in_memory=True)

    return train_ds, eval_ds


def build_dataloaders(
    train_ds: Dataset,
    eval_ds: Dataset,
    image_processor: AutoImageProcessor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[Iterable[dict[str, Tensor]], Iterable[dict[str, Tensor]]]:
    """Create DataLoaders and wrap them with CUDA prefetching when applicable. """

    collate_fn = Collator(image_processor)
    pin_memory = device.type == "cuda"

    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Only pass prefetch_factor when using workers
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common_kwargs,
    )
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, eval_loader

class Collator:
    def __init__(
        self,
        image_processor: AutoImageProcessor,
    ):  #-> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
        self.image_processor = image_processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [example["img"] for example in batch]
        labels_list = [int(example["label"]) for example in batch]

        encodings = self.image_processor(images=images, return_tensors="pt")
        # encodings["pixel_values"]: [B, C, H, W]

        labels = torch.as_tensor(labels_list, dtype=torch.long)  # [B]
        return {
            "pixel_values": encodings["pixel_values"],
            "labels": labels,
        }