"""Data loading and partitioning utilities for federated learning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sized
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.distributions import Categorical, Dirichlet
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Callable, Literal

    from fedmat.train_utils import WandbQuiver

    from transformers import AutoImageProcessor

Batch = dict[str, Tensor]

def _load_cifar10_subsets():
    raw_ds = load_dataset("uoft-cs/cifar10", num_proc=32)

    train_ds = raw_ds["train"].rename_column("img", "image")
    eval_ds = raw_ds["test"].rename_column("img", "image")

    return train_ds, eval_ds


def _load_imagenet1k_subsets():
    raw_ds = load_dataset("ILSVRC/imagenet-1k", num_proc=32)

    train_ds = raw_ds["train"]
    eval_ds = raw_ds["validation"]

    return train_ds, eval_ds


def _apply_augmentation(augmentation: Callable, examples: dict[str, Any]) -> dict[str, Any]:
    if "image" not in examples:  # e.g. we are doing some kind of label-only transform
        return examples

    images = [np.asarray(image.convert("RGB")) for image in examples["image"]]
    labels = [int(label) for label in examples["label"]]

    outputs = [augmentation(image=image, category=label) for image, label in zip(images, labels)]

    examples["image"] = [output["image"] for output in outputs]
    examples["label"] = [output["category"] for output in outputs]

    return examples


def load_named_dataset_subsets(
    dataset_name: Literal["cifar10", "imagenet1k"],
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    train_augmentation: Callable | None = None,
) -> tuple[Dataset, Dataset]:
    """Load some dataset with name dataset_name and apply optional subset selection."""
    if dataset_name == "cifar10":
        train_ds, eval_ds = _load_cifar10_subsets()
    elif dataset_name == "imagenet1k":
        train_ds, eval_ds = _load_imagenet1k_subsets()
    else:
        raise ValueError(f"dataset_name must be one of 'cifar10' or 'imagenet1k', got '{dataset_name}'")

    if max_train_samples is not None:
        train_ds = train_ds.select(range(max_train_samples), keep_in_memory=True)
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(max_eval_samples), keep_in_memory=True)

    if train_augmentation is not None:
        train_ds = train_ds.with_transform(partial(_apply_augmentation, train_augmentation))

    assert isinstance(train_ds, Dataset)
    assert isinstance(eval_ds, Dataset)

    return train_ds, eval_ds


def build_dataloaders(
    train_ds: Dataset,
    homogeneity: float,
    num_clients: int,
    eval_ds: Dataset,
    image_processor: AutoImageProcessor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    drop_last: bool = False,
) -> tuple[list[DataLoader[Batch]], DataLoader[Batch], Tensor]:
    """Create DataLoaders and wrap them with CUDA prefetching when applicable."""
    collate_fn = Collator(image_processor)
    pin_memory = device.type == "cuda"

    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }

    # Only pass prefetch_factor when using workers
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = prefetch_factor

    samplers, client_histograms = partition_by_client(
        (dict["label"] for dict in train_ds.select_columns("label")),
        num_clients,
        homogeneity,
    )
    train_loaders = [
        DataLoader(
            train_ds,
            sampler=sampler,
            **common_kwargs,
        )
        for sampler in samplers
    ]
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **common_kwargs,
    )

    assert isinstance(train_loaders[0], Sized)
    assert isinstance(eval_loader, Sized)

    return train_loaders, eval_loader, client_histograms


class Collator:
    """Image batch collator for processing images with a given processor."""

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        augmentation: Callable | None = None,
    ) -> None:
        """Initialize the collator with an image processor.

        Parameters
        ----------
        image_processor : AutoImageProcessor
            Processor to apply to images (e.g., from transformers)
        """
        self.image_processor = image_processor
        self.augmentation = augmentation

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Process a batch of examples.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of examples, each with 'image' and 'label' keys

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'pixel_values' and 'labels' tensors
        """
        images = [example["image"] for example in batch]
        labels = [example["label"] for example in batch]

        encodings = self.image_processor(images=images, return_tensors="pt")
        # encodings["pixel_values"]: [B, C, H, W]

        labels = torch.as_tensor(labels, dtype=torch.long)  # [B]
        return {
            "pixel_values": encodings["pixel_values"],
            "labels": labels,
        }


def partition_by_client(labels: Iterator[int], num_clients: int, alpha: float = 1.0) -> tuple[list[Sampler], Tensor]:
    """Partition dataset labels among clients using Dirichlet distribution.

    Parameters
    ----------
    labels : Iterator[int]
        Iterator of class labels
    num_clients : int
        Number of clients to partition data among
    alpha : float, optional
        Concentration parameter for Dirichlet distribution, by default 1.0

    Returns
    -------
    list[Sampler]
        List of samplers, one for each client
    """
    class_indices = partition_by_labels(labels)
    dirichlet = Dirichlet(alpha * torch.ones(num_clients) / num_clients)
    client_indices = [[] for _ in range(num_clients)]
    client_histograms = [[0 for _ in class_indices] for _ in range(num_clients)]
    for class_idx, indices in enumerate(class_indices):
        categorical = Categorical(dirichlet.sample())
        assignments = categorical.sample((len(indices),))
        for idx_within_class, assignment in enumerate(assignments):
            client_idx = int(assignment.item())
            client_histograms[client_idx][class_idx] += 1
            client_indices[client_idx].append(indices[idx_within_class])

    client_histograms = torch.as_tensor(client_histograms)

    return [SubsetRandomSampler(indices) for indices in client_indices], client_histograms


def partition_by_labels(labels: Iterator[int]) -> list[list[int]]:
    """Group data indices by their class labels.

    Parameters
    ----------
    labels : Iterator[int]
        Iterator of class labels

    Returns
    -------
    list[list[int]]
        List of lists, where each inner list contains indices for a class
    """
    class_indices = defaultdict(list)
    num_classes = 0
    for i, class_ in enumerate(labels):
        class_indices[class_].append(i)
        num_classes = max(num_classes, class_ + 1)
    class_indices = [class_indices[i] for i in range(num_classes)]
    return class_indices


class DirichletSampler(Sampler):
    """Sampler that uses Dirichlet distribution for class-balanced sampling."""

    def __init__(self, class_indices: list[list[int]], alpha: float = 0.0) -> None:
        """Initialize the sampler with class indices and concentration parameter.

        Parameters
        ----------
        class_indices : list[list[int]]
            List of indices grouped by class
        alpha : float, optional
            Concentration parameter for Dirichlet distribution, by default 0.0
        """
        super().__init__()

        num_classes = len(class_indices)
        class_sizes = torch.as_tensor([len(indices) for indices in class_indices])
        dataset_size = torch.sum(class_sizes)
        concentration = class_sizes / dataset_size
        dirichlet = Dirichlet(alpha * concentration)

        self.class_indices = [
            torch.as_tensor(class_indices[i])[torch.randperm(class_sizes[i])] for i in range(num_classes)
        ]
        self.class_sizes = class_sizes
        self.dataset_size = dataset_size
        self.categorical = Categorical(dirichlet.sample())

    def __iter__(self) -> Iterator[int]:
        """Iterate over shuffled indices sampled from classes via Dirichlet distribution."""
        class_indices = [indices[torch.randperm(len(indices))].tolist() for indices in self.class_indices]
        for _ in range(len(self)):
            class_ = self.categorical.sample()
            yield class_indices[class_].pop()

    def __len__(self) -> int:
        """Return the minimum class size to ensure balanced sampling."""
        return torch.min(self.class_sizes)
