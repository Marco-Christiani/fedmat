"""Data loading and partitioning utilities for federated learning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sized
from typing import TYPE_CHECKING

import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.distributions import Categorical, Dirichlet
from torch.utils.data import DataLoader, RandomSampler, Sampler

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal

    from transformers import AutoImageProcessor

Batch = dict[str, Tensor]

def _load_cifar10_subsets(
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    raw_ds = load_dataset("uoft-cs/cifar10", num_proc=32)

    train_ds = raw_ds["train"]
    eval_ds = raw_ds["test"]

    if max_train_samples is not None:
        train_ds = train_ds.select(range(max_train_samples), keep_in_memory=True)
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(max_eval_samples), keep_in_memory=True)

    return train_ds, eval_ds

def _load_imagenet1k_subsets(
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    raw_ds = load_dataset("ILSVRC/imagenet-1k", num_proc=32)

    train_ds = raw_ds["train"].rename_column("image", "img")
    eval_ds = raw_ds["validation"].rename_column("image", "img")

    if max_train_samples is not None:
        train_ds = train_ds.select(range(max_train_samples), keep_in_memory=True)
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(max_eval_samples), keep_in_memory=True)

    return train_ds, eval_ds


def load_named_dataset_subsets(
    dataset_name: Literal["cifar10", "imagenet1k"],
    **kwargs,
) -> tuple[Dataset | IterableDataset, Dataset | IterableDataset]:
    """Load some dataset with name dataset_name and apply optional subset selection."""

    if dataset_name == "cifar10":
        train_ds, eval_ds = _load_cifar10_subsets(**kwargs)
    elif dataset_name == "imagenet1k":
        train_ds, eval_ds = _load_imagenet1k_subsets(**kwargs)
    else:
        raise ValueError(f"dataset_name must be one of 'cifar10' or 'imagenet1k', got '{dataset_name}'")

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
) -> tuple[list[DataLoader[Batch]], DataLoader[Batch]]:
    """Create DataLoaders and wrap them with CUDA prefetching when applicable."""
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

    train_loaders = [
        DataLoader(
            train_ds,
            sampler=sampler,
            **common_kwargs,
        )
        for sampler in partition_by_client(
            (dict["label"] for dict in train_ds.select_columns("label")),
            num_clients,
            homogeneity,
        )
    ]
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **common_kwargs,
    )
    assert isinstance(train_loaders[0], Sized)
    assert isinstance(eval_loader, Sized)
    return train_loaders, eval_loader


class Collator:
    """Image batch collator for processing images with a given processor."""

    def __init__(
        self,
        image_processor: AutoImageProcessor,
    ) -> None:
        """Initialize the collator with an image processor.

        Parameters
        ----------
        image_processor : AutoImageProcessor
            Processor to apply to images (e.g., from transformers)
        """
        self.image_processor = image_processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Process a batch of examples.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            List of examples, each with 'img' and 'label' keys

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with 'pixel_values' and 'labels' tensors
        """
        images = [example["img"].convert("RGB") for example in batch]
        labels_list = [int(example["label"]) for example in batch]

        encodings = self.image_processor(images=images, return_tensors="pt")
        # encodings["pixel_values"]: [B, C, H, W]

        labels = torch.as_tensor(labels_list, dtype=torch.long)  # [B]
        return {
            "pixel_values": encodings["pixel_values"],
            "labels": labels,
        }


def partition_by_client(labels: Iterator[int], num_clients: int, alpha: float = 1.0) -> list[Sampler]:
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
    for indices in class_indices:
        categorical = Categorical(dirichlet.sample())
        assignments = categorical.sample((len(indices),))
        for idx_within_class, assignment in enumerate(assignments):
            client_indices[int(assignment.item())].append(indices[idx_within_class])
    return [RandomSampler(indices) for indices in client_indices]


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
