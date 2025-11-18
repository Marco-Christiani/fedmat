from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Sampler, RandomSampler
from torch.distributions import Dirichlet, Categorical, Uniform

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Dict

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
    homogeneity: float,
    num_clients: int,
    eval_ds: Dataset,
    image_processor: AutoImageProcessor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[List[Iterable[dict[str, Tensor]]], Iterable[dict[str, Tensor]]]:
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

    train_loaders = [DataLoader(
        train_ds,
        sampler=sampler,
        **common_kwargs,
    ) for sampler in partition_by_client(
        (train_ds[i]["label"] for i in range(len(train_ds))),
        num_clients,
        homogeneity,
    )]
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        **common_kwargs,
    )
    return train_loaders, eval_loader

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

def partition_by_client(labels: Iterator[int], num_clients: int, alpha: float = 1.0) -> List[Sampler]:
    class_indices = partition_by_labels(labels)
    dirichlet = Dirichlet(alpha * torch.ones(num_clients) / num_clients)
    client_indices = [list() for _ in range(num_clients)]
    for indices in class_indices:
        categorical = Categorical(dirichlet.sample())
        assignments = categorical.sample((len(indices),))
        for i, assignments in enumerate(assignments):
            client_indices[assignments].append(i)
    return [RandomSampler(indices) for indices in client_indices]

def partition_by_labels(labels: Iterator[int]) -> List[List[int]]:
    class_indices = defaultdict(list)
    num_classes = 0
    for i, class_ in enumerate(labels):
        class_indices[class_].append(i)
        num_classes = max(num_classes, class_ + 1)
    class_indices = [class_indices[i] for i in range(num_classes)]
    return class_indices

class DirichletSampler(Sampler):
    def __init__(self, class_indices: List[List[int]], alpha: float = 0.0):
        super().__init__()

        num_classes = len(class_indices)
        class_sizes = torch.as_tensor([len(indices) for indices in class_indices])
        dataset_size = torch.sum(class_sizes)
        concentration = class_sizes / dataset_size
        dirichlet = Dirichlet(alpha * concentration)

        self.class_indices = [
            torch.as_tensor(class_indices[i])[torch.randperm(class_sizes[i])]
            for i in range(num_classes)
        ]
        self.class_sizes = class_sizes
        self.dataset_size = dataset_size
        self.categorical = Categorical(dirichlet.sample())

    def __iter__(self) -> Iterator[int]:
        class_indices = [indices[torch.randperm(len(indices))].tolist() for indices in self.class_indices]
        for _ in range(len(self)):
            class_ = self.categorical.sample()
            yield class_indices[class_].pop()

    def __len__(self) -> int:
        return torch.min(self.class_sizes)
