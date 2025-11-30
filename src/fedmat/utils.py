"""Utility functions for training and data management."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import torch
from transformers import ViTConfig, ViTForImageClassification

T = TypeVar("T", bound=Mapping[str, Any])


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    """Get the default device (CUDA, MPS, or CPU in that order).

    Returns
    -------
    torch.device
        The best available device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def get_amp_settings(
    device: torch.device,
    enable_bf16: bool,
) -> tuple[bool, torch.dtype]:
    """Get automatic mixed precision settings based on device and preferences.

    Parameters
    ----------
    device : torch.device
        Target device for computation
    enable_bf16 : bool
        Whether to enable bfloat16 (only supported on CUDA)

    Returns
    -------
    tuple[bool, torch.dtype]
        Tuple of (use_autocast, dtype) for autocast context
    """
    if device.type == "cuda" and enable_bf16:
        return True, torch.bfloat16
    return False, torch.float32


def create_vit_classifier(
    model_name: str,
    num_labels: int,
    use_pretrained: bool,
) -> ViTForImageClassification:
    """Construct a ViT classifier with consistent label mappings."""
    if use_pretrained:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            # device_map="auto",
        )
    else:
        model = ViTForImageClassification(
            ViTConfig(
                num_labels=num_labels,
            )
        )

    model.config.id2label = {i: str(i) for i in range(num_labels)}
    model.config.label2id = {str(i): i for i in range(num_labels)}
    return model


def aos_to_soa(aos: Sequence[T]) -> dict[str, list[Any]]:
    """AOS to SOA.

    Example:
        >>> aos = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
        >>> aos_to_soa(aos)
        {'name': ['Alice', 'Bob'], 'age': [25, 30]}
    """
    if not aos:
        return {}

    return {k: [record[k] for record in aos] for k in aos[0]}
