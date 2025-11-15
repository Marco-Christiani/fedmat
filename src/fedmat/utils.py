from __future__ import annotations

import random
from typing import TypeVar

import numpy as np
import torch

T = TypeVar('T')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def get_amp_settings(
    device: torch.device,
    enable_bf16: bool,
) -> tuple[bool, torch.dtype]:
    if device.type == "cuda" and enable_bf16:
        return True, torch.bfloat16
    return False, torch.float32

def aos_to_soa(aos: list[dict[str, T]]) -> dict[str, list[T]]:
    """AOS to SOA.
    
    Example:
        >>> aos = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
        >>> aos_to_soa(aos)
        {'name': ['Alice', 'Bob'], 'age': [25, 30]}
    """
    if not aos:
        return {}
    
    return {k: [record[k] for record in aos] for k in aos[0]}