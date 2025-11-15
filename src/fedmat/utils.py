from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch


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



