"""Utility helpers for flattening and reconstructing model state dictionaries."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ModelFlatMetadata:
    """Shape/dtype metadata for converting state dicts to flattened tensors."""

    names: Sequence[str]
    shapes: Sequence[torch.Size]
    dtypes: Sequence[torch.dtype]
    numels: Sequence[int]

    @property
    def total_numel(self) -> int:
        """Total number of scalar elements represented by this metadata."""
        return sum(self.numels)


def build_flat_metadata(state_dict: OrderedDict[str, torch.Tensor]) -> ModelFlatMetadata:
    """Create metadata describing ordering, shape, and dtype for a model state dict."""
    names: list[str] = []
    shapes: list[torch.Size] = []
    dtypes: list[torch.dtype] = []
    numels: list[int] = []
    for name, tensor in state_dict.items():
        names.append(name)
        shapes.append(tensor.shape)
        dtypes.append(tensor.dtype)
        numels.append(tensor.numel())
    return ModelFlatMetadata(names=names, shapes=shapes, dtypes=dtypes, numels=numels)


def flatten_state_dict(state_dict: OrderedDict[str, torch.Tensor], metadata: ModelFlatMetadata) -> torch.Tensor:
    """Flatten a state dict into a single CPU vector using provided metadata."""
    flat = torch.empty(metadata.total_numel, dtype=torch.float32, device="cpu")
    offset = 0
    for name, numel in zip(metadata.names, metadata.numels):
        tensor = state_dict[name].detach().to(device="cpu", dtype=torch.float32).view(-1)
        flat[offset : offset + numel] = tensor
        offset += numel
    return flat


def unflatten_state_dict(flat: torch.Tensor, metadata: ModelFlatMetadata) -> OrderedDict[str, torch.Tensor]:
    """Convert a flattened tensor back into a state dict using metadata."""
    if flat.device != torch.device("cpu"):
        flat = flat.to(device="cpu")
    offset = 0
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, shape, dtype, numel in zip(metadata.names, metadata.shapes, metadata.dtypes, metadata.numels):
        view = flat[offset : offset + numel]
        tensor = view.to(dtype=dtype).view(shape).clone()
        state[name] = tensor
        offset += numel
    return state


def clone_state_dict(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Return a CPU copy of the provided state dict."""
    clone: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        clone[name] = tensor.detach().cpu().clone()
    return clone

