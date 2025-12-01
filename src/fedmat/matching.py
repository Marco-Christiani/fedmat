"""Matching utilities for aligning ViT attention heads across clients."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List

import torch
from torch import Tensor, nn

from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from transformers.models.vit.modeling_vit import ViTLayer


def module_names(m: nn.Module) -> List[str]:
    """Return parameter names for a module."""
    return [name for name, _ in m.named_parameters()]


def vectorize(m: nn.Module, names: List[str]) -> Tensor:
    """Vectorize a module according to the order given in names."""
    return torch.cat([m.get_parameter(name).flatten() for name in names])


class Matcher(ABC):
    """Base class for head-matching implementations."""

    @staticmethod
    def match(layers: list[ViTLayer]) -> list[Tensor]:
        """Return a permutation per client layer."""
        raise NotImplementedError("Subclasses must implement match().")

def _flatten_layer(layer: ViTLayer) -> Tensor:
    sa = layer.attention.attention  # ViTSelfAttention
    so = layer.attention.output  # ViTSelfOutput

    n_head = sa.num_attention_heads
    hd = sa.attention_head_size
    d_embed = n_head * hd

    parts: List[Tensor] = []

    for proj in (sa.query, sa.key, sa.value):
        w = proj.weight.data  # [D, D]
        w4 = w.view(n_head, hd, d_embed)  # [H, hd, d_embed]
        parts.append(w4.contiguous().view(n_head, -1))

    w_out = so.dense.weight.data  # [D_out, D_in]
    out_features, in_features = w_out.shape
    assert in_features == d_embed
    w3 = w_out.view(out_features, n_head, hd)  # [D_out, H, hd]
    w_heads = w3.permute(1, 0, 2).contiguous().view(n_head, -1)
    parts.append(w_heads)

    return torch.cat(parts, dim=1)


class GreedyMatcher(Matcher):
    """Greedy matching implementation."""

    @staticmethod
    def match(layers: list[ViTLayer]) -> list[Tensor]:
        if len(layers) == 0:
            return []

        ref_layer = layers[0]
        # try to infer a device from reference parameters, fall back to cpu
        try:
            device = next(ref_layer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        ref_flat = _flatten_layer(ref_layer).to(device)
        num_heads = ref_flat.shape[0]

        perms: List[Tensor] = []
        perms.append(torch.arange(num_heads, dtype=torch.long, device=device))

        for layer in layers[1:]:
            cli_flat = _flatten_layer(layer).to(device)
            cost = torch.cdist(ref_flat, cli_flat, p=2)

            perm = torch.empty(num_heads, dtype=torch.long, device=device)
            unused = set(range(num_heads))
            for i in range(num_heads):
                row = cost[i]
                best_j = min(unused, key=lambda j: float(row[j].item()))
                perm[i] = best_j
                unused.remove(best_j)

            perms.append(perm)

        return perms


class HungarianMatcher(Matcher):
    """Hungarian-matching implementation."""

    @staticmethod
    def match(layers: list["ViTLayer"]) -> list[Tensor]:
        if len(layers) == 0:
            return []

        ref_layer = layers[0]
        # try to infer a device from reference parameters, fall back to cpu
        try:
            device = next(ref_layer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        ref_flat = _flatten_layer(ref_layer).to(device)
        num_heads = ref_flat.shape[0]

        perms: List[Tensor] = []
        perms.append(torch.arange(num_heads, dtype=torch.long, device=device))

        flattened_layers = [_flatten_layer(layer).to(device) for layer in layers]
        for layer_index in range(1, len(layers)):
            cli_flat = flattened_layers[layer_index]
            cost = torch.sum(torch.stack([torch.cdist(ref_flat, cli_flat, p=2) for ref_flat in flattened_layers[:layer_index]]), dim=0)

            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            perm = torch.empty(num_heads, dtype=torch.long, device=device)
            for r, c in zip(row_ind, col_ind):
                perm[r] = c

            perms.append(perm)

        return perms

