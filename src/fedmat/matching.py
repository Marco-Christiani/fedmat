from __future__ import annotations
from abc import ABC
from typing import List, TYPE_CHECKING, Dict, Type

import torch
from torch import nn, Tensor

if TYPE_CHECKING:
    from transformers.models.vit.modeling_vit import ViTLayer

def module_names(m: nn.Module) -> List[str]:
    return [name for name, _ in m.named_parameters()]

def vectorize(m: nn.Module, names: List[str]) -> Tensor:
    """Vectorize a module according to the order given in names."""
    return torch.cat([m.get_parameter(name).flatten() for name in names])

class Matcher(ABC):
    @staticmethod
    def match(layers: list[ViTLayer]) -> list[Tensor]:
        """
        clients: list of ViTLayer modules from different clients
        returns: list of permutation tensors, one per client
        """
        raise NotImplementedError()

# registry for matcher implementations
_matcher_registry: Dict[str, Type[Matcher]] = {}
def register_matcher(name: str):
    """Decorator to register a Matcher implementation under a string name."""
    def _decorator(cls: Type[Matcher]) -> Type[Matcher]:
        if not issubclass(cls, Matcher):
            raise TypeError("Can only register subclasses of Matcher")
        _matcher_registry[name] = cls
        return cls
    return _decorator

def get_matcher(name: str, *args, **kwargs) -> Matcher:
    """Return an instance of the matcher registered under `name`."""
    try:
        cls = _matcher_registry[name]
    except KeyError:
        raise KeyError(f"Unknown matcher '{name}'. Available: {list(_matcher_registry.keys())}")
    return cls(*args, **kwargs)

def registered_matchers() -> List[str]:
    return list(_matcher_registry.keys())

@register_matcher("greedy")
class GreedyMatcher(Matcher):
    @staticmethod
    def match(layers: list[ViTLayer]) -> list[Tensor]:
        """
        Greedy matching implementation.
        """
        if len(layers) == 0:
            return []

        def _flatten_layer(layer: ViTLayer) -> Tensor:
            sa = layer.attention.attention  # ViTSelfAttention
            so = layer.attention.output  # ViTSelfOutput

            n_head = sa.num_attention_heads
            hd = sa.attention_head_size
            d_embed = n_head * hd

            parts: List[Tensor] = []

            for proj in (sa.query, sa.key, sa.value):
                W = proj.weight.data  # [D, D]
                W4 = W.view(n_head, hd, d_embed)  # [H, hd, d_embed]
                parts.append(W4.contiguous().view(n_head, -1))

            W_out = so.dense.weight.data  # [D_out, D_in]
            out_features, in_features = W_out.shape
            assert in_features == d_embed
            W3 = W_out.view(out_features, n_head, hd)  # [D_out, H, hd]
            W_heads = W3.permute(1, 0, 2).contiguous().view(n_head, -1)
            parts.append(W_heads)

            return torch.cat(parts, dim=1)

        ref_layer = layers[0]
        # try to infer a device from reference parameters, fall back to cpu
        try:
            device = next(ref_layer.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        ref_flat = _flatten_layer(ref_layer).to(device)
        H = ref_flat.shape[0]

        perms: List[Tensor] = []
        perms.append(torch.arange(H, dtype=torch.long, device=device))

        for layer in layers[1:]:
            cli_flat = _flatten_layer(layer).to(device)
            cost = torch.cdist(ref_flat, cli_flat, p=2)

            perm = torch.empty(H, dtype=torch.long, device=device)
            unused = set(range(H))
            for i in range(H):
                row = cost[i]
                best_j = min(unused, key=lambda j: float(row[j].item()))
                perm[i] = best_j
                unused.remove(best_j)

            perms.append(perm)

        return perms

@register_matcher("hungarian")
class HungarianMatcher(Matcher):
    @staticmethod
    def match(layers: list["ViTLayer"]) -> list[Tensor]:
        """Le Hungarian ala FedMA (not implemented)."""
        raise NotImplementedError()
