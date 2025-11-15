from __future__ import annotations

import torch
from torch import Tensor, nn


def permute_heads(
    qkv: Tensor,
    perm: Tensor,
    n_head: int,
    d_head: int,
    out: Tensor | None = None
):
    D = n_head * d_head
    qkv = qkv.view(D, 3, n_head, d_head)  # [H*hd, 3*H*hd] to [D, 3, H, hd]
    qkv = qkv[:, :, perm, :]     # permute
    qkv = qkv.view(D, 3 * D)     # [H*hd, 3*H*hd]
    if out:
        out.data.copy_(qkv)
        return out
    return qkv


@torch.inference_mode()
def permute_mha_block(
    layer: nn.Module,
    perm: torch.Tensor
):
    mha = layer.mha
    assert isinstance(mha, nn.Module)
    H = mha.num_heads
    assert isinstance(H, int)
    hd = mha.head_dim
    assert isinstance(hd, int)
    # D = H * hd
    qkv = mha.qkv.data
    assert isinstance(qkv, Tensor)
    permute_heads(qkv=qkv, perm=perm, n_head=H, d_head=hd)
