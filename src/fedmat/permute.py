from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from transformers.models.vit.modeling_vit import ViTLayer, ViTSelfAttention, ViTSelfOutput


@torch.inference_mode()
def permute_self_attention_heads(attn: ViTSelfAttention, perm: Tensor) -> None:
    """
    Attn is a ViTSelfAttention module:
        attn.query, attn.key, attn.value : Linear
    """
    n_head = attn.num_attention_heads
    d_head = attn.attention_head_size
    d_embed = n_head * d_head

    for proj in (attn.query, attn.key, attn.value):
        W = proj.weight.data        # [D, D]
        b = proj.bias.data          # [D]

        # Permute output heads
        W4 = W.view(n_head, d_head, d_embed)
        W4 = W4[perm]
        proj.weight.data.copy_(W4.view(d_embed, d_embed))

        # Permute bias
        b4 = b.view(n_head, d_head)
        b4 = b4[perm]
        proj.bias.data.copy_(b4.view(d_embed))


@torch.inference_mode()
def permute_output_projection(
    attn_output: ViTSelfOutput,
    perm: Tensor,
    n_head: int,
    d_head: int
) -> None:
    dense = attn_output.dense
    W = dense.weight.data  # [D, D]
    d_embed = n_head * d_head

    W3 = W.view(d_embed, n_head, d_head)  # [D_out, H, hd]
    W3 = W3[:, perm, :]
    dense.weight.data.copy_(W3.view(d_embed, d_embed))
    # do nothing to bias


@torch.inference_mode()
def permute_vit_layer_heads(layer: ViTLayer, perm: Tensor) -> None:
    """Permute heads of MHA block.

    layer: a HF ViTLayer
    perm: Tensor of shape [H]

    Example:
    >>> for layer in model.vit.encoder.layer:
    >>>     permute_vit_layer_heads(layer, perm)
    """
    sa = layer.attention.attention     # ViTSelfAttention
    so = layer.attention.output        # ViTSelfOutput

    n_head = sa.num_attention_heads
    hd = sa.attention_head_size
    permute_self_attention_heads(sa, perm)
    permute_output_projection(so, perm, n_head=n_head, d_head=hd)
