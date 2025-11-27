"""Transformer models and utilities for federated learning simulations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MHA(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        embed_dim: int | None = None,
        causal: bool = False,
    ) -> None:
        """Initialize multi-head attention.

        Parameters
        ----------
        num_heads : int
            Number of attention heads
        head_dim : int
            Dimension of each head
        embed_dim : int | None, optional
            Embedding dimension, defaults to num_heads * head_dim
        causal : bool, optional
            Whether to use causal attention mask, by default False
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        if embed_dim is None:
            embed_dim = num_heads * head_dim
        self.causal = causal

        self.qkv = nn.Parameter(torch.randn(embed_dim, 3 * num_heads * head_dim))
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention."""
        qkv = x @ self.qkv
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))

        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(-1, -2).flatten(-2)
        out = self.o_proj(out)

        return out


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, intermediate_dim: int, hidden_dim: int) -> None:
        """Initialize MLP.

        Parameters
        ----------
        intermediate_dim : int
            Input/output dimension
        hidden_dim : int
            Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(intermediate_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.fc2(functional.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        mlp_hidden_dim: int,
        embed_dim: int | None = None,
        causal: bool = False,
    ) -> None:
        """Initialize transformer block.

        Parameters
        ----------
        num_heads : int
            Number of attention heads
        head_dim : int
            Dimension of each head
        mlp_hidden_dim : int
            Hidden dimension in MLP
        embed_dim : int | None, optional
            Embedding dimension, by default None
        causal : bool, optional
            Whether to use causal attention, by default False
        """
        super().__init__()
        self.mha = MHA(num_heads, head_dim, embed_dim, causal)
        emb = embed_dim if embed_dim is not None else num_heads * head_dim
        self.mlp = MLP(emb, mlp_hidden_dim)
        self.ln1 = nn.LayerNorm(emb)
        self.ln2 = nn.LayerNorm(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Transformer model composed of stacked transformer blocks."""

    def __init__(self, num_layers: int = 2, num_heads: int = 4, d_model: int = 16) -> None:
        """Initialize transformer.

        Parameters
        ----------
        num_layers : int, optional
            Number of transformer layers, by default 2
        num_heads : int, optional
            Number of attention heads, by default 4
        d_model : int, optional
            Model dimension, by default 16
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(num_heads, d_model // num_heads, mlp_hidden_dim=4 * d_model, embed_dim=d_model)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


def permute_heads_layer(layer: TransformerBlock, perm: torch.Tensor) -> None:
    """Permute attention heads in a transformer layer.

    Parameters
    ----------
    layer : TransformerBlock
        Transformer layer to permute
    perm : torch.Tensor
        Permutation indices for heads
    """
    mha = layer.mha
    H = mha.num_heads
    hd = mha.head_dim

    with torch.no_grad():
        embed_dim = mha.qkv.shape[0]
        qkv = mha.qkv.data.view(embed_dim, 3, H, hd)
        qkv = qkv[:, :, perm, :]
        qkv = qkv.contiguous().view(embed_dim, 3 * H * hd)
        mha.qkv.data.copy_(qkv)

        W = mha.o_proj.weight.data
        out_features, in_features = W.shape
        assert in_features == H * hd
        W = W.view(out_features, H, hd)
        W = W[:, perm, :]
        W = W.contiguous().view(out_features, H * hd)
        mha.o_proj.weight.data.copy_(W)


def apply_random_permutation_to_model_heads(model: Transformer, p: float = 1.0) -> None:
    """Apply random permutations to attention heads in a model.

    Parameters
    ----------
    model : Transformer
        Transformer model to modify
    p : float, optional
        Probability of permuting each layer, by default 1.0
    """
    for layer in model.layers:
        if torch.rand(()) < p:
            perm = torch.randperm(layer.mha.num_heads)
            permute_heads_layer(layer, perm)


def model_param_distance(m1: nn.Module, m2: nn.Module) -> float:
    """Compute L2 distance between model parameters.

    Parameters
    ----------
    m1 : nn.Module
        First model
    m2 : nn.Module
        Second model

    Returns
    -------
    float
        L2 distance between parameters
    """
    s1 = m1.state_dict()
    s2 = m2.state_dict()
    sq_sum = 0.0
    for k in s1:
        sq_sum += (s1[k] - s2[k]).pow(2).sum().item()
    return sq_sum**0.5


def average_models(models: list[Transformer]) -> Transformer:
    """Compute the average of multiple transformer models.

    Parameters
    ----------
    models : list[Transformer]
        List of transformer models to average

    Returns
    -------
    Transformer
        New model with averaged parameters
    """
    base = copy.deepcopy(models[0])
    avg_state = base.state_dict()
    n = len(models)

    for k in avg_state:
        avg_state[k] = sum(m.state_dict()[k] for m in models) / n

    base.load_state_dict(avg_state)
    return base


def flatten_heads_for_matching(layer: TransformerBlock) -> torch.Tensor:
    """Flatten attention heads for head matching/alignment.

    Parameters
    ----------
    layer : TransformerBlock
        Transformer layer to extract heads from

    Returns
    -------
    torch.Tensor
        Flattened head representations
    """
    mha = layer.mha
    H = mha.num_heads
    hd = mha.head_dim

    embed_dim = mha.qkv.shape[0]
    qkv = mha.qkv.data.view(embed_dim, 3, H, hd)
    qkv_heads = qkv.permute(2, 0, 1, 3).contiguous().view(H, -1)

    W = mha.o_proj.weight.data
    out_features, in_features = W.shape
    assert in_features == H * hd
    W = W.view(out_features, H, hd)
    W_heads = W.permute(1, 0, 2).contiguous().view(H, -1)

    return torch.cat([qkv_heads, W_heads], dim=1)


def align_client_to_ref(client: Transformer, ref: Transformer) -> Transformer:
    """Align a client model's heads to a reference model.

    Parameters
    ----------
    client : Transformer
        Client model to align
    ref : Transformer
        Reference model to align to

    Returns
    -------
    Transformer
        Client model with aligned heads
    """
    for ref_layer, cli_layer in zip(ref.layers, client.layers):
        H = ref_layer.mha.num_heads

        ref_flat = flatten_heads_for_matching(ref_layer)
        cli_flat = flatten_heads_for_matching(cli_layer)

        cost = torch.cdist(ref_flat, cli_flat, p=2)

        perm = torch.empty(H, dtype=torch.long)
        unused = set(range(H))
        for i in range(H):
            row = cost[i]
            best_j = min(unused, key=lambda j: row[j].item())
            perm[i] = best_j
            unused.remove(best_j)

        permute_heads_layer(cli_layer, perm)

    return client


def align_all_clients_to_first(clients: list[Transformer]) -> list[Transformer]:
    """Align all client models to the first client model.

    Parameters
    ----------
    clients : list[Transformer]
        List of client models

    Returns
    -------
    list[Transformer]
        List of aligned client models
    """
    ref = clients[0]
    aligned = [copy.deepcopy(ref)]
    for c in clients[1:]:
        aligned.append(align_client_to_ref(copy.deepcopy(c), ref))
    return aligned


def make_clients_from_global(
    global_model: Transformer,
    num_clients: int = 5,
    noise_std: float = 0.05,
    perm_prob: float = 0.5,
) -> list[Transformer]:
    """Create client models from a global model with noise and permutations.

    Parameters
    ----------
    global_model : Transformer
        Global model to create clients from
    num_clients : int, optional
        Number of clients to create, by default 5
    noise_std : float, optional
        Standard deviation of Gaussian noise to add, by default 0.05
    perm_prob : float, optional
        Probability of permuting each layer, by default 0.5

    Returns
    -------
    list[Transformer]
        List of client models
    """
    clients = []
    for _ in range(num_clients):
        c = copy.deepcopy(global_model)
        with torch.no_grad():
            for p in c.parameters():
                p.add_(torch.randn_like(p) * noise_std)
        apply_random_permutation_to_model_heads(c, p=perm_prob)
        clients.append(c)
    return clients


def main() -> None:
    """Run simulation demo comparing naive and head-aligned federated averaging."""
    num_layers = 3
    num_heads = 4
    d_model = 32
    num_clients = 8

    global_model = Transformer(num_layers=num_layers, num_heads=num_heads, d_model=d_model)

    clients = make_clients_from_global(
        global_model,
        num_clients=num_clients,
        noise_std=0.1,
        perm_prob=0.7,
    )

    naive_agg = average_models(clients)

    aligned_clients = align_all_clients_to_first(clients)
    aligned_agg = average_models(aligned_clients)

    dist_naive = model_param_distance(global_model, naive_agg)
    dist_aligned = model_param_distance(global_model, aligned_agg)

    print("Parameter distance to original global model:")
    print(f" - Naive FedAvg:        {dist_naive:.4f}")
    print(f" - Head-aligned FedAvg: {dist_aligned:.4f}")

    x = torch.randn(128, d_model)
    with torch.no_grad():
        y_true = global_model(x)
        y_naive = naive_agg(x)
        y_aligned = aligned_agg(x)

    mse_naive = functional.mse_loss(y_naive, y_true).item()
    mse_aligned = functional.mse_loss(y_aligned, y_true).item()

    print("\nOutput MSE vs original global model on random data:")
    print(f" - Naive FedAvg:        {mse_naive:.6f}")
    print(f" - Head-aligned FedAvg: {mse_aligned:.6f}")


if __name__ == "__main__":
    main()
