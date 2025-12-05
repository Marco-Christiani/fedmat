"""Utility helpers for flattening and reconstructing model state dictionaries."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import wandb
from torch import Tensor, nn

from fedmat.distributed_context import DistributedContext
from fedmat.permute import permute_vit_layer_heads
from fedmat.utils import as_vit_layer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl
    from transformers import ViTForImageClassification

    from fedmat.config import TrainConfig
    from fedmat.matching import Matcher

StateDict = OrderedDict[str, Tensor]


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

    def __iter__(self):
        # transform to columnar layout and yield tuples (name, shape, ...) for all fields
        # structurally dynamic in case anything is added
        yield from zip(*[getattr(self, e.name) for e in fields(self)])


@torch.no_grad()
def build_flat_metadata(state_dict: StateDict) -> ModelFlatMetadata:
    """Create metadata describing ordering, shape, and dtype for a model state dict."""
    names, shapes, dtypes, numels = zip(*[(name, t.shape, t.dtype, t.numel()) for name, t in state_dict.items()])
    return ModelFlatMetadata(
        names=list(names),
        shapes=list(shapes),
        dtypes=list(dtypes),
        numels=list(numels),
    )


@torch.no_grad()
def flatten_state_dict(state_dict: StateDict, metadata: ModelFlatMetadata) -> Tensor:
    """Flatten a state dict into a 1D tensor."""
    first = next(iter(state_dict.values()))
    flat = torch.empty(
        metadata.total_numel,
        dtype=first.dtype,
        device=first.device,
    )

    offset = 0
    for name, _, _, numel in metadata:
        t = state_dict[name].view(-1)
        flat[offset : offset + numel] = t
        offset += numel

    return flat


@torch.no_grad()
def unflatten_state_dict(flat: Tensor, metadata: ModelFlatMetadata) -> StateDict:
    """Rebuild a state dict from a flat tensor."""
    out = StateDict()
    offset = 0

    for name, shape, dtype, numel in metadata:
        chunk = flat[offset : offset + numel]
        out[name] = chunk.to(dtype).view(shape).clone()
        offset += numel

    return out


@torch.no_grad()
def clone_state_dict(state_dict: StateDict, device: torch.device) -> StateDict:
    """Return a copy of the provided state dict on the requested device."""
    return OrderedDict((name, t.detach().to(device=device).clone()) for name, t in state_dict.items())


class ModelReshaper:
    def __init__(self) -> None:
        self._metadata: ModelFlatMetadata | None = None

    @property
    def metadata(self) -> ModelFlatMetadata | None:
        return self._metadata

    def ensure_metadata(self, model_or_state: nn.Module | StateDict) -> ModelFlatMetadata:
        """Return cached flattening metadata, creating it if needed."""
        if self._metadata is None:
            state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
            assert isinstance(state, OrderedDict), f"Got {state}"
            self._metadata = build_flat_metadata(state)
        return self._metadata

    def flatten(self, model_or_state: nn.Module | StateDict) -> torch.Tensor:
        """Flatten a model or state dict."""
        metadata = self.ensure_metadata(model_or_state)
        state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
        assert isinstance(state, OrderedDict), f"Got {state}"
        return flatten_state_dict(state, metadata)

    def unflatten_model(self, model: nn.Module, flat: torch.Tensor) -> None:
        """Load flattened weights into a model in-place."""
        metadata = self.ensure_metadata(model)
        state = unflatten_state_dict(flat, metadata)
        model.load_state_dict(state)

@torch.no_grad()
def _weighted_average(stacked: torch.Tensor, client_weights: torch.Tensor | None) -> torch.Tensor:
    if client_weights is None:
        return stacked.mean(dim=0)
    wshape = (-1,) + ((1,) * (stacked.dim() - 1))
    return (client_weights.view(wshape) * stacked).sum(dim=0)

@torch.no_grad()
def _aggregate_encoder_layers(
    client_models: list[ViTForImageClassification],
    matcher: Matcher | None,
    client_weights: torch.Tensor | None = None,
) -> StateDict:
    """Aggregate encoder layers with optional head matching."""
    aggregated_state = StateDict()
    reference_model = client_models[0]
    num_layers = len(reference_model.vit.encoder.layer)

    for layer_idx in range(num_layers):
        client_layers = [as_vit_layer(model.vit.encoder.layer[layer_idx]) for model in client_models]

        if matcher is not None:
            perms = matcher.match(client_layers)
            for layer, perm in zip(client_layers, perms):
                permute_vit_layer_heads(layer, perm)

        param_maps = [dict(layer.named_parameters()) for layer in client_layers]
        reference_param_map = param_maps[0]

        for name in reference_param_map:
            full_name = f"vit.encoder.layer.{layer_idx}.{name}" if name else f"vit.encoder.layer.{layer_idx}"
            tensors = [param_map[name].data for param_map in param_maps]
            stacked = torch.stack(tensors, dim=0)
            avg = _weighted_average(stacked, client_weights)
            aggregated_state[full_name] = avg

        buffer_maps = [dict(layer.named_buffers()) for layer in client_layers]
        if buffer_maps:
            reference_buffer_map = buffer_maps[0]
            for name, reference_buffer in reference_buffer_map.items():
                full_name = f"vit.encoder.layer.{layer_idx}.{name}" if name else f"vit.encoder.layer.{layer_idx}"
                if torch.is_floating_point(reference_buffer):
                    tensors = [buffer_map[name].data for buffer_map in buffer_maps]
                    stacked = torch.stack(tensors, dim=0)
                    avg = _weighted_average(stacked, client_weights)
                    aggregated_state[full_name] = avg
                else:
                    aggregated_state[full_name] = reference_buffer.detach().clone()

    return aggregated_state


def _aggregate_remaining_parameters(
    client_models: list[ViTForImageClassification],
    aggregated_state: StateDict,
    client_weights: torch.Tensor | None = None,
) -> StateDict:
    """Aggregate all non-encoder parameters and buffers."""
    client_state_dicts = [model.state_dict() for model in client_models]
    reference_state_dict = client_state_dicts[0]

    with torch.no_grad():
        for name, reference_tensor in reference_state_dict.items():
            if name in aggregated_state:
                continue

            if torch.is_floating_point(reference_tensor):
                tensors = [sd[name] for sd in client_state_dicts]
                stacked = torch.stack(tensors, dim=0)
                avg = _weighted_average(stacked, client_weights)
                aggregated_state[name] = avg
            else:
                aggregated_state[name] = reference_tensor.detach().clone()

    return aggregated_state


def aggregate_models(
    client_models: list[ViTForImageClassification],
    matcher: Matcher | None = None,
    client_weights: torch.Tensor | None = None,
) -> StateDict:
    """Aggregate client ViT models with per-layer matched averaging.

    Args:
        client_models: Client models already on target device
        matcher: Optional head matcher for encoder layers

    Returns
    -------
        Aggregated state dict with all tensors on device, preserving native dtypes
    """
    aggregated_state = _aggregate_encoder_layers(client_models, matcher, client_weights=client_weights)
    return _aggregate_remaining_parameters(client_models, aggregated_state, client_weights=client_weights)


@DistributedContext.on_rank(0)
def log_run_artifacts(
    *,
    train_config: TrainConfig,
    model: ViTForImageClassification,
    metrics_df: pl.DataFrame,
    final_confmat: torch.Tensor | None,
    final_accuracy: float | None,
) -> Path:
    """Save artifacts from a finished run.

    - Saves all local artifacts in the run dir
    - Logs to wandb:
        - table from Polars DF
        - confusion matrix (as image or wandb native type)
        - scalar metrics
        - model checkpoint (Artifact)
    """
    run_dir: Path = train_config.output_dir
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")

    # Local --------------------------------------------------------------------
    metrics_path = artifacts_dir / f"metrics-{timestamp}.parquet"
    metrics_df.write_parquet(metrics_path)

    summary_path = artifacts_dir / f"summary-{timestamp}.json"
    summary_payload = {
        "final_accuracy": final_accuracy,
        "n_rows": metrics_df.height,
        "n_clients": train_config.num_clients,
        "n_rounds": train_config.num_rounds,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    # Confusion matrix
    if final_confmat is not None:
        conf_path = artifacts_dir / f"confusion-{timestamp}.pt"
        torch.save(final_confmat.cpu(), conf_path)

    # Model checkpoint
    ckpt_path = artifacts_dir / f"model-{timestamp}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # W&B ----------------------------------------------------------------------
    run = wandb.run
    if run is not None:
        # table
        run.log({"train/metrics_table": wandb.Table(dataframe=metrics_df.to_pandas())})

        # scalar
        if final_accuracy is not None:
            run.log({"eval/final_accuracy": final_accuracy})

        # confusion matrix plot
        if final_confmat is not None:
            run.log({
                "eval/confusion_matrix": wandb.plots.HeatMap(
                    x=list(range(final_confmat.shape[0])),
                    y=list(range(final_confmat.shape[1])),
                    z=final_confmat.cpu().numpy().tolist(),
                    title="Confusion Matrix",
                )
            })

        # upload local artifacts from run dir
        artifact = wandb.Artifact(
            name=f"fedmat-run-{timestamp}",
            type="fedmat-run",
        )
        artifact.add_dir(str(artifacts_dir))
        run.log_artifact(artifact)

    return artifacts_dir
