"""Utility helpers for flattening and reconstructing model state dictionaries."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor, nn

import wandb
from fedmat.distributed_context import DistributedContext

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from transformers import ViTForImageClassification

    from fedmat.configs import TrainConfig

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
    out: StateDict = OrderedDict()
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
    def __init__(self):
        self._metadata: ModelFlatMetadata | None = None

    @property
    def metadata(self) -> ModelFlatMetadata | None:
        return self._metadata

    def ensure_metadata(self, model_or_state: nn.Module | StateDict) -> ModelFlatMetadata:
        """Return cached flattening metadata, creating it if needed."""
        if self._metadata is None:
            state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
            assert isinstance(state, OrderedDict)
            self._metadata = build_flat_metadata(state)
        return self._metadata

    def flatten(self, model_or_state: nn.Module | StateDict) -> torch.Tensor:
        """Flatten a model or state dict."""
        metadata = self.ensure_metadata(model_or_state)
        state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
        assert isinstance(state, OrderedDict)
        return flatten_state_dict(state, metadata)

    def unflatten_model(self, model: nn.Module, flat: torch.Tensor) -> None:
        """Load flattened weights into a model in-place."""
        metadata = self.ensure_metadata(model)
        state = unflatten_state_dict(flat, metadata)
        model.load_state_dict(state)


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
