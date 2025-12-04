"""Structured configuration dataclasses for FedMAT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from fedmat.matching import Matcher


@dataclass
class TrainConfig:
    """Runtime configuration for training or federated experiments."""

    run_name: str | None
    seed: int

    model_name: str
    use_pretrained: bool
    num_labels: int

    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    max_grad_norm: float | None
    log_every_n_steps: int
    use_bf16: bool
    use_torch_compile: bool

    max_train_samples: int | None
    max_eval_samples: int | None
    num_workers: int
    prefetch_factor: int

    output_dir: Path

    homogeneity: float
    num_clients: int
    num_rounds: int

    mode: str
    matcher: Matcher | None
    save_round_checkpoints: bool
    dry: bool

    dataset: Literal["cifar10", "imagenet1k"]
