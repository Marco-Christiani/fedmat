"""Timing-based validation of CUDA stream overlap."""

from __future__ import annotations

import types
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

from fedmat import train
from fedmat.config import TrainConfig

if TYPE_CHECKING:
    from pathlib import Path


def _format_line(label: str, value: float, baseline: float) -> str:
    speedup = baseline / value if value else float("inf")
    ok = "✓" if value < baseline * 2.0 else "✗"
    return f"{label:<18} {value:6.2f}s (speedup: {speedup:4.2f}x) {ok}"


def _run_stream_parallel_benchmark(num_clients: int, steps: int = 32, sleep_us: int = 5_000_000) -> float:
    """Synthetic CUDA-stream benchmark mirroring the training scheduling pattern."""
    device = torch.device("cuda")
    streams = [torch.cuda.Stream(device=device) for _ in range(num_clients)]
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(steps):
        for stream in streams:
            with torch.cuda.stream(stream):
                torch.cuda._sleep(sleep_us)  # pyright: ignore[reportPrivateUsage]

    for stream in streams:
        stream.synchronize()

    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / 1000


def _run_real_workload_benchmark(num_clients: int, steps: int = 8, size: int = 1536) -> float:
    """Benchmark with GPU compute (matmul + relu) across per-client streams."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    streams = [torch.cuda.Stream(device=device) for _ in range(num_clients)]
    a = [torch.randn(size, size, device=device) for _ in range(num_clients)]
    b = [torch.randn(size, size, device=device) for _ in range(num_clients)]
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for stream in streams:
        with torch.cuda.stream(stream):
            torch.matmul(a[0], b[0])

    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(steps):
        for idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                out = torch.matmul(a[idx], b[idx])
                out = torch.nn.functional.relu(out)
                out = torch.matmul(out, b[idx].t())
                out = torch.nn.functional.relu(out)
    for stream in streams:
        stream.synchronize()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / 1000


class TinyModel(nn.Module):
    """Small model to exercise the train loop without external downloads."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
        )

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
        """Mirror the ViT classifier API shape for logits."""
        logits = self.net(pixel_values)
        return types.SimpleNamespace(logits=logits, loss=None)


class TinyDataset(torch.utils.data.Dataset):
    """Synthetic dataset aligned with TinyModel input shapes."""

    def __init__(self, num_labels: int, length: int = 64) -> None:
        self.length = length
        self.num_labels = num_labels

    def __len__(self) -> int:
        """Dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Random image/label pair for a stable synthetic batch."""
        torch.manual_seed(idx)
        return {
            "pixel_values": torch.randn(3, 16, 16),
            "labels": torch.tensor(idx % self.num_labels, dtype=torch.long),
        }


def _fake_load_named_dataset_subsets(num_labels: int) -> tuple[TinyDataset, TinyDataset]:
    ds = TinyDataset(num_labels=num_labels, length=64)
    return ds, ds


def _fake_build_dataloaders(
    train_ds: TinyDataset,
    homogeneity: float,
    num_clients: int,
    eval_ds: TinyDataset,
    image_processor: Any,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    drop_last: bool = False,
) -> tuple[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader, torch.Tensor]:
    _ = homogeneity, num_workers, prefetch_factor, drop_last, image_processor
    loaders = [
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for _ in range(num_clients)
    ]
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    hist = torch.ones((num_clients, train_ds.num_labels), device=device)
    return loaders, eval_loader, hist


def _fake_aggregate_models(
    client_models: list[nn.Module],
    matcher: object | None = None,
    client_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    _ = matcher, client_weights
    return client_models[0].state_dict()


def _fake_evaluate(*args, **kwargs) -> tuple[float, torch.Tensor]:
    _ = args, kwargs
    num_labels = kwargs.get("num_labels", 3) if kwargs else 3
    return 0.0, torch.zeros((num_labels, num_labels), dtype=torch.int64)


@pytest.mark.cuda
def test_cuda_stream_parallelism_speedup() -> None:
    """Ensure multi-client stream work overlaps instead of running sequentially."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for stream timing test")

    baseline = _run_stream_parallel_benchmark(num_clients=1, steps=32, sleep_us=5_000_000)
    parallel_timings = {
        2: _run_stream_parallel_benchmark(num_clients=2, steps=32, sleep_us=5_000_000),
        4: _run_stream_parallel_benchmark(num_clients=4, steps=32, sleep_us=5_000_000),
        8: _run_stream_parallel_benchmark(num_clients=8, steps=32, sleep_us=5_000_000),
    }

    print("Testing CUDA Stream Parallelism")
    print("================================")
    print(f"Single client baseline: {baseline:6.2f}s")
    for clients, elapsed in parallel_timings.items():
        print(_format_line(f"{clients} clients:", elapsed, baseline))

    for elapsed in parallel_timings.values():
        assert elapsed < baseline * 2.0, (
            f"Expected parallel run to be <2x baseline (baseline={baseline:.2f}s, observed={elapsed:.2f}s)"
        )


@pytest.mark.cuda
def test_cuda_stream_parallelism_real_workload() -> None:
    """Validate overlap with a real matmul workload instead of a sleep kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for stream timing test")

    baseline = _run_real_workload_benchmark(num_clients=1)
    parallel_timings = {
        2: _run_real_workload_benchmark(num_clients=2),
        4: _run_real_workload_benchmark(num_clients=4),
    }

    print("Testing CUDA Stream Parallelism (real workload)")
    print("===============================================")
    print(f"Single client baseline: {baseline:6.2f}s")
    for clients, elapsed in parallel_timings.items():
        print(_format_line(f"{clients} clients:", elapsed, baseline))

    for elapsed in parallel_timings.values():
        assert elapsed < baseline * 2.0, (
            f"Expected real workload parallel run to be <2x baseline "
            f"(baseline={baseline:.2f}s, observed={elapsed:.2f}s)"
        )


@pytest.mark.cuda
def test_train_loop_parallelism_with_real_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: "Path",
) -> None:
    """Run the actual training loop with some components to verify overlap."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for stream timing test")

    num_labels = 3

    monkeypatch.setattr(
        "fedmat.train.load_named_dataset_subsets",
        lambda *_, **__: _fake_load_named_dataset_subsets(num_labels),
    )
    monkeypatch.setattr(
        "fedmat.train.build_dataloaders",
        lambda train_ds, homogeneity, num_clients, eval_ds, image_processor, batch_size, num_workers, prefetch_factor, device, drop_last=False: _fake_build_dataloaders(  # noqa: E501
            train_ds=train_ds,
            homogeneity=homogeneity,
            num_clients=num_clients,
            eval_ds=eval_ds,
            image_processor=image_processor,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            device=device,
            drop_last=drop_last,
        ),
    )
    monkeypatch.setattr("fedmat.train.create_vit_classifier", lambda *args, **kwargs: TinyModel(num_labels=num_labels))
    monkeypatch.setattr("fedmat.train.aggregate_models", _fake_aggregate_models)
    monkeypatch.setattr("fedmat.train.evaluate", _fake_evaluate)
    dummy_processor = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setattr("fedmat.train.AutoImageProcessor", dummy_processor)

    base_cfg = TrainConfig(
        run_name=None,
        seed=0,
        model_name="tiny",
        use_pretrained=False,
        num_labels=num_labels,
        learning_rate=1e-2,
        lr_decay=1.0,
        weight_decay=0.0,
        batch_size=8,
        local_steps=8,
        max_grad_norm=None,
        log_every_n_steps=16,
        enable_fed_weights=False,
        use_bf16=False,
        use_torch_compile=False,
        max_train_samples=None,
        max_eval_samples=None,
        num_workers=0,
        prefetch_factor=2,
        output_dir=tmp_path,
        homogeneity=0.0,
        num_clients=1,
        num_rounds=1,
        matcher=None,
        optimizer="sgd",
        save_round_checkpoints=False,
        save_final_checkpoint=False,
        dataset="cifar10",
        dry=False,
        enable_timing=False,
        use_wandb=False,
    )

    def timed_run(num_clients: int) -> float:
        cfg = replace(base_cfg, num_clients=num_clients)
        start = perf_counter()
        _ = train._run_fed_training(cfg, quiver=None)
        return perf_counter() - start

    t1 = timed_run(1)
    t4 = timed_run(4)

    print("Testing train loop overlap w real model")
    print("=======================================")
    print(f"Single client baseline: {t1:6.2f}s")
    print(_format_line("4 clients:", t4, t1))

    assert t4 < t1 * 2.0, f"Expected 4-client training <2x single-client time (t1={t1:.2f}, t4={t4:.2f})"
