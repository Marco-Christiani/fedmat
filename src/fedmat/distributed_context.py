"""Unified distributed runtime utilities for FedMAT experiments."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import asdict
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import wandb
from torch.multiprocessing import get_context, spawn

from fedmat import install_exception_handlers, install_global_log_context, set_log_context

from .train_utils import ModelFlatMetadata, build_flat_metadata, flatten_state_dict, unflatten_state_dict

logger = logging.getLogger(__name__)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class DistributedContext:
    """Unified owner of distributed initialization, collectives, and experiment utilities."""

    def __init__(self, world_size: int) -> None:
        self.requested_world_size = max(1, world_size)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")
        self._initialized = False
        self._is_spawn_parent = False
        self._wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        self._metadata: Optional[ModelFlatMetadata] = None

    @property
    def is_distributed(self) -> bool:
        """Return True if the current process participates in distributed training."""
        return self._initialized and self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Return True when this process is the designated main rank."""
        return self.rank == 0

    @property
    def metadata(self) -> ModelFlatMetadata | None:
        """Return cached flattening metadata, if any."""
        return self._metadata

    def launch(self, fn: Callable[[DistributedContext, Any], Any], config: Any, log_cfg: Any) -> Any:
        """Launch the provided driver either locally, via torchrun, or by spawning workers."""
        if self.requested_world_size == 1:
            self._setup_single()
            result = fn(self, config)
            self.shutdown()
            return result

        if self._is_torchrun():
            env_world_size = int(os.environ["WORLD_SIZE"])
            if env_world_size != self.requested_world_size:
                raise RuntimeError(
                    "Torchrun world_size "
                    f"({env_world_size}) does not match config num_clients ({self.requested_world_size})"
                )
            self._initialize_from_env()
            result = fn(self, config)
            self.shutdown()
            return result

        self._is_spawn_parent = True
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(_get_free_port()))
        ctx = get_context("spawn")
        queue = ctx.SimpleQueue()
        logger.info(
            "Spawning %d ranks at %s:%s",
            self.requested_world_size,
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )

        spawn(
            DistributedContext._worker_entry,
            args=(self.requested_world_size, fn, config, queue, log_cfg),
            nprocs=self.requested_world_size,
            join=True,
        )

        result: Any | None = None
        for _ in range(self.requested_world_size):
            message = queue.get()
            if isinstance(message, BaseException):
                raise message
            if message is not None:
                result = message

        return result

    @staticmethod
    def _worker_entry(
        rank: int,
        world_size: int,
        fn: Callable[[DistributedContext, Any], Any],
        config: Any,
        queue: Any,
        log_cfg: Any,
    ) -> None:
        # Configure logging for this worker
        if log_cfg is not None:
            import copy
            import logging.config
            lc = copy.deepcopy(log_cfg)
            logging.config.dictConfig(lc)
        else:
            # fallback
            import logging as _logging
            import sys

            _logging.basicConfig(
                level=_logging.INFO,
                format=f"[%(asctime)s] [rank={rank}] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[_logging.StreamHandler(sys.stdout)],
                force=True,
            )
        # dump a python stack if we hit SIGBUS/SIGSEGV
        import faulthandler
        faulthandler.enable(all_threads=True)

        install_exception_handlers(threading_=True)
        install_global_log_context()
        set_log_context(rank=rank, world_size=world_size)

        ctx = DistributedContext(world_size)
        ctx._initialize_worker(rank, world_size)
        try:
            result = fn(ctx, config)
            if rank == 0:
                queue.put(result)
            else:
                queue.put(None)
        except Exception as exc:  # surfaced to parent
            if rank == 0:
                queue.put(exc)
            raise
        finally:
            ctx.shutdown()


    def _initialize_worker(self, rank: int, world_size: int) -> None:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", str(rank))
        self._initialize_from_env()

    def _is_torchrun(self) -> bool:
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ

    def _setup_single(self) -> None:
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = self._select_device(0)
        self._initialized = True

    def _initialize_from_env(self) -> None:
        if self._initialized:
            return
        # We currently do all collectives on CPU (flattened model params), so gloo is correct although yes odd
        backend = "gloo"
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        self.device = self._select_device(rank)
        self._initialized = True
        logger.info("Initialized distributed backend=%s rank=%d world_size=%d", backend, rank, world_size)
        set_log_context(backend=backend)

    def _select_device(self, rank: int) -> torch.device:
        if torch.cuda.is_available():
            num_devices = max(1, torch.cuda.device_count())
            device_index = rank % num_devices
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device)
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover - platform specific
            return torch.device("mps")
        return torch.device("cpu")

    def shutdown(self) -> None:
        """Tear down any initialized process group."""
        if self._initialized and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    def barrier(self) -> None:
        """Synchronize all ranks."""
        if self.is_distributed:
            dist.barrier()

    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from the given source rank."""
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        if self.is_distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def all_gather_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all ranks."""
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        if not self.is_distributed:
            return [tensor]
        gather = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gather, tensor)
        return gather

    def reduce_scatter_tensor(self, output: torch.Tensor, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Reduce-scatter a list of tensors."""
        if not self.is_distributed:
            output.copy_(chunks[0])
            return output
        dist.reduce_scatter_tensor(output, chunks)
        return output

    def wandb_init(self, cfg: Any) -> None:
        """Initialize a wandb run on the main rank."""
        if not self.is_main:
            return
        self._wandb_run = wandb.init(
            entity="fedmat-team",
            project="fedmat-project",
            name=getattr(cfg, "run_name", None),
            config=asdict(cfg),
            mode="online",
            dir=str(cfg.output_dir),
        )

    def wandb_log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalars to wandb on the main rank."""
        if self.is_main and self._wandb_run is not None:
            self._wandb_run.log(data, step=step)

    def wandb_update_summary(self, key: str, value: Any) -> None:
        """Update wandb summary values on the main rank."""
        if self.is_main and self._wandb_run is not None:
            self._wandb_run.summary[key] = value

    def wandb_log_artifact(self, name: str, path: os.PathLike[str] | str, artifact_type: str = "model") -> None:
        """Log a file artifact to wandb."""
        if self.is_main and self._wandb_run is not None:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)

    def wandb_finish(self) -> None:
        """Finalize the wandb run on the main rank."""
        if self.is_main and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def ensure_metadata(self, model: torch.nn.Module) -> ModelFlatMetadata:
        """Return cached flattening metadata, creating it from the model if needed."""
        if self._metadata is None:
            with torch.no_grad():
                state = model.state_dict()
                self._metadata = build_flat_metadata(state)
        return self._metadata

    def flatten_model(self, model: torch.nn.Module) -> torch.Tensor:
        """Flatten the model's state dict to a CPU tensor."""
        metadata = self.ensure_metadata(model)
        state = model.state_dict()
        return flatten_state_dict(state, metadata)

    def unflatten_model(self, model: torch.nn.Module, flat: torch.Tensor) -> None:
        """Load flattened weights back into a model using cached metadata."""
        metadata = self.ensure_metadata(model)
        state = unflatten_state_dict(flat, metadata)
        model.load_state_dict(state)
