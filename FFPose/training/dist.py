"""Tiny distributed-training helpers (DDP via torchrun).

The training scripts opt into DDP by adding ``--ddp`` (or by setting the
``LOCAL_RANK`` env var, which torchrun sets automatically). This module wraps
the small amount of bookkeeping the existing ``PoseTrainer`` needs:

  - ``init_distributed()`` -> bool: ``True`` if running under torchrun.
  - ``DistEnv``: parsed env (rank, world_size, local_rank, device).
  - ``maybe_wrap_ddp(model)``: wraps in ``DistributedDataParallel`` if needed.
  - ``make_train_sampler(dataset)`` -> ``DistributedSampler`` or ``None``.
  - ``is_main_process()``: True on rank 0 (or in single-process runs).

Usage from a training script:
    env = init_distributed()
    train_sampler = make_train_sampler(train_ds, env)
    train_loader = DataLoader(train_ds, sampler=train_sampler,
                              shuffle=(train_sampler is None), ...)
    model = maybe_wrap_ddp(model.to(env.device), env)
    trainer = PoseTrainer(..., dist_env=env)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


def _read_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def init_distributed(backend: Optional[str] = None) -> DistEnv:
    """Initialize the default process group if torchrun env vars are set.

    Returns a DistEnv describing the current process. If ``LOCAL_RANK`` is not
    set, treats the run as single-process and returns ``enabled=False``.
    """
    if "LOCAL_RANK" not in os.environ:
        return DistEnv(rank=0, world_size=1, local_rank=0,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       enabled=False)

    rank = _read_env("RANK", 0)
    world_size = _read_env("WORLD_SIZE", 1)
    local_rank = _read_env("LOCAL_RANK", 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if backend is None:
        # Default: NCCL on CUDA, gloo on CPU. Override via the
        # ``MMPOSE_LITE_BACKEND`` env var (e.g., ``gloo`` to bypass NCCL on
        # hosts where the NCCL shim is misconfigured).
        backend = os.environ.get(
            "MMPOSE_LITE_BACKEND",
            "nccl" if torch.cuda.is_available() else "gloo",
        )
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    return DistEnv(rank=rank, world_size=world_size, local_rank=local_rank,
                   device=device, enabled=True)


def is_main_process(env: Optional[DistEnv] = None) -> bool:
    if env is None:
        return not dist.is_initialized() or dist.get_rank() == 0
    return env.rank == 0


def maybe_wrap_ddp(
    model: nn.Module,
    env: DistEnv,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not env.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[env.local_rank] if env.device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def make_train_sampler(dataset, env: DistEnv, shuffle: bool = True) -> Optional[DistributedSampler]:
    if not env.enabled:
        return None
    return DistributedSampler(
        dataset, num_replicas=env.world_size, rank=env.rank,
        shuffle=shuffle, drop_last=True,
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float) -> float:
    """All-reduce a Python scalar across ranks (mean). Useful for logging."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())
