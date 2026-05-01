"""LR schedulers and parameter-group builders.

Includes:
  - ``build_lr_scheduler``: a single helper that produces the per-step (or per-epoch)
    schedule combinations mmpose configs use:
        LinearLR warmup -> {CosineAnnealingLR | MultiStepLR}
  - ``layer_decay_param_groups``: per-layer LR decay used by ViTPose (rate=0.75)
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
)


def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    total_iters: int,
    warmup_iters: int = 1000,
    warmup_start_factor: float = 1e-5,
    schedule: str = "cosine",      # "cosine" or "multistep"
    cosine_min_factor: float = 0.05,
    multistep_milestones: Optional[List[int]] = None,
    multistep_gamma: float = 0.1,
):
    """Compose a warmup + main schedule, all measured in iterations.

    The returned scheduler must be ``.step()``-ed once per training iteration
    (not per epoch).

    Args:
        total_iters: total optimizer steps in training.
        warmup_iters: number of LinearLR warmup iterations from
            ``warmup_start_factor*lr`` to ``lr``.
        schedule: "cosine" anneals to ``cosine_min_factor*lr``;
                  "multistep" decays at ``multistep_milestones`` (iters).
    """
    if total_iters <= warmup_iters:
        raise ValueError("total_iters must exceed warmup_iters")
    main_iters = total_iters - warmup_iters

    warmup = LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0,
        total_iters=warmup_iters,
    )

    if schedule == "cosine":
        # eta_min relative to base lr per group.
        eta_min = min(g["lr"] * cosine_min_factor for g in optimizer.param_groups)
        main = CosineAnnealingLR(optimizer, T_max=main_iters, eta_min=eta_min)
    elif schedule == "multistep":
        if not multistep_milestones:
            raise ValueError("multistep schedule requires multistep_milestones")
        # Convert epoch-based milestones to iter-based (caller must compute).
        main = MultiStepLR(optimizer, milestones=multistep_milestones, gamma=multistep_gamma)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_iters])


# ---- ViT layer-wise LR decay -----------------------------------------------

def _vit_layer_id(name: str, num_layers: int) -> int:
    """Assign each ViT param a layer id in [0, num_layers + 1].

    - patch_embed / cls_token / pos_embed -> 0
    - layers.{i}.* -> i + 1
    - ln1 (final) / head -> num_layers + 1
    """
    if name.startswith(("backbone.cls_token", "backbone.pos_embed", "backbone.patch_embed")):
        return 0
    if name.startswith("backbone.layers."):
        i = int(name.split(".")[2])
        return i + 1
    return num_layers + 1


def layer_decay_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.1,
    num_layers: int = 12,
    layer_decay_rate: float = 0.75,
    no_weight_decay_names: Iterable[str] = ("bias", "norm", "pos_embed", "cls_token"),
) -> List[dict]:
    """Build optimizer param_groups with per-layer LR scaling for ViT.

    LR for layer ``i`` = ``base_lr * layer_decay_rate ** (num_layers + 1 - i)``.
    Earlier layers receive a smaller LR. Standard practice for fine-tuning
    pretrained ViT backbones (BEiT, MAE, ViTPose all use this).
    """
    no_wd = tuple(no_weight_decay_names)
    groups: dict[tuple[int, bool], dict] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _vit_layer_id(name, num_layers)
        is_no_wd = any(s in name for s in no_wd) or p.ndim == 1
        scale = layer_decay_rate ** (num_layers + 1 - layer_id)
        key = (layer_id, is_no_wd)
        if key not in groups:
            groups[key] = dict(
                params=[],
                lr=base_lr * scale,
                weight_decay=0.0 if is_no_wd else weight_decay,
                lr_scale=scale,  # informational
                layer_id=layer_id,
                is_no_wd=is_no_wd,
            )
        groups[key]["params"].append(p)

    return list(groups.values())
