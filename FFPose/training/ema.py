"""Exponential Moving Average for model parameters.

Direct port of mmpose's ExpMomentumEMA + the relevant base behavior of
mmengine's ExponentialMovingAverage. Standalone — no mmengine dependency.

Used by RTMPose configs (``momentum=0.0002``, ``update_buffers=True``).
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ExpMomentumEMA(nn.Module):
    """EMA with momentum that anneals from a high value (warmup) to ``momentum``.

    Update rule per step (after ``interval`` steps between updates):
        m = (1 - momentum) * exp(-(1 + steps) / gamma) + momentum
        ema_param = (1 - m) * ema_param + m * source_param

    With ``gamma=2000`` and the default ``momentum=0.0002`` this means the
    averaged weights track the live model closely at first and stabilize over
    a few thousand iterations.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.0002,
        gamma: int = 2000,
        interval: int = 1,
        device: Optional[torch.device] = None,
        update_buffers: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < momentum < 1.0:
            raise ValueError("momentum must be in (0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

        self.momentum = float(momentum)
        self.gamma = int(gamma)
        self.interval = int(interval)
        self.update_buffers = bool(update_buffers)
        # ``steps`` counts EMA updates (not optimizer steps) — matches mmengine.
        self.register_buffer("steps", torch.zeros(1, dtype=torch.long))

    def _avg_factor(self, steps: int) -> float:
        return (1.0 - self.momentum) * math.exp(-(1 + steps) / self.gamma) + self.momentum

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        """Run one EMA update step. Call after each optimizer step."""
        self.steps += 1
        if int(self.steps.item()) % self.interval != 0:
            return
        m = self._avg_factor(int(self.steps.item()))

        # Parameters
        for p_ema, p_src in zip(self.module.parameters(), source.parameters()):
            p_ema.data.mul_(1.0 - m).add_(p_src.data, alpha=m)

        # Buffers (BN running stats etc.)
        if self.update_buffers:
            for b_ema, b_src in zip(self.module.buffers(), source.buffers()):
                if b_ema.dtype == b_src.dtype and b_ema.is_floating_point():
                    b_ema.data.mul_(1.0 - m).add_(b_src.data, alpha=m)
                else:
                    b_ema.data.copy_(b_src.data)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
