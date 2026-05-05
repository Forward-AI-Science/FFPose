"""Pose estimation losses (RTMPose KL on SimCC, MSE on heatmaps)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KLDiscretLoss(nn.Module):
    """Discrete KL divergence between predicted (log-)softmax SimCC and a
    Gaussian-smoothed label distribution. Used by RTMPose.

    Direct port of mmpose's ``KLDiscretLoss``. ``label_softmax=True`` and
    ``beta`` are the standard RTMPose settings (config: ``beta=10.``,
    ``label_softmax=True``).
    """

    def __init__(
        self,
        beta: float = 1.0,
        label_softmax: bool = False,
        label_beta: float = 10.0,
        use_target_weight: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight

    def _criterion(self, pred: Tensor, label: Tensor) -> Tensor:
        log_p = F.log_softmax(pred * self.beta, dim=1)
        if self.label_softmax:
            label = F.softmax(label * self.label_beta, dim=1)
        # KLDivLoss expects log-probs and probs; reduction='none' -> per-bin loss.
        return F.kl_div(log_p, label, reduction="none").mean(dim=1)

    def forward(
        self,
        pred_simcc: tuple[Tensor, Tensor],
        gt_simcc: tuple[Tensor, Tensor],
        target_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Args:
            pred_simcc: (pred_x, pred_y), each (B, K, Wx) / (B, K, Hy).
            gt_simcc:   (gt_x, gt_y).
            target_weight: (B, K) per-keypoint weight (1 = visible, 0 = invisible).
        """
        N, K, _ = pred_simcc[0].shape
        if self.use_target_weight and target_weight is not None:
            weight = target_weight.reshape(-1)
        else:
            weight = 1.0

        loss = pred_simcc[0].new_zeros(())
        for pred, target in zip(pred_simcc, gt_simcc):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))
            t_loss = self._criterion(pred, target) * weight
            loss = loss + t_loss.sum()
        return loss / K


class KeypointMSELoss(nn.Module):
    """Pixel-wise MSE between predicted and target heatmaps, optionally weighted
    by per-keypoint visibility. Used by HRNet, ViTPose, etc.
    """

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(
        self,
        output: Tensor,            # (B, K, H, W)
        target: Tensor,            # (B, K, H, W)
        target_weights: Optional[Tensor] = None,  # (B, K) or (B, K, H, W)
    ) -> Tensor:
        mask = self._build_mask(target, target_weights)
        if mask is None:
            loss = F.mse_loss(output, target)
        else:
            loss = F.mse_loss(output, target, reduction="none")
            loss = (loss * mask).mean()
        return loss * self.loss_weight

    def _build_mask(
        self, target: Tensor, target_weights: Optional[Tensor]
    ) -> Optional[Tensor]:
        m: Optional[Tensor] = None
        if self.use_target_weight and target_weights is not None:
            # Reshape to (B, K, 1, 1) so it broadcasts to (B, K, H, W).
            ndim_pad = target.ndim - target_weights.ndim
            m = target_weights.view(target_weights.shape + (1,) * ndim_pad)
        if self.skip_empty_channel:
            empty = (target.amax(dim=(-2, -1), keepdim=True) == 0)
            cond = (~empty).to(target.dtype)
            m = cond if m is None else m * cond
        return m
