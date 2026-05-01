"""Test-time augmentation utilities (horizontal flip)."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def flip_heatmaps(
    heatmaps: Tensor,
    flip_indices: Sequence[int],
    shift_heatmap: bool = False,
) -> Tensor:
    """Spatially-flip heatmaps and swap paired-keypoint channels.

    For top-down heatmap models, predictions on the horizontally-flipped image
    are flipped back along the width axis and re-indexed so that left/right
    keypoint channels align with the original image.

    Args:
        heatmaps: ``(B, K, H, W)`` heatmaps from the flipped input.
        flip_indices: per-keypoint symmetric index (``flip_indices[k]`` is the
            channel that becomes channel ``k`` after flipping).
        shift_heatmap: if True, shift the flipped heatmaps by 1 pixel right.
            mmpose enables this for HRNet/SimpleBaselines (config: ``shift_heatmap=True``)
            because the heatmap argmax tends to land 0.5 pixels left of the
            center after flipping.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"expected (B,K,H,W), got {tuple(heatmaps.shape)}")
    if len(flip_indices) != heatmaps.shape[1]:
        raise ValueError(
            f"flip_indices length {len(flip_indices)} != num channels {heatmaps.shape[1]}"
        )
    flipped = heatmaps.flip(-1)
    flipped = flipped[:, list(flip_indices)]
    if shift_heatmap:
        # Slide values right by 1 column. Column 0 keeps its original value
        # (matches mmpose's flip_heatmaps: ``hm[..., 1:] = hm[..., :-1].clone()``).
        flipped = flipped.clone()
        flipped[..., 1:] = flipped[..., :-1].clone()
    return flipped


def flip_simcc_vectors(
    pred_x: Tensor, pred_y: Tensor, flip_indices: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Mirror SimCC x-axis bins and re-index paired keypoint channels.

    ``pred_x`` is shape ``(B, K, Wx)`` and ``pred_y`` is ``(B, K, Wy)``. Only
    the x-axis logits are spatially flipped (since horizontal image flip
    only affects x).
    """
    if pred_x.ndim != 3 or pred_y.ndim != 3:
        raise ValueError("expected (B, K, W) tensors")
    if len(flip_indices) != pred_x.shape[1]:
        raise ValueError("flip_indices does not match number of channels")
    pred_x = pred_x[:, list(flip_indices)].flip(-1)
    pred_y = pred_y[:, list(flip_indices)]
    return pred_x, pred_y
