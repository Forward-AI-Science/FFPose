"""SimCC decoder: argmax over x/y logits -> pixel coordinates in input image space.

Inference-only; encoding/training-target generation is intentionally omitted.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax-based decode of SimCC representation.

    Args:
        simcc_x: (N, K, Wx) or (K, Wx).
        simcc_y: (N, K, Wy) or (K, Wy).

    Returns:
        locs: (N, K, 2) keypoint coords in SimCC bins.
        vals: (N, K) max-response score (min of per-axis maxima).
    """
    assert simcc_x.ndim == simcc_y.ndim and simcc_x.ndim in (2, 3)
    if simcc_x.ndim == 3:
        N, K, _ = simcc_x.shape
        sx = simcc_x.reshape(N * K, -1)
        sy = simcc_y.reshape(N * K, -1)
    else:
        N, K = None, simcc_x.shape[0]
        sx, sy = simcc_x, simcc_y

    x_locs = np.argmax(sx, axis=1)
    y_locs = np.argmax(sy, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vx = np.amax(sx, axis=1)
    vy = np.amax(sy, axis=1)
    vals = np.minimum(vx, vy)
    locs[vals <= 0.0] = -1

    if N is not None:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
    return locs, vals


class SimCCDecoder:
    """Decode RTMPose SimCC logits to keypoints in the *model input* coord space.

    The downstream caller is responsible for mapping these coordinates back to
    the original image via the inverse of the top-down affine warp.
    """

    def __init__(self, input_size: Tuple[int, int], simcc_split_ratio: float = 2.0):
        self.input_size = input_size  # (w, h)
        self.simcc_split_ratio = simcc_split_ratio

    def decode(
        self, simcc_x: np.ndarray, simcc_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
        if keypoints.ndim == 2:
            keypoints = keypoints[None]
            scores = scores[None]
        keypoints /= self.simcc_split_ratio
        return keypoints, scores
