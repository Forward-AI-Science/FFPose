"""Training-target generation: encoders for SimCC, MSRA heatmap, UDP heatmap.

These functions take keypoint coordinates in *model-input image space* (after
TopdownAffine has warped the bbox region to the input canvas) and produce
training targets in the format the corresponding head consumes.

Direct port of mmpose's codec encoders; pure numpy.
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Tuple, Union

import numpy as np


# ---- SimCC encoder (RTMPose) ------------------------------------------------

def encode_simcc_gaussian(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    input_size: Tuple[int, int],
    sigma: Tuple[float, float],
    simcc_split_ratio: float = 2.0,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Gaussian SimCC labels per axis.

    Returns:
        (target_x, target_y, keypoint_weights):
          - target_x:  (N, K, Wx)  with ``Wx = input_w * simcc_split_ratio``
          - target_y:  (N, K, Hy)  with ``Hy = input_h * simcc_split_ratio``
          - keypoint_weights: (N, K)
    """
    if keypoints.ndim != 3 or keypoints.shape[-1] < 2:
        raise ValueError(f"keypoints expected (N,K,>=2), got {keypoints.shape}")
    N, K = keypoints.shape[:2]
    w, h = input_size
    W = int(np.around(w * simcc_split_ratio))
    H = int(np.around(h * simcc_split_ratio))

    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.shape == ():
        sigma = np.array([float(sigma), float(sigma)], dtype=np.float32)

    # Map keypoints into SimCC bins (rounded).
    kp_split = np.around(keypoints[..., :2] * simcc_split_ratio).astype(np.int64)
    weights = keypoints_visible.copy().astype(np.float32)

    target_x = np.zeros((N, K, W), dtype=np.float32)
    target_y = np.zeros((N, K, H), dtype=np.float32)

    radius = sigma * 3
    grid_x = np.arange(W, dtype=np.float32)
    grid_y = np.arange(H, dtype=np.float32)

    for n, k in product(range(N), range(K)):
        if keypoints_visible[n, k] < 0.5:
            continue
        mu = kp_split[n, k]
        left = mu[0] - radius[0]
        top = mu[1] - radius[1]
        right = mu[0] + radius[0] + 1
        bottom = mu[1] + radius[1] + 1
        if left >= W or top >= H or right < 0 or bottom < 0:
            weights[n, k] = 0
            continue
        target_x[n, k] = np.exp(-((grid_x - mu[0]) ** 2) / (2 * sigma[0] ** 2))
        target_y[n, k] = np.exp(-((grid_y - mu[1]) ** 2) / (2 * sigma[1] ** 2))

    if normalize:
        norm = sigma * np.sqrt(2 * np.pi)
        target_x /= norm[0]
        target_y /= norm[1]

    return target_x, target_y, weights


# ---- MSRA / SimpleBaselines / HRNet heatmap encoder -------------------------

def encode_msra_heatmap(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    input_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    sigma: float,
    unbiased: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """MSRA-style Gaussian heatmap targets in heatmap-space coordinates.

    Returns (heatmaps shape (K, H, W), keypoint_weights shape (N, K)).
    Note that keypoint coords are first divided by ``input_size / heatmap_size``
    to map to heatmap space.
    """
    if keypoints.shape[0] != 1:
        raise ValueError("MSRA encoder expects single-instance keypoints (N=1)")

    scale = (np.array(input_size, dtype=np.float32)
             / np.array(heatmap_size, dtype=np.float32))
    kp_hm = keypoints[..., :2] / scale  # (1, K, 2)

    if unbiased:
        return _generate_unbiased_gaussian(heatmap_size, kp_hm, keypoints_visible, sigma)
    return _generate_gaussian(heatmap_size, kp_hm, keypoints_visible, sigma)


def _generate_gaussian(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: Union[float, Tuple[float, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    N, K, _ = keypoints.shape
    W, H = heatmap_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    weights = keypoints_visible.copy().astype(np.float32)
    if isinstance(sigma, (int, float)):
        sigma_per_n = (float(sigma),) * N
    else:
        sigma_per_n = tuple(sigma)

    for n in range(N):
        radius = int(sigma_per_n[n] * 3)
        size = 2 * radius + 1
        x = np.arange(size, dtype=np.float32)
        y = x[:, None]
        x0 = y0 = size // 2

        for k in range(K):
            if keypoints_visible[n, k] < 0.5:
                continue
            mu = (keypoints[n, k] + 0.5).astype(np.int64)
            left, top = mu - radius
            right, bottom = mu + radius + 1
            if left >= W or top >= H or right < 0 or bottom < 0:
                weights[n, k] = 0
                continue
            gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma_per_n[n] ** 2))
            gx1, gx2 = max(0, -left), min(W, right) - left
            gy1, gy2 = max(0, -top), min(H, bottom) - top
            hx1, hx2 = max(0, left), min(W, right)
            hy1, hy2 = max(0, top), min(H, bottom)
            region = heatmaps[k, hy1:hy2, hx1:hx2]
            np.maximum(region, gauss[gy1:gy2, gx1:gx2], out=region)

    return heatmaps, weights


def _generate_unbiased_gaussian(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """DARK-style: floating-point keypoint center (no rounding)."""
    N, K, _ = keypoints.shape
    W, H = heatmap_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    weights = keypoints_visible.copy().astype(np.float32)
    radius = sigma * 3
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)[:, None]
    for n, k in product(range(N), range(K)):
        if keypoints_visible[n, k] < 0.5:
            continue
        mu = keypoints[n, k]
        left, top = mu - radius
        right, bottom = mu + radius + 1
        if left >= W or top >= H or right < 0 or bottom < 0:
            weights[n, k] = 0
            continue
        gauss = np.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * sigma ** 2))
        np.maximum(gauss, heatmaps[k], out=heatmaps[k])
    return heatmaps, weights


# ---- UDP heatmap encoder (ViTPose) ------------------------------------------

def encode_udp_heatmap(
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    input_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """UDP-style Gaussian heatmap targets — preserves sub-pixel offset of the
    keypoint relative to the rounded grid cell. Required to match ViTPose's
    decoded predictions during validation.
    """
    if keypoints.shape[0] != 1:
        raise ValueError("UDP encoder expects single-instance keypoints (N=1)")

    # UDP warp produces output in [0, output-1] coords (note ``- 1``). Map
    # input-image coords to heatmap coords using the same convention.
    in_arr = np.array(input_size, dtype=np.float32)
    hm_arr = np.array(heatmap_size, dtype=np.float32)
    kp_hm = keypoints[..., :2] / (in_arr - 1) * (hm_arr - 1)
    return _generate_udp_gaussian(heatmap_size, kp_hm, keypoints_visible, sigma)


def _generate_udp_gaussian(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    N, K, _ = keypoints.shape
    W, H = heatmap_size
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    weights = keypoints_visible.copy().astype(np.float32)
    radius = int(sigma * 3)
    size = 2 * radius + 1
    x = np.arange(size, dtype=np.float32)
    y = x[:, None]

    for n, k in product(range(N), range(K)):
        if keypoints_visible[n, k] < 0.5:
            continue
        mu = (keypoints[n, k] + 0.5).astype(np.int64)
        left, top = mu - radius
        right, bottom = mu + radius + 1
        if left >= W or top >= H or right < 0 or bottom < 0:
            weights[n, k] = 0
            continue

        # Sub-pixel offset preserved by shifting the gaussian center.
        mu_ac = keypoints[n, k]
        x0 = size // 2 + (mu_ac[0] - mu[0])
        y0 = size // 2 + (mu_ac[1] - mu[1])
        gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        gx1, gx2 = max(0, -left), min(W, right) - left
        gy1, gy2 = max(0, -top), min(H, bottom) - top
        hx1, hx2 = max(0, left), min(W, right)
        hy1, hy2 = max(0, top), min(H, bottom)
        region = heatmaps[k, hy1:hy2, hx1:hx2]
        np.maximum(region, gauss[gy1:gy2, gx1:gx2], out=region)

    return heatmaps, weights
