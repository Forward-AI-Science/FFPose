"""UDP-style heatmap decoding (argmax + Dark UDP modulation).

Direct port of:
    mmpose.codecs.utils.post_processing.{get_heatmap_maximum, gaussian_blur}
    mmpose.codecs.utils.refinement.refine_keypoints_dark_udp
    mmpose.codecs.UDPHeatmap.decode

UDP also requires its own bbox warp matrix; see :func:`get_udp_warp_matrix`.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


# ---- post-processing primitives ---------------------------------------------

def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax per keypoint over (H, W). Accepts (K, H, W) or (B, K, H, W)."""
    assert heatmaps.ndim in (3, 4), f"invalid shape {heatmaps.shape}"
    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        flat = heatmaps.reshape(K, -1)
        B = None
    else:
        B, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(flat, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(flat, axis=1)
    locs[vals <= 0.0] = -1
    if B is not None:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)
    return locs, vals


def _gaussian_blur(heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    """Per-keypoint Gaussian blur preserving each map's max value (mmpose port)."""
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    out = heatmaps.copy()
    for k in range(K):
        origin_max = np.max(out[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = out[k]
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        out[k] = dr[border:-border, border:-border]
        m = np.max(out[k])
        if m > 0:
            out[k] *= origin_max / m
    return out


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Sub-pixel refinement using Taylor expansion of log-heatmap (UDP variant)."""
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    heatmaps = _gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)

    out = keypoints.copy()
    for n in range(N):
        # Index of each kp's "center" cell in the flattened padded array.
        idx = out[n, :, 0] + 1 + (out[n, :, 1] + 1) * (W + 2)
        idx += (W + 2) * (H + 2) * np.arange(0, K)
        idx = idx.astype(int).reshape(-1, 1)

        i_     = pad[idx]
        ix1    = pad[idx + 1]
        iy1    = pad[idx + (W + 2)]
        ix1y1  = pad[idx + (W + 3)]
        ix1_y1 = pad[idx - (W + 3)]
        ix1_   = pad[idx - 1]
        iy1_   = pad[idx - (W + 2)]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1).reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + 2 * i_ - ix1_ - iy1_ + ix1_y1)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        out[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()
    return out


# ---- UDP-aware bbox warp matrix --------------------------------------------

def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """UDP affine matrix mapping bbox -> (output_w-1, output_h-1) canvas.

    Direct port of mmpose.structures.bbox.transforms.get_udp_warp_matrix.
    """
    input_size = center * 2  # quirk preserved from mmpose
    rot_rad = np.deg2rad(rot)
    warp = np.zeros((2, 3), dtype=np.float32)
    sx = (output_size[0] - 1) / scale[0]
    sy = (output_size[1] - 1) / scale[1]
    cs, sn = math.cos(rot_rad), math.sin(rot_rad)
    warp[0, 0] = cs * sx
    warp[0, 1] = -sn * sx
    warp[0, 2] = sx * (-0.5 * input_size[0] * cs + 0.5 * input_size[1] * sn + 0.5 * scale[0])
    warp[1, 0] = sn * sy
    warp[1, 1] = cs * sy
    warp[1, 2] = sy * (-0.5 * input_size[0] * sn - 0.5 * input_size[1] * cs + 0.5 * scale[1])
    return warp


# ---- decoder used at inference ---------------------------------------------

def refine_keypoints(keypoints: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Sub-pixel refinement: shift each keypoint by 0.25 toward the higher
    neighbor along x and y. Direct port of mmpose's basic refine_keypoints.
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    out = keypoints.copy()
    for n in range(N):
        for k in range(K):
            x, y = int(out[n, k, 0]), int(out[n, k, 1])
            dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1] if 1 < x < W - 1 and 0 < y < H else 0.0
            dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x] if 1 < y < H - 1 and 0 < x < W else 0.0
            out[n, k] += np.sign([dx, dy], dtype=np.float32) * 0.25
    return out


def refine_keypoints_dark(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """DARK Taylor sub-pixel refinement on log-Gaussian-blurred heatmap."""
    from itertools import product
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    heatmaps = _gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)
    out = keypoints.copy()
    for n, k in product(range(N), range(K)):
        x, y = int(out[n, k, 0]), int(out[n, k, 1])
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
            dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])
            dxx = 0.25 * (heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] + heatmaps[k, y, x - 2])
            dxy = 0.25 * (heatmaps[k, y + 1, x + 1] - heatmaps[k, y - 1, x + 1]
                          - heatmaps[k, y + 1, x - 1] + heatmaps[k, y - 1, x - 1])
            dyy = 0.25 * (heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] + heatmaps[k, y - 2, x])
            det = dxx * dyy - dxy ** 2
            if det != 0:
                hess_inv = np.linalg.inv(np.array([[dxx, dxy], [dxy, dyy]]))
                offset = -hess_inv @ np.array([dx, dy])
                out[n, k] += offset.astype(np.float32)
    return out


class MSRAHeatmapDecoder:
    """Standard heatmap decode used by HRNet/SimpleBaselines.

    Decoded coords are in the model-input canvas (not heatmap space). Map
    back to image coords with the inverse of the topdown affine warp.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        unbiased: bool = False,
        blur_kernel_size: int = 11,
    ) -> None:
        self.input_size = np.array(input_size, dtype=np.float32)   # (w, h)
        self.heatmap_size = np.array(heatmap_size, dtype=np.float32)  # (w, h)
        self.unbiased = unbiased
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (self.input_size / self.heatmap_size).astype(np.float32)

    def decode(self, heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if heatmaps.ndim != 3:
            raise ValueError(f"expected (K,H,W) heatmaps, got {heatmaps.shape}")
        keypoints, scores = get_heatmap_maximum(heatmaps)
        keypoints, scores = keypoints[None], scores[None]
        if self.unbiased:
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps.copy(), blur_kernel_size=self.blur_kernel_size
            )
        else:
            keypoints = refine_keypoints(keypoints, heatmaps)
        keypoints = keypoints * self.scale_factor
        return keypoints, scores


class UDPHeatmapDecoder:
    """Decode (K, H, W) heatmap to (1, K, 2) keypoint coords in input-image space.

    The returned coordinates are in the *model input* canvas; map back to the
    original image with the inverse of the warp matrix used during preprocess.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        blur_kernel_size: int = 11,
    ) -> None:
        self.input_size = input_size       # (w, h)
        self.heatmap_size = heatmap_size   # (w, h)
        self.blur_kernel_size = blur_kernel_size

    def decode(self, heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if heatmaps.ndim != 3:
            raise ValueError(f"expected (K,H,W) heatmaps, got {heatmaps.shape}")
        keypoints, scores = get_heatmap_maximum(heatmaps)
        keypoints = keypoints[None]
        scores = scores[None]
        keypoints = refine_keypoints_dark_udp(
            keypoints, heatmaps.copy(), blur_kernel_size=self.blur_kernel_size
        )
        W, H = self.heatmap_size
        keypoints = keypoints / np.array([W - 1, H - 1], dtype=np.float32) * np.array(self.input_size, dtype=np.float32)
        return keypoints, scores
