"""Top-down preprocessing: bbox -> center/scale -> affine warp -> normalized tensor.

Direct port of:
  - mmpose.structures.bbox.transforms.bbox_xyxy2cs / get_warp_matrix
  - mmpose.datasets.transforms.GetBBoxCenterScale + TopdownAffine

Plus an inverse warp utility for mapping decoded keypoints back to image space.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
    return rot @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]])


def bbox_xyxy_to_center_scale(
    bbox: np.ndarray, padding: float = 1.25
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (x1, y1, x2, y2) bbox(es) to center, scale (w, h * padding)."""
    bbox = np.asarray(bbox, dtype=np.float32)
    single = bbox.ndim == 1
    if single:
        bbox = bbox[None]
    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
    if single:
        return center[0], scale[0]
    return center, scale


def fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Pad the bbox scale (w, h) to match a target w/h aspect ratio."""
    scale = np.asarray(scale, dtype=np.float32)
    single = scale.ndim == 1
    if single:
        scale = scale[None]
    w, h = scale[..., :1], scale[..., 1:]
    out = np.where(
        w > h * aspect_ratio,
        np.concatenate([w, w / aspect_ratio], axis=-1),
        np.concatenate([h * aspect_ratio, h], axis=-1),
    )
    return out[0] if single else out


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    inv: bool = False,
) -> np.ndarray:
    """2x3 affine matrix mapping the bbox region (defined by center/scale/rot)
    to a canvas of ``output_size`` (w, h). With ``inv=True``, returns the
    inverse mapping (canvas -> image)."""
    src_w, src_h = float(scale[0]), float(scale[1])
    dst_w, dst_h = float(output_size[0]), float(output_size[1])

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.0], dtype=np.float32), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def topdown_affine(
    img: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_size: Tuple[int, int],
    rot: float = 0.0,
    use_udp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine-warp ``img`` so the (center, scale) bbox fills ``input_size``.

    Args:
        use_udp: if True, use the UDP warp matrix (output in [0, w-1] coords).

    Returns (warped image HxWx3, fixed scale, warp matrix).
    """
    w, h = input_size
    scale = fix_aspect_ratio(scale, aspect_ratio=w / h)
    if use_udp:
        from .heatmap_codec import get_udp_warp_matrix
        warp_mat = get_udp_warp_matrix(center, scale, rot, output_size=(w, h))
    else:
        warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))
    warped = cv2.warpAffine(img, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    return warped, scale, warp_mat


# ImageNet-style stats used by RTMPose (config: bgr_to_rgb=True)
DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def normalize_to_tensor(
    img_rgb: np.ndarray,
    mean: np.ndarray = DEFAULT_MEAN,
    std: np.ndarray = DEFAULT_STD,
) -> np.ndarray:
    """HxWx3 uint8/float RGB -> 1x3xHxW float32, mean/std normalized."""
    arr = img_rgb.astype(np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(arr)


def keypoints_from_input_to_image(
    keypoints_input: np.ndarray, warp_mat: np.ndarray
) -> np.ndarray:
    """Apply the inverse of ``warp_mat`` to map keypoints from the
    model-input canvas back to the original image."""
    inv_mat = cv2.invertAffineTransform(warp_mat)
    pts = np.asarray(keypoints_input, dtype=np.float32)
    flat = pts.reshape(-1, 2)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    out = (inv_mat @ homog.T).T
    return out.reshape(pts.shape)
