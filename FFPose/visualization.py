"""Skeleton visualizer using OpenCV.

``draw_skeleton(image, keypoints, scores, schema)`` overlays keypoints and
skeleton edges on a copy of the image. Color convention:

  - left-side keypoints/edges  -> blue
  - right-side keypoints/edges -> red
  - midline (e.g. nose, hips)  -> green

Per-keypoint scores below ``score_thr`` are skipped.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .skeletons import (
    COCO_17,
    KeypointSchema,
    is_left_keypoint,
    is_right_keypoint,
)


# BGR (cv2 convention)
_LEFT = (255, 128, 0)    # cyan-ish blue
_RIGHT = (0, 0, 255)     # red
_MIDLINE = (0, 255, 0)   # green


def _kpt_color(schema: KeypointSchema, idx: int) -> Tuple[int, int, int]:
    if is_left_keypoint(schema, idx):
        return _LEFT
    if is_right_keypoint(schema, idx):
        return _RIGHT
    return _MIDLINE


def _edge_color(
    schema: KeypointSchema, a: int, b: int
) -> Tuple[int, int, int]:
    # An edge that has any left-side endpoint -> left color (and same for right).
    a_l, b_l = is_left_keypoint(schema, a), is_left_keypoint(schema, b)
    a_r, b_r = is_right_keypoint(schema, a), is_right_keypoint(schema, b)
    if a_l or b_l:
        return _LEFT
    if a_r or b_r:
        return _RIGHT
    return _MIDLINE


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    schema: KeypointSchema = COCO_17,
    *,
    score_thr: float = 0.3,
    radius: int = 4,
    edge_thickness: int = 2,
    inplace: bool = False,
) -> np.ndarray:
    """Draw a single instance's keypoints + skeleton edges on a BGR image.

    Args:
        image: HxWx3 BGR uint8.
        keypoints: (K, 2) coords in image-space pixels.
        scores: (K,) per-keypoint confidence.
        schema: keypoint schema (defines edges and left/right side).
        score_thr: skip keypoints (and edges referencing them) with score below this.
        radius: keypoint dot radius in px.
        edge_thickness: skeleton edge thickness in px.
        inplace: if True, draws onto ``image`` directly.

    Returns:
        The annotated image (same array if ``inplace=True``).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    if keypoints.shape[0] != schema.num_keypoints:
        raise ValueError(
            f"keypoints has {keypoints.shape[0]} rows; "
            f"schema {schema.name!r} expects {schema.num_keypoints}"
        )
    canvas = image if inplace else image.copy()

    # Edges first so points are drawn on top.
    for a, b in schema.skeleton:
        if a >= len(scores) or b >= len(scores):
            continue
        if scores[a] < score_thr or scores[b] < score_thr:
            continue
        pa = tuple(int(round(v)) for v in keypoints[a])
        pb = tuple(int(round(v)) for v in keypoints[b])
        cv2.line(canvas, pa, pb, _edge_color(schema, a, b), thickness=edge_thickness, lineType=cv2.LINE_AA)

    for k in range(schema.num_keypoints):
        if scores[k] < score_thr:
            continue
        p = tuple(int(round(v)) for v in keypoints[k])
        cv2.circle(canvas, p, radius, _kpt_color(schema, k), thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def draw_skeletons(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    schema: KeypointSchema = COCO_17,
    *,
    boxes: Optional[np.ndarray] = None,
    box_color: Tuple[int, int, int] = (0, 255, 255),
    box_thickness: int = 1,
    score_thr: float = 0.3,
    radius: int = 4,
    edge_thickness: int = 2,
) -> np.ndarray:
    """Draw multiple instances. ``keypoints`` is (N, K, 2), ``scores`` is (N, K).

    Optionally also draws bbox rectangles if ``boxes`` is (N, 4) xyxy.
    """
    canvas = image.copy()
    if keypoints.ndim != 3:
        raise ValueError(f"expected (N, K, 2), got {keypoints.shape}")
    n = keypoints.shape[0]
    if boxes is not None:
        for i in range(n):
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[i])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color,
                           thickness=box_thickness, lineType=cv2.LINE_AA)
    for i in range(n):
        draw_skeleton(canvas, keypoints[i], scores[i], schema=schema,
                      score_thr=score_thr, radius=radius,
                      edge_thickness=edge_thickness, inplace=True)
    return canvas
