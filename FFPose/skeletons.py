"""Keypoint metadata: names, left/right pair indices, skeleton edges.

These are used at:
- inference time, for test-time horizontal flip (paired keypoint swap)
- training time, for the RandomFlip augmentation
- visualization, to draw skeleton edges
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class KeypointSchema:
    name: str
    num_keypoints: int
    keypoint_names: Tuple[str, ...]
    flip_indices: Tuple[int, ...]    # flip_indices[i] is the symmetric kp of i
    skeleton: Tuple[Tuple[int, int], ...]  # edges for visualization


# COCO body 17 keypoints
COCO_17 = KeypointSchema(
    name="coco_body_17",
    num_keypoints=17,
    keypoint_names=(
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ),
    flip_indices=(
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
    ),
    skeleton=(
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
        (1, 3), (2, 4), (3, 5), (4, 6),
    ),
)


def _coco_wholebody_flip_indices() -> Tuple[int, ...]:
    """COCO-Wholebody 133-keypoint flip indices.

    Indexing follows the COCO-Wholebody schema:
      - Body  0..16
      - Foot  17..22  (l_big_toe, l_small_toe, l_heel, r_big_toe, r_small_toe, r_heel)
      - Face  23..90  (68 face landmarks)
      - L hand 91..111 (21)
      - R hand 112..132 (21)
    """
    out: List[int] = list(COCO_17.flip_indices)
    # Foot: 17<->20, 18<->21, 19<->22
    out += [20, 21, 22, 17, 18, 19]
    # Face 68: pair indices follow standard 68-point flip table.
    # Source: mmpose/datasets/datasets/_base_/coco_wholebody.py
    face_flip = [
        # 0..16 jawline (mirrored 16..0)
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        # 17..21 right brow / 22..26 left brow -> swap
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        # 27..30 nose bridge stays
        27, 28, 29, 30,
        # 31..35 nose bottom (mirror)
        35, 34, 33, 32, 31,
        # 36..41 right eye / 42..47 left eye
        45, 44, 43, 42, 47, 46,   # 36..41 -> 45,44,43,42,47,46
        39, 38, 37, 36, 41, 40,   # 42..47 -> 39,38,37,36,41,40
        # 48..59 outer mouth (mirror)
        54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
        # 60..67 inner mouth (mirror)
        64, 63, 62, 61, 60, 67, 66, 65,
    ]
    out += [23 + i for i in face_flip]
    # Hands: l_hand[91..111] <-> r_hand[112..132], same fingertip ordering.
    out += list(range(112, 133)) + list(range(91, 112))
    return tuple(out)


COCO_WHOLEBODY_133 = KeypointSchema(
    name="coco_wholebody_133",
    num_keypoints=133,
    keypoint_names=tuple([f"kp_{i}" for i in range(133)]),  # detailed names omitted
    flip_indices=_coco_wholebody_flip_indices(),
    skeleton=(),  # large; populate later if needed for visualization
)


SCHEMAS = {
    "coco_body_17": COCO_17,
    "coco_wholebody_133": COCO_WHOLEBODY_133,
}


def is_left_keypoint(schema: KeypointSchema, idx: int) -> bool:
    """True if keypoint ``idx`` is a left-side body part."""
    name = schema.keypoint_names[idx].lower() if idx < len(schema.keypoint_names) else ""
    return name.startswith("left_") or name.startswith("l_")


def is_right_keypoint(schema: KeypointSchema, idx: int) -> bool:
    name = schema.keypoint_names[idx].lower() if idx < len(schema.keypoint_names) else ""
    return name.startswith("right_") or name.startswith("r_")
