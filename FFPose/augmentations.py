"""Top-down keypoint augmentations.

Each augmentation is a callable taking a :class:`Sample` and returning a
:class:`Sample`. Compose with :class:`Pipeline`. Direct port of the
augmentations mmpose uses for top-down keypoint training:

    GetBBoxCenterScale, RandomFlip, RandomHalfBody, RandomBBoxTransform,
    TopdownAffine.

Optional: photometric jitter and Albumentations wrappers are at the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocess import (
    bbox_xyxy_to_center_scale,
    fix_aspect_ratio,
    get_warp_matrix,
)
from .heatmap_codec import get_udp_warp_matrix


# ---- sample container -------------------------------------------------------

@dataclass
class Sample:
    """One training sample passing through the pipeline.

    Coordinates conventions:
      - ``image``: HxWx3 (BGR or RGB depending on the loader)
      - ``bbox_xyxy``: (4,) absolute pixel coords in ``image``
      - ``bbox_center``: (2,) center xy in image space
      - ``bbox_scale``: (2,) wh in image space (already padded by ``GetBBoxCenterScale``)
      - ``bbox_rotation``: scalar degrees, applied during affine warp
      - ``keypoints``: (K, 2) coords, by default in *image* space; rewritten to
        *input* space after :class:`TopdownAffine` runs.
      - ``keypoints_visible``: (K,) 0/1 mask
    """
    image: np.ndarray
    bbox_xyxy: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    keypoints_visible: Optional[np.ndarray] = None
    bbox_center: Optional[np.ndarray] = None
    bbox_scale: Optional[np.ndarray] = None
    bbox_rotation: float = 0.0
    flip_indices: Optional[List[int]] = None
    upper_body_ids: Optional[List[int]] = None
    lower_body_ids: Optional[List[int]] = None
    img_shape: Optional[Tuple[int, int]] = None  # (H, W)
    # filled in after TopdownAffine
    input_image: Optional[np.ndarray] = None
    warp_mat: Optional[np.ndarray] = None
    input_size: Optional[Tuple[int, int]] = None  # (W, H)
    flip: bool = False
    # filled in by GenerateXxxTarget at the end of the train pipeline
    targets: Optional[dict] = None


class Pipeline:
    """Sequentially apply a list of callables to a Sample."""
    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---- transforms -------------------------------------------------------------

class GetBBoxCenterScale:
    """Convert xyxy bbox to (center, scale) with optional padding."""

    def __init__(self, padding: float = 1.25) -> None:
        self.padding = padding

    def __call__(self, s: Sample) -> Sample:
        if s.bbox_center is None:
            assert s.bbox_xyxy is not None, "Need bbox_xyxy before GetBBoxCenterScale"
            c, sc = bbox_xyxy_to_center_scale(s.bbox_xyxy, padding=self.padding)
            s.bbox_center = c
            s.bbox_scale = sc
        if s.img_shape is None:
            s.img_shape = (s.image.shape[0], s.image.shape[1])
        return s


class RandomFlip:
    """Random horizontal flip of image, bbox center, and keypoints (with paired-channel swap)."""

    def __init__(self, prob: float = 0.5) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0,1]")
        self.prob = prob

    def __call__(self, s: Sample) -> Sample:
        if np.random.rand() >= self.prob:
            return s
        if s.flip_indices is None:
            raise ValueError("RandomFlip requires Sample.flip_indices to be set")

        h, w = s.image.shape[:2]
        s.image = s.image[:, ::-1].copy()
        s.flip = True
        # Mirror bbox center x
        if s.bbox_center is not None:
            s.bbox_center = s.bbox_center.copy()
            s.bbox_center[0] = w - 1 - s.bbox_center[0]
        # Mirror keypoints x and reorder paired channels
        if s.keypoints is not None:
            kp = s.keypoints.copy()
            kp[:, 0] = w - 1 - kp[:, 0]
            s.keypoints = kp[list(s.flip_indices)]
            if s.keypoints_visible is not None:
                s.keypoints_visible = s.keypoints_visible[list(s.flip_indices)].copy()
        return s


class RandomHalfBody:
    """Randomly tighten the bbox to upper-body-only or lower-body-only crop."""

    def __init__(
        self,
        min_total_keypoints: int = 9,
        min_upper_keypoints: int = 2,
        min_lower_keypoints: int = 3,
        padding: float = 1.5,
        prob: float = 0.3,
        upper_prioritized_prob: float = 0.7,
    ) -> None:
        self.min_total_keypoints = min_total_keypoints
        self.min_upper_keypoints = min_upper_keypoints
        self.min_lower_keypoints = min_lower_keypoints
        self.padding = padding
        self.prob = prob
        self.upper_prioritized_prob = upper_prioritized_prob

    def __call__(self, s: Sample) -> Sample:
        if s.keypoints is None or s.keypoints_visible is None:
            return s
        if s.upper_body_ids is None or s.lower_body_ids is None:
            return s
        if s.keypoints_visible.sum() < self.min_total_keypoints:
            return s
        if np.random.rand() >= self.prob:
            return s

        upper_valid = [i for i in s.upper_body_ids if s.keypoints_visible[i] > 0]
        lower_valid = [i for i in s.lower_body_ids if s.keypoints_visible[i] > 0]
        prefer_upper = np.random.rand() < self.upper_prioritized_prob

        if (len(upper_valid) < self.min_upper_keypoints
                and len(lower_valid) < self.min_lower_keypoints):
            return s
        if len(lower_valid) < self.min_lower_keypoints:
            indices = upper_valid
        elif len(upper_valid) < self.min_upper_keypoints:
            indices = lower_valid
        else:
            indices = upper_valid if prefer_upper else lower_valid

        kp = s.keypoints[indices]
        x1, y1 = kp.min(axis=0)
        x2, y2 = kp.max(axis=0)
        s.bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        s.bbox_scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * self.padding
        return s


class RandomBBoxTransform:
    """Random shift/scale/rotate jitter of the bbox, applied at warp time."""

    def __init__(
        self,
        shift_factor: float = 0.16,
        shift_prob: float = 0.3,
        scale_factor: Tuple[float, float] = (0.5, 1.5),
        scale_prob: float = 1.0,
        rotate_factor: float = 80.0,
        rotate_prob: float = 0.6,
    ) -> None:
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob

    @staticmethod
    def _truncnorm(low: float = -1.0, high: float = 1.0, size: tuple = ()) -> np.ndarray:
        """Truncated normal in [low, high]. Uses rejection on standard normal."""
        out = np.empty(size, dtype=np.float32) if size else np.zeros((), dtype=np.float32)
        flat = out.reshape(-1)
        n = flat.size
        i = 0
        while i < n:
            need = n - i
            chunk = np.random.randn(need * 2).astype(np.float32)
            keep = chunk[(chunk >= low) & (chunk <= high)]
            take = min(len(keep), need)
            flat[i:i + take] = keep[:take]
            i += take
        return out

    def __call__(self, s: Sample) -> Sample:
        if s.bbox_center is None or s.bbox_scale is None:
            return s
        rv = self._truncnorm(size=(4,))
        offset_v = rv[:2]
        scale_v = rv[2]
        rotate_v = rv[3]

        if np.random.rand() < self.shift_prob:
            offset = offset_v * self.shift_factor
        else:
            offset = np.zeros(2, dtype=np.float32)

        smin, smax = self.scale_factor
        mu = (smax + smin) * 0.5
        sigma = (smax - smin) * 0.5
        scale = float(scale_v * sigma + mu) if np.random.rand() < self.scale_prob else 1.0

        rotate = float(rotate_v * self.rotate_factor) if np.random.rand() < self.rotate_prob else 0.0

        s.bbox_center = s.bbox_center + offset * s.bbox_scale
        s.bbox_scale = s.bbox_scale * scale
        s.bbox_rotation = rotate
        return s


class TopdownAffine:
    """Affine-warp the bbox region into a fixed-size canvas. Also warps keypoints."""

    def __init__(self, input_size: Tuple[int, int], use_udp: bool = False) -> None:
        self.input_size = input_size  # (W, H)
        self.use_udp = use_udp

    def __call__(self, s: Sample) -> Sample:
        if s.bbox_center is None or s.bbox_scale is None:
            raise ValueError("TopdownAffine requires bbox_center and bbox_scale")
        w, h = self.input_size
        scale = fix_aspect_ratio(s.bbox_scale, aspect_ratio=w / h)
        s.bbox_scale = scale

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(s.bbox_center, scale, s.bbox_rotation, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(s.bbox_center, scale, s.bbox_rotation, output_size=(w, h))

        s.input_image = cv2.warpAffine(s.image, warp_mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
        s.warp_mat = warp_mat
        s.input_size = (int(w), int(h))

        if s.keypoints is not None:
            kp = s.keypoints.copy().astype(np.float32)
            ones = np.ones((kp.shape[0], 1), dtype=np.float32)
            homog = np.concatenate([kp, ones], axis=1)
            s.keypoints = (warp_mat @ homog.T).T
        return s


# ---- photometric augmentations ---------------------------------------------

class AlbumentationsWrap:
    """Apply an ``albumentations.Compose`` to ``Sample.image``.

    Used to match RTMPose's training recipe (Blur / MedianBlur / CoarseDropout).
    Pixel-level only — spatial transforms must happen elsewhere (the warp +
    flip handle geometry). Pass an already-built ``albumentations.Compose``
    so albumentations remains an optional install:

        import albumentations as A
        aug = AlbumentationsWrap(A.Compose([
            A.Blur(p=0.1), A.MedianBlur(p=0.1),
            A.CoarseDropout(max_holes=1, max_height=0.4, max_width=0.4,
                            min_holes=1, min_height=0.2, min_width=0.2, p=1.0),
        ]))
    """

    def __init__(self, compose) -> None:
        self.compose = compose

    def __call__(self, s: Sample) -> Sample:
        out = self.compose(image=s.image)
        s.image = out["image"]
        return s


def build_rtmpose_albumentations_stage1(stage: int = 1):
    """Convenience: returns the Albumentations.Compose mmpose's RTMPose configs use.

    Stage 1 (epochs 0..stage_switch): CoarseDropout p=1.0
    Stage 2 (after switch_epoch):     CoarseDropout p=0.5

    Returns ``None`` if ``albumentations`` is not installed (callers should
    skip the step in that case). Compatible with both albumentations <2 (old
    arg names ``max_holes``/``min_holes``/``max_height``...) and >=2
    (``num_holes_range``/``hole_height_range``/``hole_width_range``).
    """
    try:
        import albumentations as A
        import inspect
    except ImportError:
        return None
    coarse_p = 1.0 if stage == 1 else 0.5
    sig_params = set(inspect.signature(A.CoarseDropout.__init__).parameters)
    if "num_holes_range" in sig_params:
        coarse = A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.2, 0.4),
            hole_width_range=(0.2, 0.4),
            p=coarse_p,
        )
    else:
        coarse = A.CoarseDropout(
            max_holes=1, max_height=0.4, max_width=0.4,
            min_holes=1, min_height=0.2, min_width=0.2, p=coarse_p,
        )
    return A.Compose([A.Blur(p=0.1), A.MedianBlur(p=0.1), coarse])


class YOLOXHSVRandomAug:
    """Random HSV jitter (mmdet-style). Used by RTMPose configs."""

    def __init__(self, hue_delta: int = 5, saturation_delta: int = 30, value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, s: Sample) -> Sample:
        img = s.image
        hsv_gains = np.random.uniform(-1, 1, 3) * [self.hue_delta, self.saturation_delta, self.value_delta]
        hsv_gains = hsv_gains.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        s.image = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return s


# ---- normalization (final step before model input) -------------------------

class NormalizeAndToTensor:
    """Convert ``input_image`` (HxWx3) to a CHW float32 tensor with ImageNet stats."""

    def __init__(
        self,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
        bgr_to_rgb: bool = True,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, s: Sample) -> Sample:
        img = s.input_image
        if self.bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = (img.astype(np.float32) - self.mean) / self.std
        s.input_image = np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW
        return s
