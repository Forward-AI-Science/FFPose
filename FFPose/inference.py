"""High-level inference API and mmpose checkpoint loader.

Usage:
    pose = RTMPoseInferencer.from_pretrained("rtmpose-m_coco_256x192", "/path/to/ckpt.pth")
    keypoints, scores = pose.predict(image_bgr, bbox_xyxy)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch

from .codec import SimCCDecoder
from .model import (
    ALL_RTMPOSE_VARIANTS,
    RTMPOSE_COCO_256x192,
    RTMPose,
    RTMPoseConfig,
)
from .preprocess import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    bbox_xyxy_to_center_scale,
    keypoints_from_input_to_image,
    normalize_to_tensor,
    topdown_affine,
)


import pickle as _pickle


class _MissingClass(dict):
    """Generic placeholder produced by :func:`_safe_torch_load` when the
    pickled checkpoint references classes from libraries we no longer depend
    on (``mmengine.MessageHub``, ``mmengine.config.ConfigDict``, etc.).
    The actual model weights live under ``state_dict`` and don't need any of
    these classes, so swallowing them keeps loading robust."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)


class _SafeUnpickler(_pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return _MissingClass


class _SafePickleModule:
    """Stand-in for ``pickle`` passed to ``torch.load(pickle_module=...)``.

    Implements the small subset of the ``pickle`` API torch.load actually
    consumes: the ``Unpickler`` class for new-format zip checkpoints, and a
    top-level ``load`` for the legacy format.
    """
    Unpickler = _SafeUnpickler
    Pickler = _pickle.Pickler

    @staticmethod
    def load(file, **kwargs):
        return _SafeUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        import io
        return _SafeUnpickler(io.BytesIO(data), **kwargs).load()


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a torch checkpoint saved by mmpose without requiring mmengine.

    Substitutes a placeholder class for any unresolved pickled references.
    """
    return torch.load(
        str(path),
        map_location=map_location,
        weights_only=False,
        pickle_module=_SafePickleModule,
    )


def _strip_state_dict(raw: dict, *, head_prefix: str = "head") -> dict:
    """Pull model weights out of an mmpose-style checkpoint.

    mmpose 1.x saves a dict with ``state_dict`` / ``meta`` / ``message_hub``.
    Submodule names start with ``backbone.`` and ``head.``. Older mmpose 0.x
    checkpoints (e.g., the Swin pose weights) use ``keypoint_head.`` instead.
    This helper renames ``keypoint_head.`` -> ``{head_prefix}.`` so a single
    model layout loads both eras.
    """
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw
    out = {}
    for k, v in sd.items():
        if (k.startswith("data_preprocessor.")
                or k.startswith("head.loss_module.")
                or k.startswith("head.decoder.")):
            continue
        if k.startswith("keypoint_head."):
            k = head_prefix + k[len("keypoint_head"):]
        out[k] = v
    return out


@dataclass
class PoseResult:
    keypoints: np.ndarray  # (N, K, 2) in original-image pixel coords
    scores: np.ndarray     # (N, K)


class RTMPoseInferencer:
    """One-call top-down pose inference around an :class:`RTMPose` model."""

    def __init__(
        self,
        model: RTMPose,
        device: torch.device | str = "cuda",
        flip_test: bool = False,
        flip_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.decoder = SimCCDecoder(
            input_size=self.cfg.input_size,
            simcc_split_ratio=self.cfg.simcc_split_ratio,
        )
        self.flip_test = flip_test
        self.flip_indices = list(flip_indices) if flip_indices is not None else None
        if flip_test and flip_indices is None:
            raise ValueError("flip_test=True requires flip_indices (per-keypoint pairs)")

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        variant: str,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "RTMPoseInferencer":
        # Accept both the legacy short keys ("m", "s", ...) and the catalog
        # keys ("body-coco-m", "wholebody-coco-m", ...).
        if variant in ALL_RTMPOSE_VARIANTS:
            cfg = ALL_RTMPOSE_VARIANTS[variant]
        elif variant in RTMPOSE_COCO_256x192:
            cfg = RTMPOSE_COCO_256x192[variant]
        else:
            raise KeyError(
                f"unknown variant {variant!r}; available: "
                f"{list(ALL_RTMPOSE_VARIANTS)}"
            )
        model = RTMPose(cfg)
        cls._load_checkpoint(model, checkpoint_path)
        return cls(model, device=device)

    @staticmethod
    def _load_checkpoint(model: RTMPose, path: str | Path) -> None:
        raw = safe_torch_load(path)
        sd = _strip_state_dict(raw)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading checkpoint: {missing[:8]}{'...' if len(missing)>8 else ''}")
        if unexpected:
            # Don't fail — extra keys (e.g. data_preprocessor leftovers) just get logged.
            print(f"[FFPose] {len(unexpected)} unexpected keys ignored, e.g.: {unexpected[:5]}")

    # ---- prediction ---------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> PoseResult:
        """Run pose estimation on ``image`` (BGR, HxWx3) for one bbox.

        Args:
            image: Source image, BGR uint8 (cv2 convention).
            bbox_xyxy: (4,) array (x1, y1, x2, y2) in image pixels.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got {image.shape}")
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)

        # Affine-warp the image into the model's input canvas.
        center, scale = bbox_xyxy_to_center_scale(bbox, padding=1.25)
        img_to_warp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.cfg.bgr_to_rgb else image
        warped, _, warp_mat = topdown_affine(
            img_to_warp, center, scale, input_size=self.cfg.input_size
        )

        # Normalize and run the model.
        mean = np.array(self.cfg.mean, dtype=np.float32)
        std = np.array(self.cfg.std, dtype=np.float32)
        x = normalize_to_tensor(warped, mean=mean, std=std)
        x_t = torch.from_numpy(x).to(self.device)

        pred_x, pred_y = self.model(x_t)

        if self.flip_test:
            x_flip = torch.flip(x_t, dims=[-1])
            px_f, py_f = self.model(x_flip)
            # Flip x bins: argmax over w should mirror, and re-order keypoints
            # by left/right pair indices.
            px_f = torch.flip(px_f, dims=[-1])
            idx = torch.tensor(self.flip_indices, device=self.device)
            px_f = px_f.index_select(1, idx)
            py_f = py_f.index_select(1, idx)
            pred_x = (pred_x + px_f) * 0.5
            pred_y = (pred_y + py_f) * 0.5

        # Decode in input-canvas pixel space, then unwarp to image coords.
        kp_input, scores = self.decoder.decode(
            pred_x.cpu().numpy(), pred_y.cpu().numpy()
        )
        kp_image = keypoints_from_input_to_image(kp_input, warp_mat)
        return PoseResult(keypoints=kp_image, scores=scores)
