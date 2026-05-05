"""HRFormer pose estimator: HRFormer backbone + HeatmapHead + MSRA decode."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .heatmap_codec import MSRAHeatmapDecoder
from .heatmap_head import HeatmapHead
from .hrformer import HRFORMER_EXTRAS, HRFormer
from .inference import PoseResult, _strip_state_dict, safe_torch_load
from .preprocess import (
    bbox_xyxy_to_center_scale,
    keypoints_from_input_to_image,
    normalize_to_tensor,
    topdown_affine,
)
from .skeletons import COCO_17
from .tta import flip_heatmaps


@dataclass(frozen=True)
class HRFormerPoseConfig:
    variant: str                  # "small" | "base"
    out_channels: int             # K keypoints
    input_size: Tuple[int, int]   # (w, h)
    heatmap_size: Tuple[int, int] # (w, h)
    blur_kernel_size: int = 11
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    bgr_to_rgb: bool = True


_VARIANT_TO_WIDTH = {"small": 32, "base": 78}


HRFORMER_POSE_COCO_256x192: Dict[str, HRFormerPoseConfig] = {
    "small": HRFormerPoseConfig(variant="small", out_channels=17,
                                  input_size=(192, 256), heatmap_size=(48, 64)),
    "base": HRFormerPoseConfig(variant="base", out_channels=17,
                                 input_size=(192, 256), heatmap_size=(48, 64)),
}


class HRFormerPose(nn.Module):
    def __init__(self, cfg: HRFormerPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = HRFormer(extra=HRFORMER_EXTRAS[cfg.variant], in_channels=3)
        # Head: 1x1 conv on the highest-resolution branch (no deconvs).
        self.head = HeatmapHead(
            in_channels=_VARIANT_TO_WIDTH[cfg.variant],
            out_channels=cfg.out_channels,
            deconv_out_channels=(),
            deconv_kernel_sizes=(),
            final_kernel_size=1,
            final_padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class HRFormerPoseInferencer:
    def __init__(
        self,
        model: HRFormerPose,
        device: torch.device | str = "cuda",
        flip_test: bool = False,
        flip_indices: Optional[Sequence[int]] = None,
        shift_heatmap: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.decoder = MSRAHeatmapDecoder(
            input_size=self.cfg.input_size,
            heatmap_size=self.cfg.heatmap_size,
            unbiased=False,
            blur_kernel_size=self.cfg.blur_kernel_size,
        )
        self.flip_test = flip_test
        self.shift_heatmap = shift_heatmap
        if flip_test:
            if flip_indices is None:
                flip_indices = COCO_17.flip_indices
            self.flip_indices = list(flip_indices)
        else:
            self.flip_indices = None

    @classmethod
    def from_pretrained(
        cls, variant: str, checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
        flip_test: bool = False,
        flip_indices: Optional[Sequence[int]] = None,
        shift_heatmap: bool = True,
    ) -> "HRFormerPoseInferencer":
        if variant not in HRFORMER_POSE_COCO_256x192:
            raise KeyError(f"unknown variant {variant!r}")
        cfg = HRFORMER_POSE_COCO_256x192[variant]
        model = HRFormerPose(cfg)
        cls._load_checkpoint(model, checkpoint_path)
        return cls(model, device=device, flip_test=flip_test,
                    flip_indices=flip_indices, shift_heatmap=shift_heatmap)

    @staticmethod
    def _load_checkpoint(model: HRFormerPose, path: str | Path) -> None:
        raw = safe_torch_load(path)
        sd = _strip_state_dict(raw)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading checkpoint: {missing[:8]}")
        if unexpected:
            print(f"[FFPose] HRFormer: {len(unexpected)} unexpected keys ignored, e.g.: {unexpected[:5]}")

    @torch.no_grad()
    def predict(self, image: np.ndarray, bbox_xyxy: np.ndarray) -> PoseResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got {image.shape}")
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
        center, scale = bbox_xyxy_to_center_scale(bbox, padding=1.25)
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.cfg.bgr_to_rgb else image
        warped, _, warp_mat = topdown_affine(
            img_in, center, scale, input_size=self.cfg.input_size, use_udp=False,
        )
        x_t = torch.from_numpy(normalize_to_tensor(
            warped, mean=np.array(self.cfg.mean, dtype=np.float32),
            std=np.array(self.cfg.std, dtype=np.float32),
        )).to(self.device)

        heatmaps = self.model(x_t)
        if self.flip_test:
            x_flipped = torch.flip(x_t, dims=[-1])
            heatmaps_flipped = self.model(x_flipped)
            heatmaps_flipped = flip_heatmaps(
                heatmaps_flipped, flip_indices=self.flip_indices,
                shift_heatmap=self.shift_heatmap,
            )
            heatmaps = (heatmaps + heatmaps_flipped) * 0.5

        hm = heatmaps[0].float().cpu().numpy()
        kp_input, scores = self.decoder.decode(hm)
        kp_image = keypoints_from_input_to_image(kp_input, warp_mat)
        return PoseResult(keypoints=kp_image, scores=scores)
