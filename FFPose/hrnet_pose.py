"""HRNet pose estimator: HRNet backbone -> HeatmapHead -> MSRAHeatmap decode.

For COCO 256x192, HRNet-W32 has the highest-resolution branch at stride 4
(64x48), so the head is just a single 1x1 Conv2d(width, K) — no deconvs.
"""
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
from .hrnet import HRNET_EXTRAS, HRNet
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
class HRNetPoseConfig:
    width: int                     # 32 or 48
    out_channels: int              # K keypoints
    input_size: Tuple[int, int]    # (w, h)
    heatmap_size: Tuple[int, int]  # (w, h)
    unbiased: bool = False
    blur_kernel_size: int = 11
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    bgr_to_rgb: bool = True


class HRNetPose(nn.Module):
    """HRNet backbone + HeatmapHead with no deconv (1x1 final conv)."""

    def __init__(self, cfg: HRNetPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = HRNet(extra=HRNET_EXTRAS[f"w{cfg.width}"], in_channels=3)
        self.head = HeatmapHead(
            in_channels=cfg.width,
            out_channels=cfg.out_channels,
            deconv_out_channels=(),
            deconv_kernel_sizes=(),
            final_kernel_size=1,
            final_padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        return self.head(feats)


HRNET_POSE_COCO_256x192: Dict[str, HRNetPoseConfig] = {
    "w32": HRNetPoseConfig(
        width=32, out_channels=17,
        input_size=(192, 256), heatmap_size=(48, 64),
    ),
    "w48": HRNetPoseConfig(
        width=48, out_channels=17,
        input_size=(192, 256), heatmap_size=(48, 64),
    ),
}


class HRNetPoseInferencer:
    """Top-down inference for HRNet (heatmap, standard warp matrix)."""

    def __init__(
        self,
        model: HRNetPose,
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
            unbiased=self.cfg.unbiased,
            blur_kernel_size=self.cfg.blur_kernel_size,
        )
        self.flip_test = flip_test
        self.shift_heatmap = shift_heatmap
        if flip_test:
            if flip_indices is None:
                flip_indices = COCO_17.flip_indices
            if len(flip_indices) != self.cfg.out_channels:
                raise ValueError(
                    f"flip_indices length {len(flip_indices)} != out_channels {self.cfg.out_channels}"
                )
            self.flip_indices = list(flip_indices)
        else:
            self.flip_indices = None

    @classmethod
    def from_pretrained(
        cls,
        variant: str,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
        flip_test: bool = False,
        flip_indices: Optional[Sequence[int]] = None,
        shift_heatmap: bool = True,
    ) -> "HRNetPoseInferencer":
        if variant not in HRNET_POSE_COCO_256x192:
            raise KeyError(f"unknown variant {variant!r}; available: {list(HRNET_POSE_COCO_256x192)}")
        model = HRNetPose(HRNET_POSE_COCO_256x192[variant])
        cls._load_checkpoint(model, checkpoint_path)
        return cls(model, device=device, flip_test=flip_test,
                   flip_indices=flip_indices, shift_heatmap=shift_heatmap)

    @staticmethod
    def _load_checkpoint(model: HRNetPose, path: str | Path) -> None:
        raw = safe_torch_load(path)
        sd = _strip_state_dict(raw)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading checkpoint: {missing[:8]}{'...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[FFPose] HRNet: {len(unexpected)} unexpected keys ignored, e.g.: {unexpected[:5]}")

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

        mean = np.array(self.cfg.mean, dtype=np.float32)
        std = np.array(self.cfg.std, dtype=np.float32)
        x = normalize_to_tensor(warped, mean=mean, std=std)
        x_t = torch.from_numpy(x).to(self.device)

        heatmaps = self.model(x_t)             # (1, K, Hh, Wh)

        if self.flip_test:
            x_flipped = torch.flip(x_t, dims=[-1])
            heatmaps_flipped = self.model(x_flipped)
            heatmaps_flipped = flip_heatmaps(
                heatmaps_flipped,
                flip_indices=self.flip_indices,
                shift_heatmap=self.shift_heatmap,
            )
            heatmaps = (heatmaps + heatmaps_flipped) * 0.5

        hm = heatmaps[0].float().cpu().numpy()

        kp_input, scores = self.decoder.decode(hm)
        kp_image = keypoints_from_input_to_image(kp_input, warp_mat)
        return PoseResult(keypoints=kp_image, scores=scores)
