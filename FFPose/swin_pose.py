"""Swin Transformer pose estimator: Swin backbone -> HeatmapHead -> MSRA decode.

Loads mmpose 0.x checkpoints (the Swin pose weights pre-date the 1.x rename
from ``keypoint_head`` -> ``head``) via :func:`_strip_state_dict` which handles
both prefixes.
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
from .inference import PoseResult, _strip_state_dict, safe_torch_load
from .preprocess import (
    bbox_xyxy_to_center_scale,
    keypoints_from_input_to_image,
    normalize_to_tensor,
    topdown_affine,
)
from .skeletons import COCO_17
from .swin import SWIN_VARIANTS, SwinTransformer
from .tta import flip_heatmaps


@dataclass(frozen=True)
class SwinPoseConfig:
    variant: str                  # "t", "b", or "l"
    out_channels: int             # K keypoints
    input_size: Tuple[int, int]   # (w, h)
    heatmap_size: Tuple[int, int] # (w, h)
    # Swin pose checkpoints from mmpose 0.x use 3 deconv layers @ 256 ch each.
    deconv_out_channels: Tuple[int, ...] = (256, 256, 256)
    deconv_kernel_sizes: Tuple[int, ...] = (4, 4, 4)
    final_kernel_size: int = 1
    final_padding: int = 0
    blur_kernel_size: int = 11
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    bgr_to_rgb: bool = True


SWIN_POSE_COCO_256x192: Dict[str, SwinPoseConfig] = {
    "t": SwinPoseConfig(variant="t", out_channels=17,
                         input_size=(192, 256), heatmap_size=(48, 64)),
    "b": SwinPoseConfig(variant="b", out_channels=17,
                         input_size=(192, 256), heatmap_size=(48, 64)),
    "l": SwinPoseConfig(variant="l", out_channels=17,
                         input_size=(192, 256), heatmap_size=(48, 64)),
}

# Swin's deepest stage outputs ``embed_dims * 2 ** 3`` channels.
def _backbone_out_channels(variant: str) -> int:
    return SWIN_VARIANTS[variant]["embed_dims"] * (2 ** 3)


class SwinPose(nn.Module):
    """SwinTransformer + HeatmapHead. Standard 32x downsample backbone followed
    by 3 deconvs (each 2x upsample) -> heatmap at stride 4.
    """

    def __init__(self, cfg: SwinPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SwinTransformer(
            **SWIN_VARIANTS[cfg.variant], out_indices=(3,),
        )
        self.head = HeatmapHead(
            in_channels=_backbone_out_channels(cfg.variant),
            out_channels=cfg.out_channels,
            deconv_out_channels=cfg.deconv_out_channels,
            deconv_kernel_sizes=cfg.deconv_kernel_sizes,
            final_kernel_size=cfg.final_kernel_size,
            final_padding=cfg.final_padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class SwinPoseInferencer:
    """Top-down inference for Swin pose checkpoints (heatmap, MSRA decode)."""

    def __init__(
        self,
        model: SwinPose,
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
    ) -> "SwinPoseInferencer":
        if variant not in SWIN_POSE_COCO_256x192:
            raise KeyError(f"unknown variant {variant!r}; available: {list(SWIN_POSE_COCO_256x192)}")
        cfg = SWIN_POSE_COCO_256x192[variant]
        model = SwinPose(cfg)
        cls._load_checkpoint(model, checkpoint_path)
        return cls(model, device=device, flip_test=flip_test,
                    flip_indices=flip_indices, shift_heatmap=shift_heatmap)

    @staticmethod
    def _load_checkpoint(model: SwinPose, path: str | Path) -> None:
        raw = safe_torch_load(path)
        sd = _strip_state_dict(raw)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading: {missing[:8]}{'...' if len(missing)>8 else ''}")
        # The mmpose Swin checkpoint includes ``norm0/1/2`` for stages we
        # don't read — they're harmless extras.
        unexpected = [k for k in unexpected if not k.startswith(("backbone.norm0", "backbone.norm1", "backbone.norm2"))]
        if unexpected:
            print(f"[FFPose] Swin: {len(unexpected)} unexpected keys ignored, e.g.: {unexpected[:5]}")

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
        x_t = torch.from_numpy(normalize_to_tensor(warped, mean=mean, std=std)).to(self.device)

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
