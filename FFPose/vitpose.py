"""ViTPose: ViT backbone -> optional bilinear neck -> HeatmapHead.

Variants supported here are the COCO 256x192 ones from mmpose configs at
configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-*_*.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .heatmap_codec import UDPHeatmapDecoder
from .heatmap_head import HeatmapHead
from .inference import PoseResult, _strip_state_dict, safe_torch_load
from .preprocess import (
    bbox_xyxy_to_center_scale,
    keypoints_from_input_to_image,
    normalize_to_tensor,
    topdown_affine,
)
from .skeletons import COCO_17
from .tta import flip_heatmaps
from .vit import VisionTransformer


# ---- model ------------------------------------------------------------------

@dataclass(frozen=True)
class ViTPoseConfig:
    arch: str | dict        # e.g. "small" or full dict
    out_channels: int       # K keypoints
    input_size: Tuple[int, int]      # (w, h) in mmpose preprocess convention
    img_size_hw: Tuple[int, int]     # (h, w) used for VisionTransformer.img_size
    patch_size: int = 16
    patch_padding: int = 2
    qkv_bias: bool = True
    drop_path_rate: float = 0.1
    with_cls_token: bool = False
    # Head
    deconv_out_channels: Tuple[int, ...] = (256, 256)
    deconv_kernel_sizes: Tuple[int, ...] = (4, 4)
    final_kernel_size: int = 1
    final_padding: int = 0
    # Optional neck (used only by ViTPose-*-simple configs)
    use_simple_neck: bool = False
    simple_neck_scale: float = 4.0
    # Codec
    heatmap_size: Tuple[int, int] = (48, 64)  # (w, h)
    blur_kernel_size: int = 11
    # Image stats
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    bgr_to_rgb: bool = True


class _SimpleUpsampleNeck(nn.Module):
    """ViTPose-simple's neck: ``F.interpolate(scale_factor=4) + ReLU``. No params."""

    def __init__(self, scale_factor: float = 4.0, apply_relu: bool = True) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        if self.apply_relu:
            x = F.relu(x, inplace=True)
        return x


class ViTPose(nn.Module):
    """Top-level ViTPose module: backbone -> optional neck -> head -> heatmap."""

    def __init__(self, cfg: ViTPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = VisionTransformer(
            arch=cfg.arch,
            img_size=cfg.img_size_hw,
            patch_size=cfg.patch_size,
            patch_padding=cfg.patch_padding,
            qkv_bias=cfg.qkv_bias,
            drop_path_rate=cfg.drop_path_rate,
            with_cls_token=cfg.with_cls_token,
            out_type="featmap",
        )
        if cfg.use_simple_neck:
            self.neck = _SimpleUpsampleNeck(scale_factor=cfg.simple_neck_scale)
        else:
            self.neck = nn.Identity()
        self.head = HeatmapHead(
            in_channels=self.backbone.embed_dims,
            out_channels=cfg.out_channels,
            deconv_out_channels=cfg.deconv_out_channels,
            deconv_kernel_sizes=cfg.deconv_kernel_sizes,
            final_kernel_size=cfg.final_kernel_size,
            final_padding=cfg.final_padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        feat = self.neck(feat)
        return self.head(feat)


# ---- variant configs (COCO 17, 256x192) -------------------------------------

VITPOSE_COCO_256x192: Dict[str, ViTPoseConfig] = {
    "small": ViTPoseConfig(
        arch=dict(embed_dims=384, num_layers=12, num_heads=12, feedforward_channels=384 * 4),
        out_channels=17,
        input_size=(192, 256),
        img_size_hw=(256, 192),
        drop_path_rate=0.1,
    ),
    "base": ViTPoseConfig(
        arch="base",
        out_channels=17,
        input_size=(192, 256),
        img_size_hw=(256, 192),
        drop_path_rate=0.3,
    ),
    "small-simple": ViTPoseConfig(
        arch=dict(embed_dims=384, num_layers=12, num_heads=12, feedforward_channels=384 * 4),
        out_channels=17,
        input_size=(192, 256),
        img_size_hw=(256, 192),
        drop_path_rate=0.1,
        deconv_out_channels=(),
        deconv_kernel_sizes=(),
        final_kernel_size=3,
        final_padding=1,
        use_simple_neck=True,
        simple_neck_scale=4.0,
    ),
    "base-simple": ViTPoseConfig(
        arch="base",
        out_channels=17,
        input_size=(192, 256),
        img_size_hw=(256, 192),
        drop_path_rate=0.3,
        deconv_out_channels=(),
        deconv_kernel_sizes=(),
        final_kernel_size=3,
        final_padding=1,
        use_simple_neck=True,
        simple_neck_scale=4.0,
    ),
}


# ---- inference --------------------------------------------------------------

class ViTPoseInferencer:
    """Top-down ViTPose inference with UDP-aware preprocessing/decoding."""

    def __init__(
        self,
        model: ViTPose,
        device: torch.device | str = "cuda",
        flip_test: bool = False,
        flip_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.decoder = UDPHeatmapDecoder(
            input_size=self.cfg.input_size,
            heatmap_size=self.cfg.heatmap_size,
            blur_kernel_size=self.cfg.blur_kernel_size,
        )
        self.flip_test = flip_test
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
    ) -> "ViTPoseInferencer":
        if variant not in VITPOSE_COCO_256x192:
            raise KeyError(
                f"unknown variant {variant!r}; available: {list(VITPOSE_COCO_256x192)}"
            )
        cfg = VITPOSE_COCO_256x192[variant]
        model = ViTPose(cfg)
        cls._load_checkpoint(model, checkpoint_path)
        return cls(model, device=device, flip_test=flip_test, flip_indices=flip_indices)

    @staticmethod
    def _load_checkpoint(model: ViTPose, path: str | Path) -> None:
        raw = safe_torch_load(path)
        sd = _strip_state_dict(raw)
        # ViTPose configs include `backbone.` and `head.` prefixes that match
        # our ViTPose module exactly (no neck params; neck is parameter-free).
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading checkpoint: {missing[:8]}{'...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[FFPose] ViTPose: {len(unexpected)} unexpected keys ignored, e.g.: {unexpected[:5]}")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> PoseResult:
        """One BGR image + one xyxy bbox -> PoseResult."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got {image.shape}")
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)

        center, scale = bbox_xyxy_to_center_scale(bbox, padding=1.25)
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.cfg.bgr_to_rgb else image
        warped, _, warp_mat = topdown_affine(
            img_in, center, scale,
            input_size=self.cfg.input_size,
            use_udp=True,
        )

        mean = np.array(self.cfg.mean, dtype=np.float32)
        std = np.array(self.cfg.std, dtype=np.float32)
        x = normalize_to_tensor(warped, mean=mean, std=std)
        x_t = torch.from_numpy(x).to(self.device)

        heatmaps = self.model(x_t)               # (1, K, Hh, Wh)

        if self.flip_test:
            x_flipped = torch.flip(x_t, dims=[-1])
            heatmaps_flipped = self.model(x_flipped)
            heatmaps_flipped = flip_heatmaps(
                heatmaps_flipped,
                flip_indices=self.flip_indices,
                shift_heatmap=False,  # ViTPose configs use shift_heatmap=False
            )
            heatmaps = (heatmaps + heatmaps_flipped) * 0.5

        hm = heatmaps[0].float().cpu().numpy()   # (K, Hh, Wh)

        kp_input, scores = self.decoder.decode(hm)
        kp_image = keypoints_from_input_to_image(kp_input, warp_mat)
        return PoseResult(keypoints=kp_image, scores=scores)
