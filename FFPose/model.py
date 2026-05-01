"""High-level RTMPose module: backbone -> head, plus pre-built variant configs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch.nn as nn
from torch import Tensor

from .backbone import CSPNeXt
from .head import RTMCCHead


@dataclass(frozen=True)
class RTMPoseConfig:
    """Inference-side configuration for an RTMPose variant."""
    # Backbone
    deepen_factor: float
    widen_factor: float
    in_channels: int        # backbone output channels (= 1024 * widen_factor)
    # Head
    out_channels: int       # = num keypoints
    input_size: Tuple[int, int]            # (w, h)
    in_featuremap_size: Tuple[int, int]    # (w, h) of last feature map
    simcc_split_ratio: float = 2.0
    final_layer_kernel_size: int = 7
    gau_hidden_dims: int = 256
    gau_s: int = 128
    gau_expansion_factor: int = 2
    gau_act_fn: str = "SiLU"
    gau_use_rel_bias: bool = False
    gau_pos_enc: bool = False
    # Image stats
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    bgr_to_rgb: bool = True


class RTMPose(nn.Module):
    """CSPNeXt + RTMCCHead, returns (pred_x, pred_y) SimCC logits.

    Submodule layout exactly mirrors ``TopdownPoseEstimator`` in mmpose so
    state dicts saved by mmpose load via ``load_state_dict(strict=False)``
    after stripping the runner wrapper.
    """

    def __init__(self, cfg: RTMPoseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CSPNeXt(
            arch="P5",
            deepen_factor=cfg.deepen_factor,
            widen_factor=cfg.widen_factor,
            out_indices=(4,),
            channel_attention=True,
            norm_cfg=dict(type="SyncBN"),
            act_cfg=dict(type="SiLU"),
        )
        self.head = RTMCCHead(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            input_size=cfg.input_size,
            in_featuremap_size=cfg.in_featuremap_size,
            simcc_split_ratio=cfg.simcc_split_ratio,
            final_layer_kernel_size=cfg.final_layer_kernel_size,
            gau_cfg=dict(
                hidden_dims=cfg.gau_hidden_dims,
                s=cfg.gau_s,
                expansion_factor=cfg.gau_expansion_factor,
                dropout_rate=0.0,
                drop_path=0.0,
                act_fn=cfg.gau_act_fn,
                use_rel_bias=cfg.gau_use_rel_bias,
                pos_enc=cfg.gau_pos_enc,
            ),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.backbone(x)
        return self.head(feats)


# ---- pre-built variant configs (COCO 17-keypoint, 256x192) ------------------

def _featuremap(input_size: Tuple[int, int], stride: int = 32) -> Tuple[int, int]:
    w, h = input_size
    return (w // stride, h // stride)


# Source: configs/body_2d_keypoint/rtmpose/coco/rtmpose-{t,s,m,l}_*coco-256x192.py
# Variant key naming: "<size>-<dataset>-<input>". "m" alone == body-coco-256x192 (legacy).
RTMPOSE_COCO_256x192: Dict[str, RTMPoseConfig] = {
    "t": RTMPoseConfig(
        deepen_factor=0.167, widen_factor=0.375,
        in_channels=int(1024 * 0.375),  # 384
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
    "s": RTMPoseConfig(
        deepen_factor=0.33, widen_factor=0.5,
        in_channels=int(1024 * 0.5),    # 512
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
    "m": RTMPoseConfig(
        deepen_factor=0.67, widen_factor=0.75,
        in_channels=int(1024 * 0.75),   # 768
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
    "l": RTMPoseConfig(
        deepen_factor=1.0, widen_factor=1.0,
        in_channels=1024,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
}


# RTMPose whole-body (133 keypoints: 17 body + 6 feet + 68 face + 42 hand).
# Source: configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-{m,l}_*256x192.py
RTMPOSE_WHOLEBODY_256x192: Dict[str, RTMPoseConfig] = {
    "m": RTMPoseConfig(
        deepen_factor=0.67, widen_factor=0.75,
        in_channels=int(1024 * 0.75),  # 768
        out_channels=133,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
    "l": RTMPoseConfig(
        deepen_factor=1.0, widen_factor=1.0,
        in_channels=1024,
        out_channels=133,
        input_size=(192, 256),
        in_featuremap_size=_featuremap((192, 256)),
    ),
}


# Catalog so callers can refer to variants by a single string.
ALL_RTMPOSE_VARIANTS: Dict[str, RTMPoseConfig] = {
    **{f"body-coco-{k}": v for k, v in RTMPOSE_COCO_256x192.items()},
    **{f"wholebody-coco-{k}": v for k, v in RTMPOSE_WHOLEBODY_256x192.items()},
}
