"""HeatmapHead used by ViTPose.

Layout (matches mmpose's HeatmapHead with default mmcv naming):
    head.deconv_layers : Sequential of [ConvTranspose2d, BN, ReLU, ConvTranspose2d, BN, ReLU, ...]
    head.final_layer   : Conv2d (1x1 by default; ViTPose-simple uses 3x3)
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor


class HeatmapHead(nn.Module):
    """Optional stack of stride-2 deconv upsamplers, then a 1x1/3x3 conv to K
    keypoint heatmaps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: Sequence[int] = (256, 256),
        deconv_kernel_sizes: Sequence[int] = (4, 4),
        final_kernel_size: int = 1,
        final_padding: int = 0,
    ) -> None:
        super().__init__()
        if len(deconv_out_channels) != len(deconv_kernel_sizes):
            raise ValueError(
                "deconv_out_channels and deconv_kernel_sizes must have same length"
            )

        layers: list[nn.Module] = []
        cur = in_channels
        for c, k in zip(deconv_out_channels, deconv_kernel_sizes):
            # mmpose HeatmapHead uses padding from kernel: 4->1, 3->1, 2->0.
            if k == 4:
                pad, output_padding = 1, 0
            elif k == 3:
                pad, output_padding = 1, 1
            elif k == 2:
                pad, output_padding = 0, 0
            else:
                raise ValueError(f"unsupported deconv kernel {k}")
            layers.append(nn.ConvTranspose2d(
                cur, c, kernel_size=k, stride=2, padding=pad,
                output_padding=output_padding, bias=False,
            ))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            cur = c
        # When deconv_out_channels is empty, deconv_layers is just an identity.
        self.deconv_layers = nn.Sequential(*layers) if layers else nn.Identity()

        self.final_layer = nn.Conv2d(
            cur, out_channels,
            kernel_size=final_kernel_size,
            stride=1,
            padding=final_padding,
        )

    def forward(self, feats) -> Tensor:
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        x = self.deconv_layers(feats)
        return self.final_layer(x)
