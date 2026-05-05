"""CSPNeXt building blocks: SPP, CSP layer, channel attention.

Direct port from mmpose/models/{utils,backbones}/{csp_layer,csp_darknet}.py with
mm dependencies replaced. Submodule names preserved for checkpoint compat.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .layers import ConvModule, DepthwiseSeparableConvModule


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention used in CSPNeXt."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class DarknetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels, hidden, 1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = conv(
            hidden, out_channels, 3, stride=1, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU"),
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels, hidden, 3, stride=1, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden, out_channels, kernel_size, stride=1, padding=kernel_size // 2,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
    ) -> None:
        super().__init__()
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        common = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.main_conv = ConvModule(in_channels, mid, 1, **common)
        self.short_conv = ConvModule(in_channels, mid, 1, **common)
        self.final_conv = ConvModule(2 * mid, out_channels, 1, **common)
        self.blocks = nn.Sequential(*[
            block(mid, mid, 1.0, add_identity, use_depthwise, **common)
            for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        x_final = torch.cat((x_main, x_short), dim=1)
        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="Swish"),
    ) -> None:
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), out_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [p(x) for p in self.poolings], dim=1)
        return self.conv2(x)
