"""Pure-PyTorch replacements for mmcv.cnn building blocks.

Parameter naming matches mmcv exactly so existing mmpose checkpoints load
without remapping:
    ConvModule  -> .conv (Conv2d), .bn (BN/SyncBN), .activate (act)
    DepthwiseSeparableConvModule -> .depthwise_conv, .pointwise_conv
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


_NORM_NAMES = {
    "BN": "bn",
    "BN1d": "bn",
    "BN2d": "bn",
    "BN3d": "bn",
    "SyncBN": "bn",
    "GN": "gn",
    "LN": "ln",
    "IN": "in",
}


def _build_norm(num_features: int, cfg: dict) -> tuple[str, nn.Module]:
    cfg = dict(cfg)
    norm_type = cfg.pop("type")
    cfg.pop("requires_grad", None)
    name = _NORM_NAMES.get(norm_type, norm_type.lower())
    if norm_type in ("BN", "BN2d", "SyncBN"):
        return name, nn.BatchNorm2d(num_features, **cfg)
    if norm_type == "BN1d":
        return name, nn.BatchNorm1d(num_features, **cfg)
    if norm_type == "BN3d":
        return name, nn.BatchNorm3d(num_features, **cfg)
    if norm_type == "GN":
        num_groups = cfg.pop("num_groups")
        return name, nn.GroupNorm(num_groups, num_features, **cfg)
    if norm_type == "LN":
        return name, nn.LayerNorm(num_features, **cfg)
    if norm_type in ("IN", "IN2d"):
        return name, nn.InstanceNorm2d(num_features, **cfg)
    raise NotImplementedError(f"norm type {norm_type!r} not supported")


def _build_act(cfg: dict, *, inplace: bool) -> nn.Module:
    cfg = dict(cfg)
    act_type = cfg.pop("type")
    if act_type in ("SiLU", "Swish"):
        return nn.SiLU(inplace=inplace)
    if act_type == "ReLU":
        return nn.ReLU(inplace=inplace)
    if act_type == "ReLU6":
        return nn.ReLU6(inplace=inplace)
    if act_type == "LeakyReLU":
        return nn.LeakyReLU(inplace=inplace, **cfg)
    if act_type == "GELU":
        return nn.GELU()
    if act_type in ("HSigmoid", "HardSigmoid"):
        return nn.Hardsigmoid(inplace=inplace)
    if act_type in ("HSwish", "HardSwish"):
        return nn.Hardswish(inplace=inplace)
    if act_type == "Sigmoid":
        return nn.Sigmoid()
    if act_type == "Tanh":
        return nn.Tanh()
    raise NotImplementedError(f"activation type {act_type!r} not supported")


class ConvModule(nn.Module):
    """Conv2d -> Norm -> Activation, mmcv-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = dict(type="ReLU"),
        inplace: bool = True,
    ) -> None:
        super().__init__()
        if conv_cfg is not None and conv_cfg.get("type", "Conv2d") != "Conv2d":
            raise NotImplementedError(f"conv type {conv_cfg.get('type')} not supported")

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.with_norm:
            # Norm is created on out_channels; mmcv puts it before activation.
            norm_name, norm_layer = _build_norm(out_channels, norm_cfg)
            self.norm_name = norm_name
            self.add_module(norm_name, norm_layer)
        else:
            self.norm_name = None

        if self.with_activation:
            self.activate = _build_act(act_cfg, inplace=inplace)

    @property
    def norm(self) -> Optional[nn.Module]:
        return getattr(self, self.norm_name) if self.norm_name else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise + pointwise Conv2d, mmcv-compatible naming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm_cfg: Optional[dict] = dict(type="BN"),
        act_cfg: Optional[dict] = dict(type="ReLU"),
        dw_norm_cfg: str | dict = "default",
        dw_act_cfg: str | dict = "default",
        pw_norm_cfg: str | dict = "default",
        pw_act_cfg: str | dict = "default",
        **kwargs,
    ) -> None:
        super().__init__()
        kwargs.pop("conv_cfg", None)
        dw_norm_cfg = norm_cfg if dw_norm_cfg == "default" else dw_norm_cfg
        dw_act_cfg = act_cfg if dw_act_cfg == "default" else dw_act_cfg
        pw_norm_cfg = norm_cfg if pw_norm_cfg == "default" else pw_norm_cfg
        pw_act_cfg = act_cfg if pw_act_cfg == "default" else pw_act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs,
        )
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class DropPath(nn.Module):
    """Stochastic depth (per-sample). Identity at inference."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask
