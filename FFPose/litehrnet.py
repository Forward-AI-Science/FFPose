"""LiteHRNet backbone (lightweight high-resolution network).

Direct port of mmpose/models/backbones/litehrnet.py. Submodule names match the
mmpose checkpoint exactly:

  stem.conv1 / stem.branch1 / stem.expand_conv / stem.depthwise_conv / stem.linear_conv
  transition{i}                                              # ModuleList
  stage{i}                                                   # Sequential of LiteHRModule
  head_layer.projects                                        # IterativeHead.projects (ModuleList)

LiteHRModule (LITE module_type) submodules:
  layers                       # Sequential of ConditionalChannelWeighting
  fuse_layers                  # ModuleList of ModuleList (None / Sequential)

ConditionalChannelWeighting submodules:
  cross_resolution_weighting.conv1 / conv2
  depthwise_convs              # ModuleList of ConvModule
  spatial_weighting            # ModuleList of SpatialWeighting (.conv1 / .conv2)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import ConvModule, DepthwiseSeparableConvModule


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Cross-group information flow used by ShuffleUnit / CCW."""
    B, C, H, W = x.size()
    assert C % groups == 0
    cpg = C // groups
    return x.view(B, groups, cpg, H, W).transpose(1, 2).contiguous().view(B, C, H, W)


# ---- channel-attention modules ---------------------------------------------

class _SpatialWeighting(nn.Module):
    """SE-style channel attention: GAP -> Conv -> ReLU -> Conv -> Sigmoid -> scale."""

    def __init__(
        self,
        channels: int,
        ratio: int = 16,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        hidden = int(channels / ratio)
        self.conv1 = ConvModule(channels, hidden, 1, stride=1,
                                 norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"))
        self.conv2 = ConvModule(hidden, channels, 1, stride=1,
                                 norm_cfg=norm_cfg, act_cfg=dict(type="Sigmoid"))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(self.global_avgpool(x)))
        return x * out


class _CrossResolutionWeighting(nn.Module):
    """Concat per-resolution descriptors (GAP+upsample) -> Conv-Conv (Sigmoid)
    -> split -> per-resolution multiplicative gate (interpolate to each res)."""

    def __init__(
        self,
        channels: Sequence[int],
        ratio: int = 16,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.channels = list(channels)
        total = sum(self.channels)
        hidden = int(total / ratio)
        self.conv1 = ConvModule(total, hidden, 1, stride=1,
                                 norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"))
        self.conv2 = ConvModule(hidden, total, 1, stride=1,
                                 norm_cfg=norm_cfg, act_cfg=dict(type="Sigmoid"))

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        mini_size = x[-1].shape[-2:]
        pooled = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = self.conv2(self.conv1(torch.cat(pooled, dim=1)))
        chunks = torch.split(out, self.channels, dim=1)
        return [s * F.interpolate(a, size=s.shape[-2:], mode="nearest")
                for s, a in zip(x, chunks)]


class _ConditionalChannelWeighting(nn.Module):
    """Half-channel split + cross-resolution weighting + depthwise + spatial weighting."""

    def __init__(
        self,
        in_channels: Sequence[int],
        stride: int,
        reduce_ratio: int,
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = stride
        branch_channels = [c // 2 for c in in_channels]

        self.cross_resolution_weighting = _CrossResolutionWeighting(
            branch_channels, ratio=reduce_ratio, norm_cfg=norm_cfg,
        )
        self.depthwise_convs = nn.ModuleList([
            ConvModule(c, c, 3, stride=stride, padding=1, groups=c,
                        norm_cfg=norm_cfg, act_cfg=None)
            for c in branch_channels
        ])
        self.spatial_weighting = nn.ModuleList([
            _SpatialWeighting(c, ratio=4) for c in branch_channels
        ])

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        x_pairs = [s.chunk(2, dim=1) for s in x]
        x1 = [p[0] for p in x_pairs]
        x2 = [p[1] for p in x_pairs]
        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]
        return [_channel_shuffle(torch.cat([s1, s2], dim=1), 2)
                for s1, s2 in zip(x1, x2)]


class _Stem(nn.Module):
    """LiteHRNet stem: stride-2 conv, then split-into-2-branches downsample."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        out_channels: int,
        expand_ratio: int = 1,
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvModule(
            in_channels, stem_channels, 3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"),
        )

        mid = int(round(stem_channels * expand_ratio))
        branch_c = stem_channels // 2
        if stem_channels == out_channels:
            inc = out_channels - branch_c
        else:
            inc = out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(branch_c, branch_c, 3, stride=2, padding=1,
                        groups=branch_c, norm_cfg=norm_cfg, act_cfg=None),
            ConvModule(branch_c, inc, 1, stride=1, padding=0,
                        norm_cfg=norm_cfg, act_cfg=dict(type="ReLU")),
        )
        self.expand_conv = ConvModule(branch_c, mid, 1, stride=1, padding=0,
                                       norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"))
        self.depthwise_conv = ConvModule(mid, mid, 3, stride=2, padding=1,
                                          groups=mid, norm_cfg=norm_cfg, act_cfg=None)
        self.linear_conv = ConvModule(
            mid, branch_c if stem_channels == out_channels else stem_channels,
            1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.linear_conv(self.depthwise_conv(self.expand_conv(x2)))
        out = torch.cat([self.branch1(x1), x2], dim=1)
        return _channel_shuffle(out, 2)


# ---- LiteHRModule (LITE only — naive variant uses ShuffleUnit, omitted) ----

class _LiteHRModule(nn.Module):
    """Multi-resolution module for LiteHRNet (``LITE`` module type).

    The naive variant (per-branch ShuffleUnit) is not used by any of the
    Lite-HRNet COCO configs (all use module_type='LITE'); we don't port it.
    """

    def __init__(
        self,
        num_branches: int,
        num_blocks: int,
        in_channels: List[int],
        reduce_ratio: int,
        module_type: str = "LITE",
        multiscale_output: bool = True,
        with_fuse: bool = True,
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super().__init__()
        if module_type.upper() != "LITE":
            raise NotImplementedError("only LITE module_type ported in FFPose")
        if num_branches != len(in_channels):
            raise ValueError("num_branches != len(in_channels)")

        self.num_branches = num_branches
        self.in_channels = list(in_channels)
        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg

        self.layers = nn.Sequential(*[
            _ConditionalChannelWeighting(in_channels, stride=1,
                                          reduce_ratio=reduce_ratio,
                                          norm_cfg=norm_cfg)
            for _ in range(num_blocks)
        ])
        if with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _make_fuse_layers(self) -> nn.ModuleList:
        if self.num_branches == 1:
            return None
        in_c = self.in_channels
        num_out = self.num_branches if self.multiscale_output else 1
        fuse_layers = []
        for i in range(num_out):
            inner = []
            for j in range(self.num_branches):
                if j > i:
                    # Upsample path: 1x1 conv + bn + nearest upsample.
                    from .layers import _build_norm
                    inner.append(nn.Sequential(
                        nn.Conv2d(in_c[j], in_c[i], 1, stride=1, padding=0, bias=False),
                        _build_norm(in_c[i], self.norm_cfg)[1],
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    inner.append(None)
                else:
                    # Downsample path: depthwise 3x3 stride 2 + bn + 1x1 conv + bn (+ ReLU on non-last)
                    downs = []
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_c = in_c[i] if is_last else in_c[j]
                        from .layers import _build_norm
                        seq = [
                            nn.Conv2d(in_c[j], in_c[j], 3, stride=2, padding=1,
                                       groups=in_c[j], bias=False),
                            _build_norm(in_c[j], self.norm_cfg)[1],
                            nn.Conv2d(in_c[j], out_c, 1, stride=1, padding=0, bias=False),
                            _build_norm(out_c, self.norm_cfg)[1],
                        ]
                        if not is_last:
                            seq.append(nn.ReLU(inplace=True))
                        downs.append(nn.Sequential(*seq))
                    inner.append(nn.Sequential(*downs))
            fuse_layers.append(nn.ModuleList(inner))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        if self.num_branches == 1:
            # single-branch case: layers is Sequential of CCW which expects a list
            return [self.layers([x[0]])[0]]
        out = self.layers(list(x))   # Sequential of CCW: list -> list
        if not self.with_fuse:
            if not self.multiscale_output:
                out = [out[0]]
            return out

        # Bit-exact port of mmpose's fuse loop. mmpose uses ``y += ...``
        # which is an in-place op; for ``i == 0`` y aliases out[0] and the
        # ``j == 0`` branch ``y += out[j]`` mutates out[0] mid-loop. Keeping
        # this behavior matters because subsequent iterations (i = 1, 2, ...)
        # read the now-mutated out[0]. Reproduced here with explicit
        # in-place adds so numerics match the upstream model exactly.
        out_fuse = []
        for i in range(len(self.fuse_layers)):
            y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
            for j in range(self.num_branches):
                if i == j:
                    y = y + out[j]                                  # not in-place; we'll mutate out[j] below
                else:
                    y = y + self.fuse_layers[i][j](out[j])
            # When i == 0, mmpose's in-place semantics mean out[0] should
            # equal y at this point. Reflect that so the i=1, i=2 iterations
            # of this same module read the same downstream input mmpose
            # reads.
            if i == 0:
                out[0] = y
            out_fuse.append(self.relu(y))
        if not self.multiscale_output:
            out_fuse = [out_fuse[0]]
        return out_fuse


# ---- iterative head --------------------------------------------------------

class _IterativeHead(nn.Module):
    """Top-down feature fusion across resolutions using DW-separable projections."""

    def __init__(self, in_channels: List[int], norm_cfg: dict = dict(type="BN")) -> None:
        super().__init__()
        self.in_channels = list(in_channels[::-1])  # high-res first reversed
        n = len(in_channels)
        projects = []
        for i in range(n):
            in_c = self.in_channels[i]
            out_c = self.in_channels[i + 1] if i != n - 1 else in_c
            projects.append(DepthwiseSeparableConvModule(
                in_channels=in_c, out_channels=out_c,
                kernel_size=3, stride=1, padding=1,
                norm_cfg=norm_cfg, act_cfg=dict(type="ReLU"),
                dw_act_cfg=None, pw_act_cfg=dict(type="ReLU"),
            ))
        self.projects = nn.ModuleList(projects)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        x = list(x[::-1])   # process from lowest resolution upward
        y = []
        last = None
        for i, s in enumerate(x):
            if last is not None:
                last = F.interpolate(last, size=s.shape[-2:],
                                       mode="bilinear", align_corners=True)
                s = s + last
            s = self.projects[i](s)
            y.append(s)
            last = s
        return y[::-1]


# ---- LiteHRNet top-level ---------------------------------------------------

class LiteHRNet(nn.Module):
    """LiteHRNet backbone."""

    def __init__(
        self,
        extra: dict,
        in_channels: int = 3,
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super().__init__()
        self.extra = extra
        self.norm_cfg = norm_cfg

        self.stem = _Stem(
            in_channels,
            stem_channels=extra["stem"]["stem_channels"],
            out_channels=extra["stem"]["out_channels"],
            expand_ratio=extra["stem"]["expand_ratio"],
            norm_cfg=norm_cfg,
        )

        self.num_stages = extra["num_stages"]
        self.stages_spec = extra["stages_spec"]

        prev = [self.stem.out_channels]
        for i in range(self.num_stages):
            cur = list(self.stages_spec["num_channels"][i])
            self.add_module(f"transition{i}", self._make_transition(prev, cur))
            stage, prev = self._make_stage(self.stages_spec, i, cur,
                                            multiscale_output=True)
            self.add_module(f"stage{i}", stage)

        self.with_head = extra.get("with_head", True)
        if self.with_head:
            self.head_layer = _IterativeHead(in_channels=prev, norm_cfg=norm_cfg)

    def _make_transition(self, pre: List[int], cur: List[int]) -> nn.ModuleList:
        n_pre, n_cur = len(pre), len(cur)
        from .layers import _build_norm
        layers: list[Optional[nn.Module]] = []
        for i in range(n_cur):
            if i < n_pre:
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], pre[i], 3, stride=1, padding=1,
                                   groups=pre[i], bias=False),
                        _build_norm(pre[i], self.norm_cfg)[1],
                        nn.Conv2d(pre[i], cur[i], 1, stride=1, padding=0, bias=False),
                        _build_norm(cur[i], self.norm_cfg)[1],
                        nn.ReLU(),
                    ))
                else:
                    layers.append(None)
            else:
                downs: list[nn.Module] = []
                for j in range(i + 1 - n_pre):
                    in_c = pre[-1]
                    out_c = cur[i] if j == i - n_pre else in_c
                    downs.append(nn.Sequential(
                        nn.Conv2d(in_c, in_c, 3, stride=2, padding=1,
                                   groups=in_c, bias=False),
                        _build_norm(in_c, self.norm_cfg)[1],
                        nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False),
                        _build_norm(out_c, self.norm_cfg)[1],
                        nn.ReLU(),
                    ))
                layers.append(nn.Sequential(*downs))
        return nn.ModuleList(layers)

    def _make_stage(
        self, stages_spec: dict, stage_index: int,
        in_channels: List[int], multiscale_output: bool = True,
    ):
        num_modules = stages_spec["num_modules"][stage_index]
        num_branches = stages_spec["num_branches"][stage_index]
        num_blocks = stages_spec["num_blocks"][stage_index]
        reduce_ratio = stages_spec["reduce_ratios"][stage_index]
        with_fuse = stages_spec["with_fuse"][stage_index]
        module_type = stages_spec["module_type"][stage_index]

        modules = []
        for i in range(num_modules):
            reset_ms = True
            if not multiscale_output and i == num_modules - 1:
                reset_ms = False
            mod = _LiteHRModule(
                num_branches=num_branches,
                num_blocks=num_blocks,
                in_channels=in_channels,
                reduce_ratio=reduce_ratio,
                module_type=module_type,
                multiscale_output=reset_ms,
                with_fuse=with_fuse,
                norm_cfg=self.norm_cfg,
            )
            modules.append(mod)
            in_channels = mod.in_channels
        return nn.Sequential(*modules), list(in_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.stem(x)
        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition: nn.ModuleList = getattr(self, f"transition{i}")
            for j in range(self.stages_spec["num_branches"][i]):
                t = transition[j]
                if t is not None:
                    if j >= len(y_list):
                        x_list.append(t(y_list[-1]))
                    else:
                        x_list.append(t(y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f"stage{i}")(x_list)
        if self.with_head:
            y_list = self.head_layer(y_list)
        return (y_list[0],)


# ---- variant configs --------------------------------------------------------

def _litehrnet_extra(num_modules_per_stage: Sequence[int]) -> dict:
    return dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=tuple(num_modules_per_stage),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=("LITE", "LITE", "LITE"),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (40, 80),
                (40, 80, 160),
                (40, 80, 160, 320),
            ),
        ),
        with_head=True,
    )


LITEHRNET_EXTRAS = {
    # LiteHRNet-18: (2, 4, 2) modules per stage.
    "18": _litehrnet_extra((2, 4, 2)),
    # LiteHRNet-30: (3, 8, 3) modules per stage.
    "30": _litehrnet_extra((3, 8, 3)),
}
