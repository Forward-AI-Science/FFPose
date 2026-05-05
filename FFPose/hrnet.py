"""HRNet backbone for COCO pose estimation.

Direct port of mmpose/models/backbones/{hrnet,resnet}.py. The submodule
naming exactly matches mmpose's use of ``build_norm_layer(..., postfix=N)``:

    - ``norm{N}_name`` -> ``'bn{N}'`` (so the actual attribute is e.g. ``bn1``)
    - ``downsample`` is ``Sequential(Conv1x1, BN)`` — no ReLU
    - ``transition{n}[i]`` is either ``None`` (no shape change) or
      ``Sequential(Conv, BN, ReLU)`` for the existing-branch case, or
      ``Sequential(Sequential(Conv, BN, ReLU), ...)`` for the new-branch case
    - ``stage{n}.{m}.branches[i][j]`` is a ``BasicBlock`` or ``Bottleneck``
    - ``stage{n}.{m}.fuse_layers[i][j]`` is None if ``i == j`` (kept in
      ModuleList as a None entry, no params), else Conv+BN+Upsample (j > i)
      or stacked stride-2 Conv+BN(+ReLU) (j < i).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---- residual blocks --------------------------------------------------------

class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock. ``conv1 -> bn1 -> relu -> conv2 -> bn2 -> + identity -> relu``."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert out_channels % self.expansion == 0
        mid = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet bottleneck block with 1x1 -> 3x3 -> 1x1 convs and expansion=4."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert out_channels % self.expansion == 0
        mid = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


_BLOCKS = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


def _make_layer(block, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
    """Stack ``blocks`` of ``block`` with the first one optionally downsampling."""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    layers: list[nn.Module] = [block(in_channels, out_channels, stride=stride, downsample=downsample)]
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)


# ---- HR module --------------------------------------------------------------

class HRModule(nn.Module):
    """One multi-resolution module: per-branch residual stack + cross-branch fusion."""

    def __init__(
        self,
        num_branches: int,
        block: type,
        num_blocks: Sequence[int],
        in_channels: list[int],
        num_channels: Sequence[int],
        multiscale_output: bool = True,
    ) -> None:
        super().__init__()
        if len(num_blocks) != num_branches or len(num_channels) != num_branches or len(in_channels) != num_branches:
            raise ValueError("num_blocks, num_channels, in_channels must all match num_branches")

        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        # Track and update in_channels per branch (mutated by _make_one_branch).
        self.in_channels = list(in_channels)

        self.branches = nn.ModuleList([
            self._make_one_branch(i, block, num_blocks, num_channels)
            for i in range(num_branches)
        ])
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self, branch_index: int, block: type,
        num_blocks: Sequence[int], num_channels: Sequence[int],
        stride: int = 1,
    ) -> nn.Sequential:
        target = num_channels[branch_index] * block.expansion
        layers = _make_layer(
            block,
            self.in_channels[branch_index],
            target,
            num_blocks[branch_index],
            stride=stride,
        )
        self.in_channels[branch_index] = target
        return layers

    def _make_fuse_layers(self) -> Optional[nn.ModuleList]:
        if self.num_branches == 1:
            return None
        in_channels = self.in_channels
        num_out = self.num_branches if self.multiscale_output else 1

        fuse_layers: list[nn.ModuleList] = []
        for i in range(num_out):
            inner: list[Optional[nn.Module]] = []
            for j in range(self.num_branches):
                if j > i:
                    inner.append(nn.Sequential(
                        nn.Conv2d(in_channels[j], in_channels[i], 1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(in_channels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    ))
                elif j == i:
                    inner.append(None)
                else:
                    downs: list[nn.Module] = []
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_c = in_channels[i] if is_last else in_channels[j]
                        seq = [
                            nn.Conv2d(in_channels[j], out_c, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_c),
                        ]
                        if not is_last:
                            seq.append(nn.ReLU(inplace=True))
                        downs.append(nn.Sequential(*seq))
                    inner.append(nn.Sequential(*downs))
            fuse_layers.append(nn.ModuleList(inner))
        return nn.ModuleList(fuse_layers)

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        x = [self.branches[i](x[i]) for i in range(self.num_branches)]

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[i] if i < len(x) else 0
            for j in range(self.num_branches):
                if i == j:
                    continue
                fl = self.fuse_layers[i][j]
                if fl is None:
                    y = y + x[j]
                else:
                    y = y + fl(x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


# ---- HRNet -----------------------------------------------------------------

class HRNet(nn.Module):
    """Pure-PyTorch HRNet backbone.

    Args:
        extra: dict with keys ``stage1..stage4`` matching mmpose's HRNet
            config schema (``num_modules``, ``num_branches``, ``block``,
            ``num_blocks``, ``num_channels``).
        in_channels: input image channels (3 for RGB).
    """

    def __init__(self, extra: dict, in_channels: int = 3) -> None:
        super().__init__()
        self.extra = extra
        upsample_cfg = extra.get("upsample", {"mode": "nearest"})
        self._upsample_mode = upsample_cfg.get("mode", "nearest")

        # Stem: stride-4 downsample.
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # stage1
        s1 = extra["stage1"]
        block1 = _BLOCKS[s1["block"]]
        s1_out = s1["num_channels"][0] * block1.expansion
        self.layer1 = _make_layer(block1, 64, s1_out, s1["num_blocks"][0])
        self.stage1_cfg = s1

        # stage2
        s2 = extra["stage2"]
        block2 = _BLOCKS[s2["block"]]
        s2_channels = [c * block2.expansion for c in s2["num_channels"]]
        self.transition1 = self._make_transition_layer([s1_out], s2_channels)
        self.stage2, pre = self._make_stage(s2, s2_channels)
        self.stage2_cfg = s2

        # stage3
        s3 = extra["stage3"]
        block3 = _BLOCKS[s3["block"]]
        s3_channels = [c * block3.expansion for c in s3["num_channels"]]
        self.transition2 = self._make_transition_layer(pre, s3_channels)
        self.stage3, pre = self._make_stage(s3, s3_channels)
        self.stage3_cfg = s3

        # stage4
        s4 = extra["stage4"]
        block4 = _BLOCKS[s4["block"]]
        s4_channels = [c * block4.expansion for c in s4["num_channels"]]
        self.transition3 = self._make_transition_layer(pre, s4_channels)
        self.stage4, pre = self._make_stage(
            s4, s4_channels,
            multiscale_output=s4.get("multiscale_output", False),
        )
        self.stage4_cfg = s4

    def _make_transition_layer(
        self, pre_channels: list[int], cur_channels: list[int],
    ) -> nn.ModuleList:
        num_pre = len(pre_channels)
        num_cur = len(cur_channels)
        layers: list[Optional[nn.Module]] = []
        for i in range(num_cur):
            if i < num_pre:
                if cur_channels[i] != pre_channels[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre_channels[i], cur_channels[i], 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(cur_channels[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    layers.append(None)
            else:
                downs: list[nn.Module] = []
                for j in range(i + 1 - num_pre):
                    in_c = pre_channels[-1]
                    out_c = cur_channels[i] if j == (i - num_pre) else in_c
                    downs.append(nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                    ))
                layers.append(nn.Sequential(*downs))
        return nn.ModuleList(layers)

    def _make_stage(
        self, layer_cfg: dict, in_channels: list[int],
        multiscale_output: bool = True,
    ):
        num_modules = layer_cfg["num_modules"]
        num_branches = layer_cfg["num_branches"]
        num_blocks = layer_cfg["num_blocks"]
        num_channels = layer_cfg["num_channels"]
        block = _BLOCKS[layer_cfg["block"]]

        modules: list[HRModule] = []
        for i in range(num_modules):
            reset_ms = True
            if (not multiscale_output) and i == num_modules - 1:
                reset_ms = False
            mod = HRModule(num_branches, block, num_blocks,
                           in_channels=list(in_channels),
                           num_channels=num_channels,
                           multiscale_output=reset_ms)
            modules.append(mod)
            in_channels = mod.in_channels
        return nn.Sequential(*modules), list(in_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # transition1: at this point x is single-tensor; build first branch list.
        x_list: list[Tensor] = []
        for i in range(self.stage2_cfg["num_branches"]):
            t = self.transition1[i]
            x_list.append(t(x) if t is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            t = self.transition2[i]
            x_list.append(t(y_list[-1]) if t is not None else y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            t = self.transition3[i]
            x_list.append(t(y_list[-1]) if t is not None else y_list[i])
        y_list = self.stage4(x_list)

        return tuple(y_list)


# ---- variant configs --------------------------------------------------------

def _hrnet_extra(width: int) -> dict:
    """HRNet-W{width} extra config, as used by COCO-256x192 mmpose configs."""
    return dict(
        stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK",
                    num_blocks=(4,), num_channels=(64,)),
        stage2=dict(num_modules=1, num_branches=2, block="BASIC",
                    num_blocks=(4, 4), num_channels=(width, width * 2)),
        stage3=dict(num_modules=4, num_branches=3, block="BASIC",
                    num_blocks=(4, 4, 4),
                    num_channels=(width, width * 2, width * 4)),
        stage4=dict(num_modules=3, num_branches=4, block="BASIC",
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(width, width * 2, width * 4, width * 8)),
    )


HRNET_EXTRAS = {
    "w32": _hrnet_extra(32),
    "w48": _hrnet_extra(48),
}
