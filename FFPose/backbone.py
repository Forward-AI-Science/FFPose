"""CSPNeXt backbone for RTMPose. Submodule names match mmpose checkpoint keys."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor

from .blocks import CSPLayer, SPPBottleneck
from .layers import ConvModule, DepthwiseSeparableConvModule


class CSPNeXt(nn.Module):
    """CSPNeXt as used in RTMDet/RTMPose. Forward returns ``out_indices`` stage
    outputs as a tuple, matching mmpose's CSPNeXt.forward."""

    arch_settings = {
        # in, out, num_blocks, add_identity, use_spp
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (4,),
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU"),
    ) -> None:
        super().__init__()
        arch_setting = self.arch_settings[arch]
        assert set(out_indices).issubset(range(len(arch_setting) + 1))

        self.out_indices = tuple(out_indices)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        stem_c = int(arch_setting[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvModule(3, stem_c // 2, 3, padding=1, stride=2,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(stem_c // 2, stem_c // 2, 3, padding=1, stride=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(stem_c // 2, stem_c, 3, padding=1, stride=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.layers = ["stem"]

        for i, (in_c, out_c, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_c = int(in_c * widen_factor)
            out_c = int(out_c * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage: list[nn.Module] = [
                conv(in_c, out_c, 3, stride=2, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ]
            if use_spp:
                stage.append(SPPBottleneck(
                    out_c, out_c, kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                ))
            stage.append(CSPLayer(
                out_c, out_c,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
            ))
            name = f"stage{i + 1}"
            self.add_module(name, nn.Sequential(*stage))
            self.layers.append(name)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
