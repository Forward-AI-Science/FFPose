"""HRFormer backbone (multi-resolution local-window transformer).

Direct port of mmpose's ``HRFormer``. Builds on our :class:`HRNet` (stem,
``layer1`` Bottleneck stage, transition layers) but replaces the BasicBlock
branches in stages 2-4 with :class:`HRFormerBlock` and uses a depthwise
fusion path between branches.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hrnet import Bottleneck, _make_layer
from .layers import DropPath, _build_norm


# ---- attention -------------------------------------------------------------

class _WindowMSA(nn.Module):
    """Window-MSA with optional relative position bias.

    Submodule names match mmpose's ``WindowMSA``: ``qkv``, ``proj``,
    ``relative_position_bias_table``, ``relative_position_index``.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        with_rpe: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5
        self.with_rpe = with_rpe

        if with_rpe:
            Wh, Ww = window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
            )
            seq1 = torch.arange(0, (2 * Ww - 1) * Wh, 2 * Ww - 1)
            seq2 = torch.arange(0, Ww)
            rel_index_coords = (seq1[:, None] + seq2[None, :]).reshape(1, -1)
            rel_index = rel_index_coords + rel_index_coords.T
            rel_index = rel_index.flip(1).contiguous()
            self.register_buffer("relative_position_index", rel_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.with_rpe:
            Wh, Ww = self.window_size
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(Wh * Ww, Wh * Ww, -1).permute(2, 0, 1).contiguous()
            attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class _LocalWindowSelfAttention(nn.Module):
    """Center-pads the feature map then runs window-MSA on each window."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        with_rpe: bool = True,
        with_pad_mask: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = (window_size, window_size)
        self.with_pad_mask = with_pad_mask
        self.attn = _WindowMSA(
            embed_dims=embed_dims, num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias, with_rpe=with_rpe,
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        Wh, Ww = self.window_size
        x = x.view(B, H, W, C)

        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = F.pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2))
        H_pad, W_pad = x.shape[1], x.shape[2]

        x = x.view(B, H_pad // Wh, Wh, W_pad // Ww, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, Wh * Ww, C)

        if self.with_pad_mask and (pad_h or pad_w):
            mask = x.new_zeros(1, H, W, 1)
            mask = F.pad(mask, (0, 0, pad_w // 2, pad_w - pad_w // 2,
                                pad_h // 2, pad_h - pad_h // 2),
                          value=float("-inf"))
            mask = mask.view(1, H_pad // Wh, Wh, W_pad // Ww, Ww, 1)
            mask = mask.permute(1, 3, 0, 2, 4, 5).reshape(-1, Wh * Ww)
            mask = mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, mask)
        else:
            out = self.attn(x)

        out = out.reshape(B, H_pad // Wh, W_pad // Ww, Wh, Ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)


# ---- FFN with depthwise 3x3 -------------------------------------------------

class _CrossFFN(nn.Module):
    """Conv2d-based FFN with a depthwise 3x3 in the middle (HRFormer's CrossFFN).

    Submodule names match mmpose's CrossFFN: fc1, act1, norm1, dw3x3, act2, norm2, fc2, act3, norm3.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        norm_cfg: dict,
        act_cfg: dict = dict(type="GELU"),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = nn.GELU() if act_cfg["type"] == "GELU" else nn.ReLU(inplace=True)
        _, self.norm1 = _build_norm(hidden_features, norm_cfg)
        self.dw3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                stride=1, groups=hidden_features, padding=1)
        self.act2 = nn.GELU() if act_cfg["type"] == "GELU" else nn.ReLU(inplace=True)
        _, self.norm2 = _build_norm(hidden_features, norm_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = nn.GELU() if act_cfg["type"] == "GELU" else nn.ReLU(inplace=True)
        _, self.norm3 = _build_norm(out_features, norm_cfg)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # x: (B, N, C) -> (B, C, H, W)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.act1(self.norm1(self.fc1(x)))
        x = self.act2(self.norm2(self.dw3x3(x)))
        x = self.act3(self.norm3(self.fc2(x)))
        return x.flatten(2).transpose(1, 2).contiguous()


# ---- HRFormer block --------------------------------------------------------

class HRFormerBlock(nn.Module):
    """LN -> LocalWindowSelfAttention -> + ; LN -> CrossFFN -> +."""

    expansion = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        with_rpe: bool = True,
        with_pad_mask: bool = False,
        norm_cfg: dict = dict(type="BN"),
        transformer_norm_cfg: dict = dict(type="LN", eps=1e-6),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        _, self.norm1 = _build_norm(in_features, transformer_norm_cfg)
        self.attn = _LocalWindowSelfAttention(
            in_features, num_heads=num_heads, window_size=window_size,
            with_rpe=with_rpe, with_pad_mask=with_pad_mask,
        )
        _, self.norm2 = _build_norm(out_features, transformer_norm_cfg)
        self.ffn = _CrossFFN(
            in_features=in_features,
            hidden_features=int(in_features * mlp_ratio),
            out_features=out_features,
            norm_cfg=norm_cfg,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x.permute(0, 2, 1).view(B, C, H, W)


# ---- HR-Former multi-resolution module -------------------------------------

class _HRFormerModule(nn.Module):
    """Like HRModule but branches are stacks of HRFormerBlock and the
    downsampling fuse path uses depthwise+pointwise convs."""

    def __init__(
        self,
        num_branches: int,
        in_channels: List[int],
        num_channels: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        window_sizes: Sequence[int],
        mlp_ratios: Sequence[float],
        drop_paths: Sequence[float],
        multiscale_output: bool = True,
        with_rpe: bool = True,
        with_pad_mask: bool = False,
        norm_cfg: dict = dict(type="BN"),
        transformer_norm_cfg: dict = dict(type="LN", eps=1e-6),
        upsample_cfg: dict = dict(mode="bilinear", align_corners=False),
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.in_channels = list(in_channels)
        self.upsample_cfg = upsample_cfg
        self.norm_cfg = norm_cfg

        # branches: each branch is a Sequential of HRFormerBlock(s)
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            blocks = []
            for k in range(num_blocks[i]):
                blocks.append(HRFormerBlock(
                    in_features=self.in_channels[i],
                    out_features=num_channels[i],
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=drop_paths[k],
                    with_rpe=with_rpe, with_pad_mask=with_pad_mask,
                    norm_cfg=norm_cfg, transformer_norm_cfg=transformer_norm_cfg,
                ))
            self.branches.append(nn.Sequential(*blocks))
            self.in_channels[i] = num_channels[i]

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

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
                        nn.Conv2d(in_channels[j], in_channels[i], 1, stride=1, bias=False),
                        _build_norm(in_channels[i], self.norm_cfg)[1],
                        nn.Upsample(scale_factor=2 ** (j - i),
                                     mode=self.upsample_cfg["mode"],
                                     align_corners=self.upsample_cfg.get("align_corners", None)),
                    ))
                elif j == i:
                    inner.append(None)
                else:
                    downs: list[nn.Module] = []
                    for k in range(i - j):
                        is_last = k == i - j - 1
                        out_c = in_channels[i] if is_last else in_channels[j]
                        seq = [
                            # depthwise 3x3 stride 2
                            nn.Conv2d(in_channels[j], in_channels[j], 3, stride=2,
                                      padding=1, groups=in_channels[j], bias=False),
                            _build_norm(in_channels[j], self.norm_cfg)[1],
                            # pointwise 1x1
                            nn.Conv2d(in_channels[j], out_c, 1, stride=1, bias=False),
                            _build_norm(out_c, self.norm_cfg)[1],
                        ]
                        if not is_last:
                            seq.append(nn.ReLU(inplace=False))
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


# ---- HRFormer top-level ----------------------------------------------------

class HRFormer(nn.Module):
    """HRFormer backbone (HRNet stem + Bottleneck stage1 + HRFormer stages 2-4).

    Submodule names mirror our :class:`HRNet`: ``conv1``/``bn1``/``conv2``/
    ``bn2``/``layer1``/``transition{N}``/``stage{N}``.
    """

    def __init__(
        self,
        extra: dict,
        in_channels: int = 3,
        norm_cfg: dict = dict(type="BN"),
        transformer_norm_cfg: dict = dict(type="LN", eps=1e-6),
    ) -> None:
        super().__init__()
        # Inject default upsample + drop-path schedule.
        extra = dict(extra)
        extra.setdefault("upsample", dict(mode="bilinear", align_corners=False))
        depths = [extra[s]["num_blocks"][0] * extra[s]["num_modules"]
                   for s in ("stage2", "stage3", "stage4")]
        drop_path_rate = extra.get("drop_path_rate", 0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        extra["stage2"]["drop_path_rates"] = dpr[: depths[0]]
        extra["stage3"]["drop_path_rates"] = dpr[depths[0]: depths[0] + depths[1]]
        extra["stage4"]["drop_path_rates"] = dpr[depths[0] + depths[1]:]

        self.norm_cfg = norm_cfg
        self.transformer_norm_cfg = transformer_norm_cfg
        self.with_rpe = extra.get("with_rpe", True)
        self.with_pad_mask = extra.get("with_pad_mask", False)
        self.upsample_cfg = extra["upsample"]

        # Stem (same as HRNet).
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
        _, self.bn1 = _build_norm(64, norm_cfg)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        _, self.bn2 = _build_norm(64, norm_cfg)
        self.relu = nn.ReLU(inplace=True)

        # stage1: Bottleneck (HRNet-style).
        s1 = extra["stage1"]
        s1_out = s1["num_channels"][0] * Bottleneck.expansion
        self.layer1 = _make_layer(Bottleneck, 64, s1_out, s1["num_blocks"][0])
        self.stage1_cfg = s1

        # stages 2-4: HRFormer modules.
        prev_channels = [s1_out]
        for i, name in enumerate(("stage2", "stage3", "stage4")):
            s = extra[name]
            cur_channels = list(s["num_channels"])
            transition = self._make_transition(prev_channels, cur_channels)
            self.add_module(f"transition{i + 1}", transition)

            modules = []
            in_branches = list(cur_channels)
            for m in range(s["num_modules"]):
                multiscale_output = True
                if name == "stage4" and m == s["num_modules"] - 1:
                    multiscale_output = s.get("multiscale_output", False)
                blocks_per_branch = s["num_blocks"][0]
                drop_paths = s["drop_path_rates"][m * blocks_per_branch:(m + 1) * blocks_per_branch]
                mod = _HRFormerModule(
                    num_branches=s["num_branches"],
                    in_channels=in_branches,
                    num_channels=s["num_channels"],
                    num_blocks=s["num_blocks"],
                    num_heads=s["num_heads"],
                    window_sizes=s["window_sizes"],
                    mlp_ratios=s["mlp_ratios"],
                    drop_paths=drop_paths,
                    multiscale_output=multiscale_output,
                    with_rpe=self.with_rpe, with_pad_mask=self.with_pad_mask,
                    norm_cfg=norm_cfg, transformer_norm_cfg=transformer_norm_cfg,
                    upsample_cfg=self.upsample_cfg,
                )
                modules.append(mod)
                in_branches = list(mod.in_channels)
            self.add_module(name, nn.Sequential(*modules))
            setattr(self, f"stage{i+2}_cfg", s)
            prev_channels = in_branches

    def _make_transition(self, pre: list[int], cur: list[int]) -> nn.ModuleList:
        n_pre, n_cur = len(pre), len(cur)
        layers: list[Optional[nn.Module]] = []
        for i in range(n_cur):
            if i < n_pre:
                if cur[i] != pre[i]:
                    layers.append(nn.Sequential(
                        nn.Conv2d(pre[i], cur[i], 3, stride=1, padding=1, bias=False),
                        _build_norm(cur[i], self.norm_cfg)[1],
                        nn.ReLU(inplace=True),
                    ))
                else:
                    layers.append(None)
            else:
                downs: list[nn.Module] = []
                for j in range(i + 1 - n_pre):
                    in_c = pre[-1]
                    out_c = cur[i] if j == (i - n_pre) else in_c
                    downs.append(nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                        _build_norm(out_c, self.norm_cfg)[1],
                        nn.ReLU(inplace=True),
                    ))
                layers.append(nn.Sequential(*downs))
        return nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
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

def _hrformer_extra(width: int) -> dict:
    """HRFormer-W{width} extra config (matches mmpose configs)."""
    return dict(
        drop_path_rate=0.1,
        with_rpe=True,
        stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK",
                     num_blocks=(2,), num_channels=(64,)),
        stage2=dict(num_modules=1, num_branches=2, block="HRFORMERBLOCK",
                     num_blocks=(2, 2), num_channels=(width, width * 2),
                     num_heads=[1, 2], mlp_ratios=[4, 4], window_sizes=[7, 7]),
        stage3=dict(num_modules=4, num_branches=3, block="HRFORMERBLOCK",
                     num_blocks=(2, 2, 2),
                     num_channels=(width, width * 2, width * 4),
                     num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                     window_sizes=[7, 7, 7]),
        stage4=dict(num_modules=2, num_branches=4, block="HRFORMERBLOCK",
                     num_blocks=(2, 2, 2, 2),
                     num_channels=(width, width * 2, width * 4, width * 8),
                     num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                     window_sizes=[7, 7, 7, 7]),
    )


HRFORMER_EXTRAS = {
    "small": _hrformer_extra(32),   # HRFormer-S
    "base": _hrformer_extra(78),    # HRFormer-B (base width 78)
}
