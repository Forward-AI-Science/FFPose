"""Swin Transformer backbone (V1) for top-down pose estimation.

Direct port of mmpose's ``SwinTransformer``. Submodule layout matches the
mmpose checkpoint exactly:

  patch_embed.projection.{weight,bias}
  patch_embed.norm.{weight,bias}             # if patch_norm=True (default)
  stages.{i}.blocks.{j}.norm1.{weight,bias}
  stages.{i}.blocks.{j}.attn.w_msa.relative_position_bias_table
  stages.{i}.blocks.{j}.attn.w_msa.qkv.{weight,bias}
  stages.{i}.blocks.{j}.attn.w_msa.proj.{weight,bias}
  stages.{i}.blocks.{j}.norm2.{weight,bias}
  stages.{i}.blocks.{j}.ffn.layers.0.0.{weight,bias}
  stages.{i}.blocks.{j}.ffn.layers.1.{weight,bias}
  stages.{i}.downsample.norm.{weight,bias}   # PatchMerging
  stages.{i}.downsample.reduction.weight     # PatchMerging
  norm{i}.{weight,bias}                       # for i in out_indices
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import DropPath


# ---- low-level building blocks ---------------------------------------------

class _FFN(nn.Module):
    """mmcv-compatible FFN. ``layers.0`` is Sequential(Linear, GELU, Dropout);
    ``layers.1`` is Linear; followed by Dropout (no params). ``forward`` adds
    residual identity."""

    def __init__(self, embed_dims: int, feedforward_channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.GELU(),
                nn.Dropout(drop),
            ),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.dropout_after_ffn = nn.Dropout(drop)

    def forward(self, x: Tensor, identity: Optional[Tensor] = None) -> Tensor:
        out = self.layers(x)
        out = self.dropout_after_ffn(out)
        if identity is None:
            identity = x
        return identity + out


class _WindowMSA(nn.Module):
    """Window-MSA with relative position bias.

    Submodule names match mmpose's ``WindowMSA`` (qkv, proj,
    relative_position_bias_table, relative_position_index).
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

        # 2*Wh-1 * 2*Ww-1, num_heads
        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        # rel_position_index: (Wh*Ww, Wh*Ww)
        rel_index_coords = self._double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_index = rel_index_coords + rel_index_coords.T
        rel_index = rel_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _double_step_seq(step1: int, len1: int, step2: int, len2: int) -> Tensor:
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        Wh, Ww = self.window_size
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wh * Ww, Wh * Ww, -1).permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _ShiftWindowMSA(nn.Module):
    """Shifted window MSA (with optional cyclic shift + attention mask)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        self.window_size = window_size
        self.shift_size = shift_size
        self.w_msa = _WindowMSA(
            embed_dims=embed_dims, num_heads=num_heads,
            window_size=(window_size, window_size),
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.drop = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @staticmethod
    def _window_partition(x: Tensor, window_size: int) -> Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    @staticmethod
    def _window_reverse(windows: Tensor, H: int, W: int, window_size: int) -> Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, query: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        B, L, C = query.shape
        H, W = hw_shape
        if L != H * W:
            raise ValueError("input feature has wrong size")
        query = query.view(B, H, W, C)

        # Pad to multiple of window_size.
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # SW-MSA attention mask
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            cnt = 0
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = h_slices
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            shifted = query
            attn_mask = None

        # window-partition -> attn -> reverse
        q_windows = self._window_partition(shifted, self.window_size).view(
            -1, self.window_size ** 2, C
        )
        a_windows = self.w_msa(q_windows, mask=attn_mask)
        a_windows = a_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(a_windows, H_pad, W_pad, self.window_size)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return self.drop(x)


class _SwinBlock(nn.Module):
    """LayerNorm -> ShiftWindowMSA -> + residual ; LayerNorm -> FFN -> + residual."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = _ShiftWindowMSA(
            embed_dims=embed_dims, num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop_rate, proj_drop=drop_rate,
            drop_path=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = _FFN(embed_dims, feedforward_channels, drop=drop_rate)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        # attention sub-block
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity
        # ffn sub-block
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


class _PatchEmbed(nn.Module):
    """Conv-based patch embedding with optional norm."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        kernel_size: int = 4,
        stride: int = 4,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.kernel_size = kernel_size
        self.stride = stride
        # mmpose uses an "AdaptivePadding(corner)" pad before the conv;
        # for input multiples of patch_size this is a no-op.
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride,
        )
        self.norm = nn.LayerNorm(embed_dims) if norm else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # AdaptivePadding(mode='corner'): pad right/bottom so spatial dims are
        # divisible by stride. Most pose inputs are exact multiples already.
        H, W = x.shape[-2:]
        pad_r = (self.stride - W % self.stride) % self.stride
        pad_b = (self.stride - H % self.stride) % self.stride
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.projection(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


class _PatchMerging(nn.Module):
    """Merge 2x2 patches via Unfold + Linear reduction (mmpose layout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.sampler = nn.Unfold(kernel_size=kernel_size, stride=stride, dilation=1, padding=0)
        sample_dim = kernel_size * kernel_size * in_channels
        self.norm = nn.LayerNorm(sample_dim) if norm else None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=False)

    def forward(self, x: Tensor, input_size: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int]]:
        B, L, C = x.shape
        H, W = input_size
        if L != H * W:
            raise ValueError("input has wrong size for patch merging")
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B,C,H,W
        # AdaptivePadding(corner) for unfold:
        pad_r = (self.stride - W % self.stride) % self.stride
        pad_b = (self.stride - H % self.stride) % self.stride
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            H, W = x.shape[-2:]
        x = self.sampler(x)  # (B, sample_dim, L_out)
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1
        x = x.transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


class _SwinStage(nn.Module):
    """One Swin stage: ``depth`` blocks, optional patch merging at the end."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: Sequence[float] = (0.0,),
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if isinstance(drop_path_rate, (int, float)):
            drop_path_rate = [float(drop_path_rate)] * depth
        if len(drop_path_rate) != depth:
            raise ValueError("drop_path_rate length must equal depth")
        self.blocks = nn.ModuleList([
            _SwinBlock(
                embed_dims=embed_dims, num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size, shift=(i % 2 == 1),
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate[i],
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]):
        for blk in self.blocks:
            x = blk(x, hw_shape)
        if self.downsample is not None:
            x_down, hw_down = self.downsample(x, hw_shape)
            return x_down, hw_down, x, hw_shape
        return x, hw_shape, x, hw_shape


# ---- top-level backbone ----------------------------------------------------

class SwinTransformer(nn.Module):
    """Swin Transformer V1 backbone (windowed self-attention, hierarchical).

    For pose estimation (top-down) we only use the deepest stage's output as
    the feature map fed to the heatmap head — but multi-stage outputs are
    available via ``out_indices``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: int = 4,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        strides: Sequence[int] = (4, 2, 2, 2),
        out_indices: Sequence[int] = (3,),
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        patch_norm: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if strides[0] != patch_size:
            raise ValueError("strides[0] must equal patch_size")
        self.out_indices = tuple(out_indices)
        num_layers = len(depths)

        self.patch_embed = _PatchEmbed(
            in_channels=in_channels, embed_dims=embed_dims,
            kernel_size=patch_size, stride=strides[0], norm=patch_norm,
        )
        self.drop_after_pos = nn.Dropout(drop_rate)

        # stochastic depth schedule across all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        cur_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = _PatchMerging(
                    in_channels=cur_channels,
                    out_channels=2 * cur_channels,
                    kernel_size=2, stride=strides[i + 1],
                    norm=patch_norm,
                )
            else:
                downsample = None
            stage = _SwinStage(
                embed_dims=cur_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * cur_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
            )
            self.stages.append(stage)
            if downsample is not None:
                cur_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Output norm per requested stage. Submodule names: ``norm{i}``.
        for i in self.out_indices:
            self.add_module(f"norm{i}", nn.LayerNorm(self.num_features[i]))

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw = stage(x, hw_shape)
            if i in self.out_indices:
                norm = getattr(self, f"norm{i}")
                out_norm = norm(out)
                out_norm = out_norm.view(-1, *out_hw, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out_norm)
        return tuple(outs)


# ---- variant configs --------------------------------------------------------

# Source: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-{t,b,l}_*.py
SWIN_VARIANTS = {
    "t": dict(embed_dims=96,  depths=(2, 2, 6, 2),  num_heads=(3, 6, 12, 24),  drop_path_rate=0.2),
    "b": dict(embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),  drop_path_rate=0.3),
    "l": dict(embed_dims=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), drop_path_rate=0.3),
}
