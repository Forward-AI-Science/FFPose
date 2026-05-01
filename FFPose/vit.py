"""ViT backbone (mmpretrain-style) for ViTPose.

Submodule names match the mmpose ViTPose checkpoint exactly:
    backbone.cls_token, backbone.pos_embed
    backbone.patch_embed.projection.{weight,bias}
    backbone.layers.{i}.ln1.{weight,bias}
    backbone.layers.{i}.attn.qkv.{weight,bias}
    backbone.layers.{i}.attn.proj.{weight,bias}
    backbone.layers.{i}.ln2.{weight,bias}
    backbone.layers.{i}.ffn.layers.0.0.{weight,bias}   # first Linear
    backbone.layers.{i}.ffn.layers.1.{weight,bias}     # second Linear
    backbone.ln1.{weight,bias}
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import DropPath


class PatchEmbed(nn.Module):
    """Image -> sequence of patch embeddings via a strided Conv2d. Optional
    extra padding (used by ViTPose's ``patch_cfg=dict(padding=2)``)."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 384,
        patch_size: int = 16,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=patch_size, stride=patch_size, padding=padding,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.projection(x)             # (B, C, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # (B, Hp*Wp, C)
        return x, (Hp, Wp)


class MultiheadSelfAttention(nn.Module):
    """Standard MHSA with combined ``qkv`` linear, matching mmpretrain naming."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} not divisible by num_heads {num_heads}")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.qkv = nn.Linear(embed_dims, 3 * embed_dims, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, h, N, d)
        # PyTorch 2.x flash/sdpa fused kernel where supported.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class FFN(nn.Module):
    """Two-layer MLP with mmpretrain's nested ``layers`` Sequential structure
    so checkpoint key names like ``ffn.layers.0.0.weight`` match exactly."""

    def __init__(self, embed_dims: int, feedforward_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.GELU(),
            ),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """LN -> MSA -> drop_path -> + residual ; LN -> FFN -> drop_path -> + residual."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dims)
        self.attn = MultiheadSelfAttention(embed_dims, num_heads, qkv_bias=qkv_bias)
        self.ln2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, feedforward_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


def _resize_pos_embed(
    pos_embed: Tensor, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
) -> Tensor:
    """Bilinear-interpolate the spatial position embedding to a new HxW grid.
    Expects pos_embed shape (1, src_h*src_w, C)."""
    if src_hw == dst_hw:
        return pos_embed
    sh, sw = src_hw
    dh, dw = dst_hw
    B, N, C = pos_embed.shape
    assert N == sh * sw, f"pos_embed length {N} != src_h*src_w {sh*sw}"
    pe = pos_embed.transpose(1, 2).reshape(B, C, sh, sw)
    pe = F.interpolate(pe, size=(dh, dw), mode="bicubic", align_corners=False)
    return pe.flatten(2).transpose(1, 2).contiguous()


class VisionTransformer(nn.Module):
    """ViTPose-style ViT: patch embed -> +pos -> N transformer blocks -> ln -> reshape to feature map.

    Notes:
      * ``cls_token`` and the leading slot of ``pos_embed`` exist (loaded from
        the MAE checkpoint) but are unused at inference because
        ``with_cls_token=False``. Loading them is still correct because the
        rest of the keys would otherwise mismatch.
      * Position embedding is resized at runtime if the saved grid no longer
        matches the requested input.
    """

    # Architectures matching mmpretrain.VisionTransformer's named arches.
    arch_zoo = {
        "small": dict(embed_dims=384, num_layers=12, num_heads=12, feedforward_channels=384 * 4),
        "base":  dict(embed_dims=768, num_layers=12, num_heads=12, feedforward_channels=768 * 4),
        "large": dict(embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=1024 * 4),
        "huge":  dict(embed_dims=1280, num_layers=32, num_heads=16, feedforward_channels=1280 * 4),
    }

    def __init__(
        self,
        arch: str | dict = "base",
        img_size: Tuple[int, int] = (256, 192),   # (H, W) — mmpose convention here
        patch_size: int = 16,
        patch_padding: int = 2,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        with_cls_token: bool = False,
        out_type: str = "featmap",
    ) -> None:
        super().__init__()
        if isinstance(arch, str):
            arch_cfg = self.arch_zoo[arch]
        else:
            arch_cfg = arch

        self.embed_dims: int = int(arch_cfg["embed_dims"])
        self.num_layers: int = int(arch_cfg["num_layers"])
        self.num_heads: int = int(arch_cfg["num_heads"])
        self.feedforward_channels: int = int(arch_cfg["feedforward_channels"])
        self.with_cls_token = with_cls_token
        self.out_type = out_type
        self.img_size = img_size  # (H, W)
        self.patch_size = patch_size
        self.patch_padding = patch_padding

        H, W = img_size
        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dims=self.embed_dims,
            patch_size=patch_size,
            padding=patch_padding,
        )
        Hp = (H + 2 * patch_padding - patch_size) // patch_size + 1
        Wp = (W + 2 * patch_padding - patch_size) // patch_size + 1
        self._patch_grid = (Hp, Wp)

        # Saved sizes (cls + spatial) — matches mmpretrain layout regardless of
        # whether cls is used. Pos embed includes the cls slot when
        # with_cls_token is True; for ViTPose configs cls=False but the
        # saved pos_embed still has length 1+Hp*Wp.
        num_extra = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_extra + Hp * Wp, self.embed_dims))

        # Drop-path schedule (we still build the modules; in eval they're identity).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                feedforward_channels=self.feedforward_channels,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(self.num_layers)
        ])
        self.ln1 = nn.LayerNorm(self.embed_dims)

    def forward(self, x: Tensor) -> Tensor:
        """Returns a feature map ``(B, C, Hp, Wp)`` (out_type='featmap')."""
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)            # (B, N, C), N = Hp*Wp

        # Slice cls slot off pos_embed (config sets with_cls_token=False).
        pe = self.pos_embed[:, 1:, :]                # (1, Hp_saved*Wp_saved, C)
        Hp_saved, Wp_saved = self._patch_grid
        if (Hp, Wp) != (Hp_saved, Wp_saved):
            pe = _resize_pos_embed(pe, (Hp_saved, Wp_saved), (Hp, Wp))
        x = x + pe

        for blk in self.layers:
            x = blk(x)
        x = self.ln1(x)

        if self.out_type == "featmap":
            return x.transpose(1, 2).reshape(B, self.embed_dims, Hp, Wp).contiguous()
        if self.out_type == "raw":
            return x
        raise ValueError(f"unsupported out_type {self.out_type!r}")
