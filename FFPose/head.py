"""RTMCC head: large-kernel conv -> ScaleNorm+Linear -> GAU (RTMCCBlock) ->
two linear classifiers producing SimCC x/y logits.

Submodule names match mmpose's RTMCCHead so checkpoints load cleanly.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import DropPath


class ScaleNorm(nn.Module):
    """Scale-only layer-norm. Single learned scalar gain ``g``."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    """Per-channel learned multiplicative scale."""

    def __init__(self, dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def rope(x: Tensor, dim: int) -> Tensor:
    """Rotary position embedding along ``dim``. Direct port of mmpose's rope()."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for s in spatial_shape:
        total_len *= s

    position = torch.arange(total_len, dtype=torch.int, device=x.device).reshape(spatial_shape)
    for i in range(dim[-1] + 1, len(shape) - 1):
        position = position.unsqueeze(-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = position[..., None] * inv_freq[None, None, :]
    sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (`Transformer Quality in Linear Time`,
    https://arxiv.org/abs/2202.10447), self-attn variant only.

    Direct port preserving every parameter name (uv, o, gamma, beta, w, ln, res_scale).
    """

    def __init__(
        self,
        num_token: int,
        in_token_dims: int,
        out_token_dims: int,
        expansion_factor: int = 2,
        s: int = 128,
        eps: float = 1e-5,
        dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        attn_type: str = "self-attn",
        act_fn: str = "SiLU",
        bias: bool = False,
        use_rel_bias: bool = True,
        pos_enc: bool = False,
    ) -> None:
        super().__init__()
        if attn_type != "self-attn":
            raise NotImplementedError("only self-attn supported in FFPose")

        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.pos_enc = pos_enc
        self.attn_type = attn_type

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.e = int(in_token_dims * expansion_factor)

        if use_rel_bias:
            self.w = nn.Parameter(torch.rand([2 * num_token - 1], dtype=torch.float))

        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn in ("SiLU", nn.SiLU):
            self.act_fn = nn.SiLU(True)
        elif act_fn in ("ReLU", nn.ReLU):
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError(f"act_fn {act_fn!r}")

        self.shortcut = in_token_dims == out_token_dims
        if self.shortcut:
            self.res_scale = Scale(in_token_dims)

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len: int) -> Tensor:
        t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        return t[..., r:-r]

    def _forward(self, x: Tensor) -> Tensor:
        x = self.ln(x)
        uv = self.act_fn(self.uv(x))                # [B, K, 2e+s]
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta  # [B,K,2,s]
        if self.pos_enc:
            base = rope(base, dim=1)
        q, k = torch.unbind(base, dim=2)           # each [B, K, s]

        qk = torch.bmm(q, k.permute(0, 2, 1))       # [B, K, K]
        if self.use_rel_bias:
            bias = self.rel_pos_bias(q.size(1))
            qk = qk + bias[:, :q.size(1), :q.size(1)]

        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)
        out = u * torch.bmm(kernel, v)              # [B, K, e]
        return self.o(out)

    def forward(self, x: Tensor) -> Tensor:
        main = self.drop_path(self._forward(x))
        return self.res_scale(x) + main if self.shortcut else main


class RTMCCHead(nn.Module):
    """SimCC-classification head used in RTMPose.

    Forward output: ``(pred_x, pred_y)`` where ``pred_x`` is shape
    ``(B, K, W*split_ratio)`` and ``pred_y`` is ``(B, K, H*split_ratio)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: dict = dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0.0, drop_path=0.0, act_fn="SiLU",
            use_rel_bias=False, pos_enc=False,
        ),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]
        self.final_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1, padding=final_layer_kernel_size // 2,
        )
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg["hidden_dims"], bias=False),
        )

        W = int(input_size[0] * simcc_split_ratio)
        H = int(input_size[1] * simcc_split_ratio)

        self.gau = RTMCCBlock(
            num_token=out_channels,
            in_token_dims=gau_cfg["hidden_dims"],
            out_token_dims=gau_cfg["hidden_dims"],
            s=gau_cfg["s"],
            expansion_factor=gau_cfg["expansion_factor"],
            dropout_rate=gau_cfg["dropout_rate"],
            drop_path=gau_cfg["drop_path"],
            attn_type="self-attn",
            act_fn=gau_cfg["act_fn"],
            use_rel_bias=gau_cfg["use_rel_bias"],
            pos_enc=gau_cfg["pos_enc"],
        )
        self.cls_x = nn.Linear(gau_cfg["hidden_dims"], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg["hidden_dims"], H, bias=False)

    def forward(self, feats) -> Tuple[Tensor, Tensor]:
        # feats may be a tuple from a backbone; take last (deepest) feature.
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        feats = self.final_layer(feats)             # B, K, H, W
        feats = torch.flatten(feats, 2)             # B, K, H*W
        feats = self.mlp(feats)                     # B, K, hidden
        feats = self.gau(feats)
        return self.cls_x(feats), self.cls_y(feats)
