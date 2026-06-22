"""Projection adapter between Barbet hidden space and VoxCPM2 LM hidden space.

Barbet's hidden states are not drop-in compatible with the space VoxCPM2's
pretrained downstream modules (FSQ layer, RALM fusion, DiT projections, stop
head) were trained in. The adapter is the explicit bridge:

    barbet_hidden (H_b) --RMSNorm--Linear--> H_v --[zero-init residual MLP]*N--> H_v

The residual MLP blocks have their output projections zero-initialized, so at
init the adapter is exactly RMSNorm + Linear. Training can then grow extra
capacity without perturbing the warm-start behaviour.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import AdapterConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return x.to(dtype=self.weight.dtype) * self.weight


class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, dim: int, ffn_mult: float, eps: float) -> None:
        super().__init__()
        inner = int(dim * ffn_mult)
        self.norm = RMSNorm(dim, eps)
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class ProjectionAdapter(nn.Module):
    """Maps Barbet hidden states into the VoxCPM2 semantic LM space."""

    def __init__(self, in_dim: int, out_dim: int, config: AdapterConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(in_dim, config.rms_norm_eps)
        self.proj = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualSwiGLUBlock(out_dim, config.ffn_mult, config.rms_norm_eps)
                for _ in range(config.num_residual_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(x))
        for block in self.blocks:
            x = block(x)
        return x
