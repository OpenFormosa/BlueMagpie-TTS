"""LocEnc (VoxCPMLocEnc) — MLX forward.

Thin wrapper over :class:`MiniCPMMLX`: project each latent patch, prepend a
learned special token, run the non-causal MiniCPM encoder, and take the special
token's output as the patch embedding. Mirrors ``locenc/local_encoder.py``.
"""

from __future__ import annotations

import mlx.core as mx

from .barbet_mlx import _lin
from .convert import to_mx
from .minicpm_mlx import MiniCPMMLX


class LocEncMLX:
    def __init__(self, locenc) -> None:
        self.in_w = to_mx(locenc.in_proj.weight)
        self.in_b = to_mx(locenc.in_proj.bias)
        self.special = to_mx(locenc.special_token)        # [1, 1, 1, hidden]
        self.encoder = MiniCPMMLX(locenc.encoder)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, P, D] -> [B, T, hidden]
        b, t, p, _ = x.shape
        h = _lin(x, self.in_w, self.in_b)                 # [B, T, P, hidden]
        hidden = h.shape[-1]
        special = mx.broadcast_to(self.special, (b, t, 1, hidden))
        h = mx.concatenate([special, h], axis=2)          # [B, T, P+1, hidden]
        h = h.reshape(b * t, p + 1, hidden)
        out = self.encoder(h, is_causal=False)            # [B*T, P+1, hidden]
        cls = out[:, 0, :]                                # special-token output
        return cls.reshape(b, t, hidden)
