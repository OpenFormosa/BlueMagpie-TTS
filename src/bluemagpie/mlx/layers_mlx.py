"""Small MLX layers: FSQ and the Barbet->VoxCPM projection adapter."""

from __future__ import annotations

import mlx.core as mx

from .barbet_mlx import _lin, _rms_norm, _silu
from .convert import to_mx, torch_params_to_mx


class FSQMLX:
    """ScalarQuantizationLayer (eval): in_proj -> tanh -> round*scale/scale -> out_proj."""

    def __init__(self, fsq) -> None:
        self.in_w = to_mx(fsq.in_proj.weight); self.in_b = to_mx(fsq.in_proj.bias)
        self.out_w = to_mx(fsq.out_proj.weight); self.out_b = to_mx(fsq.out_proj.bias)
        self.scale = fsq.scale

    def __call__(self, x: mx.array) -> mx.array:
        h = mx.tanh(_lin(x, self.in_w, self.in_b))
        h = mx.round(h * self.scale) / self.scale
        return _lin(h, self.out_w, self.out_b)


class AdapterMLX:
    """ProjectionAdapter: RMSNorm + Linear + N zero-init residual SwiGLU blocks."""

    def __init__(self, adapter) -> None:
        self.p = torch_params_to_mx(adapter)
        self.eps = adapter.norm.eps
        self.num_blocks = len(adapter.blocks)

    def __call__(self, x: mx.array) -> mx.array:
        h = _rms_norm(x, self.p["norm.weight"], self.eps)
        h = _lin(h, self.p["proj.weight"], self.p["proj.bias"])
        for i in range(self.num_blocks):
            pre = f"blocks.{i}."
            n = _rms_norm(h, self.p[pre + "norm.weight"], self.eps)
            h = h + _lin(
                _silu(_lin(n, self.p[pre + "gate_proj.weight"])) * _lin(n, self.p[pre + "up_proj.weight"]),
                self.p[pre + "down_proj.weight"],
            )
        return h
