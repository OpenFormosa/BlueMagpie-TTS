#!/usr/bin/env python3
"""Compare fused vs manual mamba projection numerically."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.barbet_mlx import _lin

MD = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS"),
    "snapshots",
    max(
        os.listdir(os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots")),
        key=lambda d: os.path.getmtime(
            os.path.join(
                os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots"), d
            )
        ),
    ),
)


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    bar = m.barbet
    # pick the first mamba layer (layer 0 is attention in this hybrid model)
    midx = next(i for i, lt in enumerate(bar.layer_types) if lt == "mamba")
    pre = f"layers.{midx}.mixer."
    print(f"testing mamba layer {midx}")

    def dq(name):
        a = bar.p[name]
        if a.dtype == mx.float32:
            return a
        from bluemagpie.mlx.barbet_mlx import _QUANTIZED
        e = _QUANTIZED.get(id(a))
        q = e["q"]
        return mx.dequantize(q[0], q[1], q[2], group_size=q[3], bits=q[4])

    x = mx.random.normal((1, bar.cfg.hidden_size))
    mx.eval(x)

    # manual: five separate int4 linears (original order z,x,b,c,dt)
    zz = _lin(x, dq(pre + "in_proj_z.weight"))
    xx = _lin(x, dq(pre + "in_proj_x.weight"))
    bb = _lin(x, dq(pre + "in_proj_b.weight"))
    cc = _lin(x, dq(pre + "in_proj_c.weight"))
    dt = _lin(x, dq(pre + "in_proj_dt.weight"))
    manual = mx.concatenate([xx, bb, cc, zz, dt], axis=-1)
    mx.eval(manual)

    # fused: one int4 GEMM through the fused slot + fused bias
    fu = bar._mamba_fused(midx)
    proj = _lin(x, fu["slot"])
    bias = bar._mamba_bias(midx)
    if bias is not None:
        proj = proj + bias
    mx.eval(proj)

    # the fused slot was requantized from dequantized fp32 weights, so compare
    # against the dequantized-manual path (fp32 GEMM), which removes double-
    # quantization noise from the comparison:
    manual_fp32 = mx.concatenate([
        _lin(x, dq(pre + "in_proj_x.weight")),
        _lin(x, dq(pre + "in_proj_b.weight")),
        _lin(x, dq(pre + "in_proj_c.weight")),
        _lin(x, dq(pre + "in_proj_z.weight")),
        _lin(x, dq(pre + "in_proj_dt.weight")),
    ], axis=-1)
    if bias is not None:
        manual_fp32 = manual_fp32 + bias
    mx.eval(manual_fp32)

    print("fused vs manual(int4)  max|diff|:", float(mx.max(mx.abs(proj - manual))))
    print("fused vs manual(fp32)  max|diff|:", float(mx.max(mx.abs(proj - manual_fp32))))

    # check split correctness on the fused output: segment lengths
    parts = mx.split(proj, np.array(fu["splits"]), axis=-1)
    print("split lens:", [p.shape[-1] for p in parts],
          "| expected:", [bar.inner, bar.m_groups * bar.d_state,
                          bar.m_groups * bar.d_state, bar.inner, bar.m_heads])
    return 0


if __name__ == "__main__":
    sys.exit(main())
