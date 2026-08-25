#!/usr/bin/env python3
"""Isolate where the fused-mamba numeric difference comes from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.barbet_mlx import _lin, _QUANTIZED

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
    midx = next(i for i, lt in enumerate(bar.layer_types) if lt == "mamba")
    pre = f"layers.{midx}.mixer."

    def dq(name):
        a = bar.p[name]
        if a.dtype == mx.float32:
            return a
        e = _QUANTIZED.get(id(a))
        q = e["q"]
        return mx.dequantize(q[0], q[1], q[2], group_size=q[3], bits=q[4])

    names = ["in_proj_x", "in_proj_b", "in_proj_c", "in_proj_z", "in_proj_dt"]
    ws = [dq(pre + n + ".weight") for n in names]

    # 1. fused concat vs per-weight concat (weights themselves)
    w_all = mx.concatenate(ws, axis=0)
    fu = bar._mamba_fused(midx)
    from bluemagpie.mlx.barbet_mlx import register_quantized
    ent = _QUANTIZED.get(id(fu["slot"]))
    q = ent["q"]
    w_fused_dq = mx.dequantize(q[0], q[1], q[2], group_size=q[3], bits=q[4])
    print("weights: fused-dq vs concat-dq max|diff|:",
          float(mx.max(mx.abs(w_fused_dq - w_all))))

    # 2. activations: fp32 GEMM on w_all vs the fused int4 kernel output
    x = mx.random.normal((1, bar.cfg.hidden_size))
    mx.eval(x)
    bias = bar._mamba_bias(midx)
    y_fp32 = x @ mx.transpose(w_all)
    if bias is not None:
        y_fp32 = y_fp32 + bias
    y_int4 = _lin(x, fu["slot"])
    if bias is not None:
        y_int4 = y_int4 + bias
    mx.eval([y_fp32, y_int4])
    print("act: fp32-gemm(w_all) vs fused-int4 max|diff|:",
          float(mx.max(mx.abs(y_fp32 - y_int4))))

    # 3. five separate int4 linears (through their packed slots) vs fused int4
    ys = [_lin(x, bar.p[pre + n + ".weight"]) for n in names]
    y_sep = mx.concatenate(ys, axis=-1)
    mx.eval(y_sep)
    print("act: sep-int4 vs fused-int4      max|diff|:",
          float(mx.max(mx.abs(y_sep - y_int4))))

    # 4. which segments differ most?
    parts_a = mx.split(y_sep, np.array(fu["splits"]), axis=-1)
    parts_b = mx.split(y_int4, np.array(fu["splits"]), axis=-1)
    for n, pa, pb in zip(["x", "b", "c", "z", "dt"], parts_a, parts_b):
        print(f"  seg {n}: max|diff| = {float(mx.max(mx.abs(pa - pb))):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
