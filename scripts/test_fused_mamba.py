#!/usr/bin/env python3
"""Validate + benchmark the fused mamba step."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statistics

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import mlx_generate

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

    # --- correctness: full generation vs the pre-fusion fp32 references ---
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    txt = "這個 feature 明天上線，記得先跑 regression test。"

    def corr(a, b):
        n = min(len(a), len(b))
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    ref = torch.load("lite_out/fp32_1.pt", weights_only=True).numpy()
    mlx_generate(lite, m, txt)  # warm
    t0 = time.time()
    a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=42, cfg_value=2.8)
    dt = time.time() - t0
    dur = a.shape[-1] / lite.sample_rate
    print(f"corr vs pre-fusion ref: {corr(ref, a.numpy()):.4f} | rtf={dt/dur:.2f}", flush=True)

    # --- micro-bench: barbet.step ---
    bc = m.barbet.init_cache()
    h_b = m.barbet.cfg.hidden_size
    xb = mx.zeros((1, h_b))
    for p in range(4):
        mx.eval(m.barbet.step(xb, p, bc))

    def bench(fn, n=6):
        ts = []
        for _ in range(n):
            t0 = time.time()
            r = fn()
            mx.eval(r)
            ts.append(time.time() - t0)
        return statistics.median(ts) * 1000

    print(f"barbet.step (fused): {bench(lambda: m.barbet.step(xb, 5, bc)):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
