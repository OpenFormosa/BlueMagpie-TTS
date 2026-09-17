#!/usr/bin/env python3
"""Try int8 embedding as a middle ground (half the error of int4)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model, reactivate_int4
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

TEXT = "今天天氣真好，我們一起去公園散步吧。"


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")  # fp32 embed
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    # measure embedding quantization error directly at int8 vs int4
    import bluemagpie.mlx.barbet_mlx as bm
    w = m.embed
    for bits in (8, 6, 4):
        wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
        err = float(mx.max(mx.abs(mx.dequantize(wq, sc, bi, group_size=64, bits=bits) - w)))
        print(f"int{bits} embed: max weight error {err:.4f}, "
              f"size {(wq.nbytes + sc.nbytes + bi.nbytes)/1024/1024:.0f}MB vs "
              f"{w.nbytes/1024/1024:.0f}MB fp32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
