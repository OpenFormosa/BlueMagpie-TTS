#!/usr/bin/env python3
"""Validate int4 embedding: quality (latent parity vs fp32 embed) + speed."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
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

TEXT = "今天天氣真好，我們一起去公園散步吧。"


def main():
    # baseline without quantized embedding is not directly loadable side by
    # side on 16GB; instead compare audio against the reference from the
    # previous run (same seed, same text, pre-embed-quant).
    t0 = time.time()
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    load_t = time.time() - t0
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    mlx_generate(lite, m, TEXT)  # warm

    t0 = time.time()
    w = mlx_generate(lite, m, TEXT, speaker_centroid=c, seed=7)
    gen_t = time.time() - t0
    wav = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
    sf.write("lite_out/qembed_test.wav", wav, lite.sample_rate)

    ref, sr = sf.read("lite_out/smoke_full.wav")  # earlier fp32-embed output
    n = min(len(ref), len(wav))
    a, b = ref[:n] - ref[:n].mean(), wav[:n] - wav[:n].mean()
    cc = float(np.corrcoef(a, b)[0, 1])
    print(f"load {load_t:.1f}s | gen RTF {gen_t/(len(wav)/sr):.2f} "
          f"| corr vs pre-qembed reference: {cc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
