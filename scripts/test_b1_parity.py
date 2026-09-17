#!/usr/bin/env python3
"""B=1 batch-path vs serial-path: identical seeds -> outputs should match."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import mlx_generate
from bluemagpie.mlx.model_mlx import mlx_generate_batch

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
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    mlx_generate(lite, m, TEXT)  # warm

    a = mlx_generate(lite, m, TEXT, speaker_centroid=c, seed=7)
    b1 = mlx_generate_batch(lite, m, [TEXT], speaker_centroid=c, seed=7)[0]
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    b_np = b1.numpy() if hasattr(b1, "numpy") else np.asarray(b1)
    n = min(len(a_np), len(b_np))
    cc = float(np.corrcoef(a_np[:n] - a_np[:n].mean(), b_np[:n] - b_np[:n].mean())[0, 1])
    print(f"serial {len(a_np)/lite.sample_rate:.1f}s vs batchB=1 {len(b_np)/lite.sample_rate:.1f}s "
          f"| corr={cc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
