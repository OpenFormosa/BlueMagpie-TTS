#!/usr/bin/env python3
"""Compare batch vs serial per-row: stop step, latent stats, audio length."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import soundfile as sf
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

TEXTS = [
    "今天天氣真好，我們一起去公園散步吧。",
    "這個 feature 明天上線，記得先跑 regression test。",
    "會議改到下午三點，請大家準時參加。",
    "晚餐想吃什麼？附近新開了一間拉麵店。",
]


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    # per-text serial lengths (reference behavior)
    print("serial lengths:")
    for t in TEXTS:
        w = mlx_generate(lite, m, t, speaker_centroid=c)
        wav = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        print(f"  {len(wav)/lite.sample_rate:.1f}s  {t[:12]}")

    # instrumented batch: capture each row's patch count + first-step stop prob
    slot = "centroid"
    rows_len = []
    for t in TEXTS:
        tt_t = lite._build_inputs(t, None, None, slot)[0]
        rows_len.append(tt_t.shape[0])
    max_s = max(rows_len)
    print("row lens:", rows_len, "| padded:", max_s,
          "| offsets:", [max_s - L for L in rows_len])
    return 0


if __name__ == "__main__":
    sys.exit(main())
