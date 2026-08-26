#!/usr/bin/env python3
"""Force the WSOLA path: lower target_cps so slowing must engage."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import mlx_generate
from bluemagpie.mlx.pace import count_speech_units

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
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    txt = "哈囉大家，我是 BlueMagpie 的女生版語音。"
    mlx_generate(lite, m, txt)  # warm

    units = count_speech_units(txt)
    for cps in (3.0, 3.5):
        a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=7,
                         cfg_value=2.8, target_cps=cps)
        wav = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        dur = len(wav) / lite.sample_rate
        print(f"target {cps}: {dur:.2f}s -> {units/dur:.2f} units/s")
        sf.write(f"lite_out/pace_forced_{cps}.wav", wav, lite.sample_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
