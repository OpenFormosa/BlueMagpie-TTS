#!/usr/bin/env python3
"""Validate the WSOLA pace branch: unit counter, stretch, and A/B wavs."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import mlx_generate
from bluemagpie.mlx.pace import count_speech_units, target_pace_speed

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
    # unit counter sanity
    cases = [
        ("哈囉大家，我是 BlueMagpie 的女生版語音。", None),
        ("這個 feature 明天上線。", None),
        ("電話是 0912345678。", None),
    ]
    for text, _ in cases:
        print(f"units({text[:14]}...) = {count_speech_units(text)}")

    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    txt = "哈囉大家，我是 BlueMagpie 的女生版語音。"

    mlx_generate(lite, m, txt)  # warm

    for cps in (0.0, 4.0):
        t0 = time.time()
        a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=7,
                         cfg_value=2.8, target_cps=cps)
        dt = time.time() - t0
        wav = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        dur = len(wav) / lite.sample_rate
        units = count_speech_units(txt)
        rate = target_pace_speed(len(wav), lite.sample_rate, txt)
        print(f"cps={cps}: {dur:.2f}s audio, gen={dt:.2f}s (rtf {dt/dur:.2f}), "
              f"{units/dur:.2f} units/s, applied_rate={rate:.3f}")
        sf.write(f"lite_out/pace_cps{cps:.0f}.wav", wav, lite.sample_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
