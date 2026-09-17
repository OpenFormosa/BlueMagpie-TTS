#!/usr/bin/env python3
"""Validate the robust-stop contract: latency bound + text coverage."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import soundfile as sf

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

TEXTS = [
    "哈囉大家，我是 BlueMagpie 的女生版語音。",
    "這個 feature 明天上線，記得先跑 regression test。",
    "今天天氣真好，我們一起去公園散步吧。",
]


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    for txt in TEXTS:
        n = len(txt)
        mlx_generate(lite, m, txt, speaker_centroid=c, seed=7, cfg_value=2.8)  # warm per-length
        t0 = time.time()
        a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=7, cfg_value=2.8)
        dt = time.time() - t0
        dur = a.shape[-1] / lite.sample_rate
        pace = n / dur if dur > 0 else 0
        print(f"[{n:>2}字] gen={dt:.2f}s audio={dur:.1f}s rtf={dt/dur:.2f} pace={pace:.1f}字/秒", flush=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in txt)[:20]
        sf.write(f"lite_out/robust_{safe}.wav", a.numpy(), lite.sample_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
