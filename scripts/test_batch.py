#!/usr/bin/env python3
"""Validate + benchmark mlx_generate_batch vs serial generation."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    def corr(a, b):
        n = min(len(a), len(b))
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # warm
    mlx_generate(lite, m, TEXTS[0])

    # serial baseline (same texts, one at a time)
    t0 = time.time()
    serial = [mlx_generate(lite, m, t, speaker_centroid=c) for t in TEXTS]
    serial_t = time.time() - t0
    total_audio = sum(len(w.numpy() if hasattr(w, "numpy") else np.asarray(w)) for w in serial) / lite.sample_rate
    print(f"serial x{len(TEXTS)}: {serial_t:.1f}s wall, {total_audio:.1f}s audio "
          f"-> effective RTF {serial_t/total_audio:.2f}")

    # batched
    t0 = time.time()
    batched = mlx_generate_batch(lite, m, TEXTS, speaker_centroid=c)
    batch_t = time.time() - t0
    print(f"batch  x{len(TEXTS)}: {batch_t:.1f}s wall -> speedup {serial_t/batch_t:.2f}x")

    for i, (s_wav, b_wav, txt) in enumerate(zip(serial, batched, TEXTS)):
        s_np = s_wav.numpy() if hasattr(s_wav, "numpy") else np.asarray(s_wav)
        b_np = b_wav.numpy() if hasattr(b_wav, "numpy") else np.asarray(b_wav)
        c_v = corr(s_np, b_np)
        dur_s, dur_b = len(s_np)/lite.sample_rate, len(b_np)/lite.sample_rate
        print(f"  [{i}] len {dur_s:.1f}s vs {dur_b:.1f}s | corr(serial,batch)={c_v:.4f}")
        sf.write(f"lite_out/batch_{i}_b.wav", b_np, lite.sample_rate)

    try:
        from audio_artifact_gate import evaluate_echo_smearing
        for i in range(len(TEXTS)):
            wav, sr = sf.read(f"lite_out/batch_{i}_b.wav")
            r = evaluate_echo_smearing(wav.astype(np.float32), sr)
            print(f"  gate [{i}]: passed={r.passed}")
    except Exception as e:
        print("gate skipped:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
