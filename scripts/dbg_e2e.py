#!/usr/bin/env python3
"""End-to-end B=1 parity with FULL step dump of the actual public APIs."""
import os
import sys
import time

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

TEXT = "今天天氣真好，我們一起去公園散步吧。"


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    mlx_generate(lite, m, TEXT)  # warm

    # serial via public API (compiled sampler path)
    a = mlx_generate(lite, m, TEXT, speaker_centroid=c, seed=7)
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)

    t0 = time.time()
    b1 = mlx_generate_batch(lite, m, [TEXT], speaker_centroid=c, seed=7)[0]
    bt = time.time() - t0
    b_np = b1.numpy() if hasattr(b1, "numpy") else np.asarray(b1)

    n = min(len(a_np), len(b_np))
    cc = float(np.corrcoef(a_np[:n] - a_np[:n].mean(), b_np[:n] - b_np[:n].mean())[0, 1])
    print(f"serial {len(a_np)/lite.sample_rate:.2f}s vs batchB=1 {len(b_np)/lite.sample_rate:.2f}s "
          f"({bt:.1f}s) | corr={cc:.4f}")

    sf.write("lite_out/par_serial.wav", a_np, lite.sample_rate)
    sf.write("lite_out/par_b1.wav", b_np, lite.sample_rate)

    try:
        from audio_artifact_gate import evaluate_echo_smearing
        for name in ("par_serial", "par_b1"):
            wav, sr = sf.read(f"lite_out/{name}.wav")
            r = evaluate_echo_smearing(wav.astype(np.float32), sr)
            print(f"gate {name}: passed={r.passed}")
    except Exception as e:
        print("gate skipped:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
