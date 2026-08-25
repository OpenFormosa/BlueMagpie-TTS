#!/usr/bin/env python3
"""Build + validate the int4 cached checkpoint (one-time)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bluemagpie.mlx.lite import load_model, save_int4_checkpoint
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
    lite, m = load_model(MD, "bluemagpie-lite-fp32.pkl", compute="int4")
    save_int4_checkpoint(m, "bluemagpie-int4.pkl")

    t0 = time.time()
    lite2, m2 = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    print(f"int4-cached load: {time.time()-t0:.1f}s", flush=True)

    txt = "這個 feature 明天上線，記得先跑 regression test。"
    mlx_generate(lite2, m2, txt)  # warm
    ts = []
    for _ in range(3):
        t0 = time.time()
        a = mlx_generate(lite2, m2, txt)
        ts.append(time.time() - t0)
    dur = a.shape[-1] / lite2.sample_rate
    warm = sorted(ts)[1]
    print(f"RTF={warm/dur:.2f} (warm {warm:.2f}s / {dur:.1f}s audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
