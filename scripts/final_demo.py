#!/usr/bin/env python3
"""Generate the final demo from the int4 cached checkpoint."""
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


def main():
    t0 = time.time()
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    txt = "哈囉大家，我是 BlueMagpie 的女生版語音。"
    mlx_generate(lite, m, txt, speaker_centroid=c, seed=7, cfg_value=2.8)  # warm
    t0 = time.time()
    a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=7, cfg_value=2.8)
    dt = time.time() - t0
    out = "lite_out/FINAL_int4_ckpt.wav"
    sf.write(out, a.numpy(), lite.sample_rate)
    print(f"generated {dt:.2f}s ({a.shape[-1]/lite.sample_rate:.1f}s audio) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
