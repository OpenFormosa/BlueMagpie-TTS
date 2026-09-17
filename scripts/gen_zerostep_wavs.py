#!/usr/bin/env python3
"""Generate A/B wavs for cfg-zero-star zero_steps variants + echo gate.

zero_init_steps=1 is the current default. The sweep showed 2-4 skip DiT calls
(25-50% less FLOPs) but change latents substantially -- these wavs let the
user judge whether the quality tradeoff is acceptable for speed.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch

import bluemagpie.mlx.barbet_mlx as bm
from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.model_mlx import _proj
from bluemagpie.mlx.dit_mlx import solve_euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_zerosteps import ar_loop, TEXT, MD


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    sr = lite.sample_rate

    mx.random.seed(99)
    noises = [mx.random.normal((1, m.feat_dim, m.patch_size)) for _ in range(40)]

    try:
        from audio_artifact_gate import evaluate_echo_smearing
    except Exception:
        evaluate_echo_smearing = None

    for zs in (2, 3):
        lat, wall = ar_loop(m, lite, c, zs, noises)
        audio = m.decode_latents(lat)
        mx.eval(audio)
        wav = np.array(audio).squeeze()
        path = f"lite_out/zerostep_{zs}.wav"
        sf.write(path, wav, sr)
        dur = len(wav) / sr
        note = ""
        if evaluate_echo_smearing is not None:
            r = evaluate_echo_smearing(wav.astype(np.float32), sr)
            note = f"gate={'PASS' if r.passed else 'FAIL'}"
        print(f"zero_steps={zs}: {dur:.1f}s wall={wall:.1f}s RTF={wall/dur:.2f} {note} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
