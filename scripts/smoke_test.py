#!/usr/bin/env python3
"""End-to-end smoke test of the final pipeline."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import mlx_generate, mlx_generate_streaming

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
    print(f"[1] load int4-cached: {time.time()-t0:.1f}s")

    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    txt = "煙霧測試，確認整條管線正常運作。"

    # non-streaming
    mlx_generate(lite, m, txt)  # warm
    t0 = time.time()
    a = mlx_generate(lite, m, txt, speaker_centroid=c, seed=7, cfg_value=2.8)
    dt = time.time() - t0
    dur = a.shape[-1] / lite.sample_rate
    print(f"[2] generate: {dt:.2f}s / {dur:.1f}s audio -> RTF {dt/dur:.2f}")
    sf.write("lite_out/smoke_full.wav", a.numpy(), lite.sample_rate)

    # streaming
    t0 = time.time()
    chunks, first_at = [], None
    for ch in mlx_generate_streaming(lite, m, txt, chunk_patches=4,
                                     speaker_centroid=c, seed=7, cfg_value=2.8):
        if first_at is None:
            first_at = time.time() - t0
        chunks.append(ch)
    stream = np.concatenate([np.asarray(ch) for ch in chunks])
    n = min(len(stream), len(a))
    cc = float(np.corrcoef(stream[:n], np.asarray(a)[:n])[0, 1])
    print(f"[3] streaming: TTFB {first_at:.2f}s, {len(chunks)} chunks, corr vs full {cc:.4f}")

    # echo gate on both outputs
    from audio_artifact_gate import evaluate_echo_smearing
    for name, wav in (("full", np.asarray(a)), ("stream", stream)):
        r = evaluate_echo_smearing(wav.astype(np.float32), lite.sample_rate)
        verdict = getattr(r, "verdict", r)
        score = getattr(r, "delayed_score", "?")
        print(f"[4] gate {name}: {verdict} (score={score})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
