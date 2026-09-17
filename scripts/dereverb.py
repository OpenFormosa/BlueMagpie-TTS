#!/usr/bin/env python3
"""Reduce perceived echo: multi-seed candidate selection + inverse-comb
cancellation at the detected delay.

1. Generate several seeded candidates, score each with the official
   audio_artifact_gate cepstral detector, keep the cleanest.
2. Estimate the fixed delayed-copy gain at the detected delay via least
   squares over active frames, subtract it (gentle, gain-limited), re-check
   the gate so we never make it worse.

Usage:
    python scripts/dereverb.py [TEXT] [SPEAKER] [N_SEEDS]
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

MODEL_DIR = os.path.join(
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_artifact_gate import evaluate_echo_smearing  # noqa: E402


def gate_score(wav, sr):
    d = evaluate_echo_smearing(wav, sr)
    return max(d.delayed_copy_score, d.smearing_score), d


def cancel_delayed_copy(x, sr, delay_ms, max_gain=0.5):
    """Subtract g * x[n - D] with the LS-optimal g over active frames."""
    D = int(round(delay_ms / 1000.0 * sr))
    if D <= 0 or D >= len(x) // 2:
        return x, 0.0
    win = int(0.03 * sr)
    hop = win // 2
    num = den = 1e-12
    # accumulate correlations over active frames only
    rms_all = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))
    thr = 0.15 * rms_all.max()
    idx = np.where(rms_all > thr)[0]
    if len(idx) == 0:
        return x, 0.0
    lo, hi = max(idx[0], D), min(idx[-1], len(x))
    xa = x[lo - D:hi - D]
    xb = x[lo:hi]
    mask = rms_all[lo:hi] > thr
    num = float(np.dot(xb[mask], xa[mask]))
    den = float(np.dot(xa[mask], xa[mask])) + 1e-12
    g = float(np.clip(num / den, 0.0, max_gain))
    y = x.copy()
    y[D:] -= g * x[:-D]
    return y, g


def main() -> int:
    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx import mlx_generate

    text = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else \
        "哈囉大家，我是 BlueMagpie 的女生版語音。"
    speaker = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "female_voice"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    lite, m = load_model(MODEL_DIR, "bluemagpie-lite-fp32.pkl")
    table = torch.load(os.path.join(MODEL_DIR, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index(speaker)]

    cands = []
    for seed in range(1, n_seeds + 1):
        a = mlx_generate(lite, m, text, speaker_centroid=c, seed=seed)
        wav = a.numpy().astype(np.float64)
        s, d = gate_score(wav, lite.sample_rate)
        print(f"seed={seed}: score={s:.3f} delayed={d.delayed_copy_score:.3f} "
              f"smear={d.smearing_score:.3f} delay={d.dominant_delay_ms:.1f}ms", flush=True)
        cands.append((s, seed, wav, d))

    cands.sort(key=lambda t: t[0])
    s_best, seed_best, wav_best, d_best = cands[0]
    print(f"\nbest: seed={seed_best} score={s_best:.3f}")

    out = f"lite_out/clean_{speaker}_seed{seed_best}.wav"
    sf.write(out, wav_best.astype(np.float32), lite.sample_rate)
    print("saved:", out)

    # inverse-comb cancellation at the detected delay
    y, g = cancel_delayed_copy(wav_best.copy(), lite.sample_rate, d_best.dominant_delay_ms)
    s2, d2 = gate_score(y, lite.sample_rate)
    print(f"after cancellation(g={g:.3f}): score {s_best:.3f} -> {s2:.3f} "
          f"(delayed {d_best.delayed_copy_score:.3f} -> {d2.delayed_copy_score:.3f})")
    if s2 <= s_best:
        out2 = f"lite_out/clean_{speaker}_seed{seed_best}_decancel.wav"
        sf.write(out2, y.astype(np.float32), lite.sample_rate)
        print("saved:", out2)
    else:
        print("cancellation made the gate worse; keeping the plain best")
    return 0


if __name__ == "__main__":
    sys.exit(main())