#!/usr/bin/env python3
"""Locate first divergent step between inference() and inference_batch() B=1.

Uses noises= injection (supported by inference()) and a monkeypatched
mx.random.normal to force identical z in the batch loop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model

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
    slot = "centroid"

    text_token, audio_feat, text_mask, audio_mask, spk_mask = lite._build_inputs(TEXT, None, None, slot)
    tt = mx.array(text_token.cpu().numpy())[None]
    af = mx.array(audio_feat.float().numpy())[None]
    txm = mx.array(text_mask.float().numpy())[None]
    aum = mx.array(audio_mask.float().numpy())[None]
    sm = mx.array(spk_mask.float().numpy())[None]
    sc = mx.array(c.reshape(1, -1).float().numpy())
    L = int(text_token.shape[0])

    # pre-draw 30 noises; feed BOTH paths the same sequence
    mx.random.seed(7)
    noises = [mx.random.normal((1, m.feat_dim, m.patch_size)) for _ in range(30)]

    lat_s = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                        min_len=2, max_len=2000, inference_timesteps=9, cfg_value=2.8,
                        noises=noises, compile=False,
                        expected_steps=int(len(TEXT) * 1.5) + 6,
                        stop_prob_threshold=0.65, stop_consecutive=2)

    # monkeypatch normal() inside model_mlx to replay the same list for batch
    import bluemagpie.mlx.model_mlx as mm
    orig_normal = mx.random.normal
    counter = {"i": 0}

    def fake_normal(shape, *a, **k):
        i = counter["i"]
        counter["i"] += 1
        z = noises[i % len(noises)]
        if shape[0] != 1:
            z = mx.concatenate([z] * shape[0], axis=0)   # replicate across batch rows
        return z

    mx.random.normal = fake_normal
    try:
        lat_b = m.inference_batch(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                                  valid_lens=[L], max_len=2000, inference_timesteps=9,
                                  cfg_value=2.8, expected_steps=int(len(TEXT) * 1.5) + 6,
                                  stop_prob_threshold=0.65, stop_consecutive=2)[0]
    finally:
        mx.random.normal = orig_normal

    ns, nb = lat_s.shape[0], lat_b.shape[0]
    print(f"steps: single={ns} batch={nb}")
    s_np, b_np = np.array(lat_s), np.array(lat_b)
    k = min(ns, nb)
    diffs = np.abs(s_np[:k] - b_np[:k]).reshape(k, -1).max(axis=1)
    first = next((i for i, d in enumerate(diffs) if d > 1e-5), None)
    print("per-patch max|diff|:", np.round(diffs[:12], 6))
    print(f"first divergence > 1e-5 at patch: {first} ({diffs[first]:.3e})" if first is not None
          else "identical within 1e-5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
