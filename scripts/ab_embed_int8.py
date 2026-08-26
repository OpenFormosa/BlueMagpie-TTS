#!/usr/bin/env python3
"""int8 embedding A/B: same-noise latent parity vs fp32 embed.

int8 error is 16x smaller than int4 (0.0017 vs 0.0277) — check whether it
keeps latents close enough to be a safe RAM saver.
"""
import os
import gc
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model, reactivate_int4
from bluemagpie.mlx import barbet_mlx as bm

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
    kw = dict(min_len=2, max_len=2000, inference_timesteps=9, cfg_value=2.8,
              expected_steps=int(len(TEXT) * 1.5) + 6,
              stop_prob_threshold=0.65, stop_consecutive=2)

    noises = []
    mx.random.seed(7)
    for _ in range(40):
        noises.append(mx.random.normal((1, m.feat_dim, m.patch_size)))

    # baseline: fp32 embed (current state)
    lat_f = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc, noises=noises, **kw)

    # now quantize embed to int8 and rerun with the SAME noise
    w = m.embed
    wq, s8, b8 = mx.quantize(w, group_size=64, bits=8)
    bm.register_quantized_embed(wq, s8, b8, group_size=64, bits=8)
    m.embed = wq
    lat_8 = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc, noises=noises, **kw)

    f_np, e_np = np.array(lat_f), np.array(lat_8)
    k = min(f_np.shape[0], e_np.shape[0])
    diffs = np.abs(f_np[:k] - e_np[:k]).reshape(k, -1).max(axis=1)
    print(f"steps: fp32={f_np.shape[0]} int8embed={e_np.shape[0]}")
    print("per-patch max|diff|:", np.round(diffs[:12], 6))
    print(f"overall max|diff|: {diffs.max():.3e} | stop step identical: {f_np.shape[0] == e_np.shape[0]}")

    # restore fp32 embed for cleanliness
    bm._QUANTIZED_EMBED.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())
