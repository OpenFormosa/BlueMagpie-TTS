#!/usr/bin/env python3
"""Compare LATENTS from the two public APIs (B=1, same seed).

If latents match but audio differs -> VAE decode issue.
If latents differ -> the AR loops diverge despite dbg_diverge showing 0 diff,
meaning the divergence comes from input assembly (masks/feat) differences
between mlx_generate and mlx_generate_batch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

import bluemagpie.mlx.barbet_mlx as bm
from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.model_mlx import _proj

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

    # ---- inputs EXACTLY as mlx_generate builds them ----
    text_token, audio_feat, text_mask, audio_mask, spk_mask = lite._build_inputs(TEXT, None, None, slot)
    tt = mx.array(text_token.cpu().numpy())[None]
    af = mx.array(audio_feat.float().numpy())[None]
    txm = mx.array(text_mask.float().numpy())[None]
    aum = mx.array(audio_mask.float().numpy())[None]
    sm = mx.array(spk_mask.float().numpy())[None]
    sc = mx.array(c.reshape(1, -1).float().numpy())

    # ---- run inference() (single) ----
    mx.random.seed(7)
    lat_single = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                             min_len=2, max_len=2000, inference_timesteps=9, cfg_value=2.8,
                             expected_steps=int(len(TEXT) * 1.5) + 6,
                             stop_prob_threshold=0.65, stop_consecutive=2)

    # ---- run inference_batch() (batch) with IDENTICAL tensors ----
    mx.random.seed(7)
    lats_batch = m.inference_batch(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                                   valid_lens=[int(text_token.shape[0])],
                                   max_len=2000, inference_timesteps=9, cfg_value=2.8,
                                   expected_steps=int(len(TEXT) * 1.5) + 6,
                                   stop_prob_threshold=0.65, stop_consecutive=2)

    s_np = np.array(lat_single)
    b_np = np.array(lats_batch[0])
    print(f"latents: single {s_np.shape} vs batch {b_np.shape}")
    n = min(s_np.shape[0], b_np.shape[0])
    d = float(np.abs(s_np[:n] - b_np[:n]).max())
    print(f"latent max|diff| over {n} common patches: {d:.3e}")

    # per-step first-divergence scan
    for i in range(n):
        di = float(np.abs(s_np[i] - b_np[i]).max())
        if di > 1e-4:
            print(f"first divergent latent at patch {i}: {di:.3e}")
            break
    else:
        print("all common patches identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
