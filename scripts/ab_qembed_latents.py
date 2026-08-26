#!/usr/bin/env python3
"""A/B int4 embedding with bit-exact noise: compare latents, not audio.

Injects identical z into both runs so the ONLY difference is the embedding
quantization error. Reports latent max|diff| and whether stop step matched.
"""
import os
import gc
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

from bluemagpie.mlx.lite import load_model, reactivate_int4, _build_lite_torch, _load_config

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


def run_with_noise(m, noises):
    slot = "centroid"
    tt_t, af_t, txm_t, aum_t, spk_t = m._lite_ref if hasattr(m, "_lite_ref") else (None,) * 5
    return None


def main():
    from bluemagpie.mlx.lite import load_model as lm
    lite, m = lm(MD, "bluemagpie-int4.pkl", compute="int4-cached")
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

    # A: quantized embed (current state after load_model)
    lat_q = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc, noises=noises, **kw)

    # B: restore fp32 embed (rebuild from packed is impossible; reload without qembed)
    import bluemagpie.mlx.barbet_mlx as bm
    bm._QUANTIZED_EMBED.clear()
    # the fp32 table was replaced in-place; recover by loading fresh copy
    del lite, m
    gc.collect()
    import pickle
    with open("bluemagpie-int4.pkl", "rb") as f:
        m2 = pickle.load(f)
    reactivate_int4(m2)
    # keep fp32 embed (do NOT call quantize_embed_inplace)
    lat_f = m2.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc, noises=noises, **kw)

    s_np, b_np = np.array(lat_q), np.array(lat_f)
    k = min(s_np.shape[0], b_np.shape[0])
    diffs = np.abs(s_np[:k] - b_np[:k]).reshape(k, -1).max(axis=1)
    print(f"steps: qembed={s_np.shape[0]} fp32embed={b_np.shape[0]}")
    print("per-patch max|diff|:", np.round(diffs[:12], 6))
    print(f"overall max|diff|: {diffs.max():.3e}")
    print(f"stop step identical: {s_np.shape[0] == b_np.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
