#!/usr/bin/env python3
"""B=1 batch vs single: dump per-step p_stop for BOTH public paths (same seed)."""
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
    exp = int(len(TEXT) * 1.5) + 6

    # instrument both loops by wrapping np.array to sniff probs? Simpler:
    # re-run with tiny max_len and record latent count at various thresholds.
    # Instead: run each with max_len=2000 and count patches; then rerun single
    # with the BATCH's patch count as hard cap to compare latents directly.
    mx.random.seed(7)
    lat_b = m.inference_batch(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                              valid_lens=[L], max_len=2000, inference_timesteps=9,
                              cfg_value=2.8, expected_steps=exp,
                              stop_prob_threshold=0.65, stop_consecutive=2)[0]
    nb = lat_b.shape[0]

    mx.random.seed(7)
    lat_s = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                        min_len=2, max_len=2000, inference_timesteps=9, cfg_value=2.8,
                        expected_steps=exp, stop_prob_threshold=0.65, stop_consecutive=2)
    ns = lat_s.shape[0]
    print(f"single produced {ns} patches; batch(B=1) produced {nb} patches")

    # force single to run exactly nb steps by disabling robust stop and capping
    mx.random.seed(7)
    lat_s_cap = m.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc,
                            min_len=2, max_len=max(nb, 5), inference_timesteps=9,
                            cfg_value=2.8, expected_steps=0,
                            stop_prob_threshold=0.65, stop_consecutive=10_000)
    nsc = lat_s_cap.shape[0]
    print(f"single uncapped-stop ran {nsc} patches")
    k = min(nb, nsc)
    d = float(np.abs(np.array(lat_s_cap)[:k] - np.array(lat_b)[:k]).max())
    print(f"latent max|diff| over {k} common patches: {d:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
