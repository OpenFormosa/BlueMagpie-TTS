#!/usr/bin/env python3
"""A/B the int4 embedding: same seed, embed fp32 vs int4 (two loads)."""
import os
import gc
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx import barbet_mlx as bm
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

TEXT = "今天天氣真好，我們一起去公園散步吧。"


def gen(lite, m, seed):
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    mlx_generate(lite, m, TEXT, speaker_centroid=c)  # warm
    w = mlx_generate(lite, m, TEXT, speaker_centroid=c, seed=seed)
    return w.numpy() if hasattr(w, "numpy") else np.asarray(w)


def main():
    # A: embedding quantized
    lite_a, m_a = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    sr = lite_a.sample_rate
    a = gen(lite_a, m_a, 7)
    sf.write("lite_out/ab_qembed.wav", a, sr)
    del lite_a, m_a
    gc.collect()

    # B: same checkpoint but embedding kept fp32 (bypass quantize_embed_inplace)
    import pickle
    with open("bluemagpie-int4.pkl", "rb") as f:
        m_b = pickle.load(f)
    from bluemagpie.mlx.lite import reactivate_int4
    reactivate_int4(m_b)  # skip quantize_embed_inplace on purpose
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MD, "tokenizer.json"))
    from bluemagpie.mlx.lite import _build_lite_torch, _load_config
    config = _load_config(MD)
    from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2
    from bluemagpie.mlx.audiovae_mlx import AudioVAEMLX
    vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
    st = torch.load(os.path.join(MD, "audiovae.pth"), map_location="cpu", weights_only=True)
    vae.load_state_dict(st.get("state_dict", st))
    m_b.vae = AudioVAEMLX(vae)
    lite_b = _build_lite_torch(MD, config, tok)

    b = gen(lite_b, m_b, 7)
    sf.write("lite_out/ab_fp32embed.wav", b, lite_b.sample_rate)

    n = min(len(a), len(b))
    aa, bb = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    cc = float(np.corrcoef(aa, bb)[0, 1])
    print(f"qembed vs fp32-embed (same seed): corr={cc:.4f} "
          f"| len {len(a)/sr:.1f}s vs {len(b)/lite_b.sample_rate:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
