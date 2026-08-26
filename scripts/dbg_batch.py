#!/usr/bin/env python3
"""B=1 sanity: compare batch-path internals vs single path step by step."""
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
    tt_t, af_t, txm_t, aum_t, spk_t = lite._build_inputs(TEXT, None, None, slot)
    print("seq len:", tt_t.shape[0], "| spk_mask nonzero:", int(spk_t.sum().item()))

    # exactly as inference() builds them (single path)
    tt1 = mx.array(tt_t.cpu().numpy().astype(np.int32))[None]
    af1 = mx.array(af_t.float().numpy())[None]
    txm1 = mx.array(txm_t.float().numpy())[None]
    aum1 = mx.array(aum_t.float().numpy())[None]
    sc1 = mx.array(c.reshape(1, -1).float().numpy())
    sm1 = mx.array(spk_t.float().numpy())[None]

    feat_locenc = m.locenc(af1)
    text_embed = mx.take(m.embed, tt1, axis=0)
    combined = txm1[..., None] * text_embed + aum1[..., None] * _proj(feat_locenc, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc1, nw, eps), pw, pb)
    print("spk_vec norm:", float(mx.abs(spk_vec).sum()))
    combined = combined + sm1[..., None] * spk_vec[:, None, :]
    bcache = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    hidden1 = m.barbet(combined, cache=bcache)
    tslm_h1 = m.adapter(hidden1)
    enc1 = m.fsq(tslm_h1) * aum1[..., None] + tslm_h1 * txm1[..., None]

    # now the batch path with B=1 (identical inputs, same shapes)
    feat_locenc_b = m.locenc(af1)                                # same call
    text_embed_b = mx.take(m.embed, tt1, axis=0)
    combined_b = txm1[..., None] * text_embed_b + aum1[..., None] * _proj(feat_locenc_b, m.enc_tslm)
    combined_b = combined_b + sm1[..., None] * spk_vec[:, None, :]
    bcache_b = m.barbet.init_cache()
    hidden_b = m.barbet(combined_b, cache=bcache_b)
    tslm_hb = m.adapter(hidden_b)
    enc_b = m.fsq(tslm_hb) * aum1[..., None] + tslm_hb * txm1[..., None]

    print("prefill hidden diff:", float(mx.max(mx.abs(hidden1 - hidden_b))))
    print("enc diff:", float(mx.max(mx.abs(enc1 - enc_b))))

    # stop head at the last prefill position
    stop_logits = bm._lin(bm._silu(bm._lin(enc1[:, -1, :], m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
    probs = mx.softmax(stop_logits.astype(mx.float32), axis=-1)
    print("stop prob from prefill-last:", np.array(probs))

    # check: in inference(), lm_hidden for the FIRST AR step comes from
    # enc_outputs[:, -1, :] — but the loop's first stop_logits uses that same
    # lm_hidden. Verify what the single path actually produces on step 0 by
    # running one real generation and dumping is overkill; instead verify the
    # batched locenc call shape equivalence used inside the AR loop:
    z_fake = mx.random.normal((2, m.feat_dim, m.patch_size))
    le_single_style = m.locenc(z_fake[0][None][None])            # [1,1,p,d]->[1,1,h]
    le_batch_style = m.locenc(z_fake[:, None])                   # [B,1,p,d]->[B,1,h]
    print("locenc B=1 vs sliced B=2 max|diff|:",
          float(mx.max(mx.abs(le_batch_style[0] - le_single_style))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
