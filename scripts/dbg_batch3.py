#!/usr/bin/env python3
"""Instrument inference_batch: dump per-row stop probs for the first steps."""
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

TEXTS = [
    "今天天氣真好，我們一起去公園散步吧。",
    "這個 feature 明天上線，記得先跑 regression test。",
    "會議改到下午三點，請大家準時參加。",
    "晚餐想吃什麼？附近新開了一間拉麵店。",
]


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    slot = "centroid"
    rows = [lite._build_inputs(t, None, None, slot) for t in TEXTS]
    rows_len = [r[0].shape[0] for r in rows]
    max_s = max(rows_len)
    b = len(TEXTS)

    tt = np.zeros((b, max_s), dtype=np.int32)
    txm = np.zeros((b, max_s), dtype=np.float32)
    aum = np.zeros((b, max_s), dtype=np.float32)
    af = np.zeros((b, max_s, *rows[0][1].shape[1:]), dtype=np.float32)
    for i, r in enumerate(rows):
        off = max_s - rows_len[i]
        tt[i, off:] = r[0].numpy()
        txm[i, off:] = 1.0
        af[i, off:] = r[1].float().numpy()

    tt_mx, af_mx = mx.array(tt), mx.array(af)
    txm_mx, aum_mx = mx.array(txm), mx.array(aum)
    sc = mx.array(c.reshape(1, -1).float().numpy())
    sm = mx.ones((b, max_s))

    # batched prefill (mirror inference_batch)
    feat_locenc = m.locenc(af_mx)
    text_embed = mx.take(m.embed, tt_mx, axis=0)
    combined = txm_mx[..., None] * text_embed + aum_mx[..., None] * _proj(feat_locenc, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc, nw, eps), pw, pb)
    combined = combined + spk_vec[:, None, :]
    bcache = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    hidden = m.barbet(combined, cache=bcache)
    tslm_h = m.adapter(hidden)
    enc = m.fsq(tslm_h) * aum_mx[..., None] + tslm_h * txm_mx[..., None]
    lm_hidden = enc[:, -1, :]

    stop_logits = bm._lin(bm._silu(bm._lin(lm_hidden, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
    probs = mx.softmax(stop_logits.astype(mx.float32), axis=-1)
    p_stop = np.array(probs)[:, 1]
    print("step-0 p_stop per row:", np.round(p_stop, 4))
    print("(serial single-path reference would be ~0 at step 0)")

    # also check the serial path's own first-step stop prob for comparison:
    r0 = rows[0]
    tt1 = mx.array(r0[0].numpy().astype(np.int32))[None]
    fl1 = m.locenc(mx.array(r0[1].float().numpy())[None])
    te1 = mx.take(m.embed, tt1, axis=0)
    tx1 = mx.array(r0[2].float().numpy())[None]
    au1 = mx.array(r0[3].float().numpy())[None]
    comb1 = tx1[..., None] * te1 + au1[..., None] * _proj(fl1, m.enc_tslm)
    sm1 = mx.ones((1, r0[0].shape[0]))
    comb1 = comb1 + sm1[..., None] * spk_vec[:, None, :]
    bc1 = m.barbet.init_cache()
    h1 = m.barbet(comb1, cache=bc1)
    e1 = m.fsq(m.adapter(h1)) * au1[..., None] + m.adapter(h1) * tx1[..., None]
    sl1 = bm._lin(bm._silu(bm._lin(e1[:, -1, :], m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
    print("single-path row0 p_stop:", float(mx.softmax(sl1.astype(mx.float32), axis=-1)[0, 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
