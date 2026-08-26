#!/usr/bin/env python3
"""Trace serial single-path stop probs for the same texts (reference)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import numpy as np
import torch

import bluemagpie.mlx.barbet_mlx as bm
from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.model_mlx import _proj
from bluemagpie.mlx.dit_mlx import solve_euler

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


def trace_one(m, lite, text, c):
    slot = "centroid"
    tt_t, af_t, txm_t, aum_t, _ = lite._build_inputs(text, None, None, slot)
    tt1 = mx.array(tt_t.numpy().astype(np.int32))[None]
    af1 = mx.array(af_t.float().numpy())[None]
    tx1 = mx.array(txm_t.float().numpy())[None]
    au1 = mx.array(aum_t.float().numpy())[None]
    sc1 = mx.array(c.reshape(1, -1).float().numpy())

    feat_locenc = m.locenc(af1)
    combined = tx1[..., None] * mx.take(m.embed, tt1, axis=0) + au1[..., None] * _proj(feat_locenc, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc1, nw, eps), pw, pb)
    combined = combined + spk_vec[:, None, :]
    bcache = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    hidden = m.barbet(combined, cache=bcache)
    tslm_h = m.adapter(hidden)
    enc = m.fsq(tslm_h) * au1[..., None] + tslm_h * tx1[..., None]
    lm_hidden = enc[:, -1, :]
    res_in = _proj(mx.concatenate([enc, au1[..., None] * _proj(feat_locenc, m.enc_lm)], axis=-1), m.fusion)
    rcache = m.ralm.init_cache()
    res_seq = m.ralm(res_in, cache=rcache)
    bm._PREFILL_FP32[0] = False
    residual_hidden = res_seq[:, -1, :]
    prefix_cond = af1[:, -1, ...]
    pos = int(tt1.shape[1])
    t_span = m._t_span(9)

    def stop_prob(lm_h):
        sl = bm._lin(bm._silu(bm._lin(lm_h, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
        return float(mx.softmax(sl.astype(mx.float32), axis=-1)[0, 1])

    out = []
    for i in range(14):
        dit_hidden = mx.concatenate([_proj(lm_hidden, m.lm_dit), _proj(residual_hidden, m.res_dit)], axis=-1)
        cond = mx.transpose(prefix_cond, (0, 2, 1))
        mx.random.seed(100 + i)
        z = mx.random.normal((1, m.feat_dim, m.patch_size))
        pred = solve_euler(m.dit, z, t_span, dit_hidden, cond, 2.8)
        pred_feat = mx.transpose(pred, (0, 2, 1))
        curr_le = m.locenc(pred_feat[:, None])
        curr_tslm = _proj(curr_le, m.enc_tslm)
        curr_lm = _proj(curr_le, m.enc_lm)
        out.append(stop_prob(lm_hidden))
        bs = m.barbet.step(curr_tslm[:, 0, :], pos, bcache)
        lm_hidden = m.fsq(m.adapter(bs[:, None, :]))[:, 0, :]
        cr = _proj(mx.concatenate([lm_hidden, curr_lm[:, 0, :]], axis=-1), m.fusion)
        residual_hidden = m.ralm.step(cr, pos, rcache)
        prefix_cond = pred_feat
        pos += 1
    return out


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]
    print("SERIAL reference (same seeds per step):")
    for t in TEXTS:
        p = trace_one(m, lite, t, c)
        print(f"  {t[:10]:<12} {np.round(p, 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
