#!/usr/bin/env python3
"""Trace batch stop probs with the fixed right-pad + per-row pos machinery."""
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
        L = rows_len[i]
        tt[i, :L] = r[0].numpy()
        txm[i, :L] = 1.0
        af[i, :L] = r[1].float().numpy()

    tt_mx, af_mx = mx.array(tt), mx.array(af)
    txm_mx, aum_mx = mx.array(txm), mx.array(aum)
    sc = mx.array(c.reshape(1, -1).float().numpy())
    sm = txm_mx

    # use the real inference_batch internals via a monkeypatched stop dump
    probs_log = []
    orig_softmax = mx.softmax

    import bluemagpie.mlx.model_mlx as mm

    real_inference_batch = m.inference_batch

    # simplest: replicate the loop here using the same code path pieces
    feat_locenc = m.locenc(af_mx)
    fets = _proj(feat_locenc, m.enc_tslm)
    text_embed = mx.take(m.embed, tt_mx, axis=0)
    combined = txm_mx[..., None] * text_embed + aum_mx[..., None] * fets
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc, nw, eps), pw, pb)
    combined = combined + spk_vec[:, None, :] * sm[..., None]
    bcache = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    hidden = m.barbet(combined, cache=bcache, valid_lens=rows_len)
    tslm_h = m.adapter(hidden)
    enc = m.fsq(tslm_h) * aum_mx[..., None] + tslm_h * txm_mx[..., None]
    felm = _proj(feat_locenc, m.enc_lm)
    res_in = _proj(mx.concatenate([enc, aum_mx[..., None] * felm], axis=-1), m.fusion)
    rcache = m.ralm.init_cache()
    res_seq = m.ralm(res_in, cache=rcache)
    bm._PREFILL_FP32[0] = False

    lm_rows = [enc[r, rows_len[r] - 1, :] for r in range(b)]
    lm_hidden = mx.stack(lm_rows, axis=0)
    residual_hidden = mx.stack([res_seq[r, rows_len[r] - 1, :] for r in range(b)], axis=0)
    prefix_cond = af_mx[:, -1, ...]
    pos_arr = np.array(rows_len, dtype=np.int32)

    from bluemagpie.mlx.dit_mlx import solve_euler

    def stop_prob(lm_h):
        sl = bm._lin(bm._silu(bm._lin(lm_h, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
        return np.array(mx.softmax(sl.astype(mx.float32), axis=-1))[:, 1]

    print("step | p_stop per row")
    print(f"   -1 | {np.round(stop_prob(lm_hidden), 3)}  (prefill)")
    for i in range(12):
        pos_mx = mx.array(pos_arr)
        dit_hidden = mx.concatenate([_proj(lm_hidden, m.lm_dit), _proj(residual_hidden, m.res_dit)], axis=-1)
        cond = mx.transpose(prefix_cond, (0, 2, 1))
        mx.random.seed(100 + i)
        z = mx.random.normal((b, m.feat_dim, m.patch_size))
        pred = solve_euler(m.dit, z, m._t_span(9), dit_hidden, cond, 2.8)
        pred_feat = mx.transpose(pred, (0, 2, 1))
        curr_le = m.locenc(pred_feat[:, None])
        curr_tslm = _proj(curr_le, m.enc_tslm)
        curr_lm = _proj(curr_le, m.enc_lm)
        p = stop_prob(lm_hidden)
        print(f"{i:4d} | {np.round(p, 3)}")
        bs = m.barbet.step(curr_tslm[:, 0, :], pos_mx, bcache)
        lm_hidden = m.fsq(m.adapter(bs[:, None, :]))[:, 0, :]
        cr = _proj(mx.concatenate([lm_hidden, curr_lm[:, 0, :]], axis=-1), m.fusion)
        residual_hidden = m.ralm.step(cr, pos_mx, rcache)
        prefix_cond = pred_feat
        pos_arr += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
