#!/usr/bin/env python3
"""B=1 parity: batch path vs single path, dump first-step stop probs.

If the prefill machinery is correct, the batch path at B=1 must produce the
same step-0 stop probability and the same first DiT output as inference().
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

TEXT = "今天天氣真好，我們一起去公園散步吧。"


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    slot = "centroid"
    tt_t, af_t, txm_t, aum_t, spk_t = lite._build_inputs(TEXT, None, None, slot)
    L = int(tt_t.shape[0])
    print("seq len:", L)
    tt1 = mx.array(tt_t.numpy().astype(np.int32))[None]
    af1 = mx.array(af_t.float().numpy())[None]
    tx1 = mx.array(txm_t.float().numpy())[None]
    au1 = mx.array(aum_t.float().numpy())[None]
    sc1 = mx.array(c.reshape(1, -1).float().numpy())
    sm1 = mx.array(spk_t.float().numpy())[None]

    # ---- single path (inference()) prefill ----
    fl1 = m.locenc(af1)
    comb1 = tx1[..., None] * mx.take(m.embed, tt1, axis=0) + au1[..., None] * _proj(fl1, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc1, nw, eps), pw, pb)
    comb1 = comb1 + sm1[..., None] * spk_vec[:, None, :]
    bc1 = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    h1 = m.barbet(comb1, cache=bc1)                     # no valid_lens -> normal path
    tslm1 = m.adapter(h1)
    enc1 = m.fsq(tslm1) * au1[..., None] + tslm1 * tx1[..., None]
    felm1 = _proj(fl1, m.enc_lm)
    rin1 = _proj(mx.concatenate([enc1, au1[..., None] * felm1], axis=-1), m.fusion)
    rc1 = m.ralm.init_cache()
    rs1 = m.ralm(rin1, cache=rc1)
    bm._PREFILL_FP32[0] = False
    lm1 = enc1[:, -1, :]
    res1 = rs1[:, -1, :]

    def sp(lm_h):
        sl = bm._lin(bm._silu(bm._lin(lm_h, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
        return np.array(mx.softmax(sl.astype(mx.float32), axis=-1))[:, 1]

    print("single-path step-0 p_stop:", sp(lm1))

    # ---- batch path with B=1, valid_lens=[L] ----
    bcache_b = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    h_b = m.barbet(comb1, cache=bcache_b, valid_lens=[L])       # same inputs!
    tslm_b = m.adapter(h_b)
    enc_b = m.fsq(tslm_b) * au1[..., None] + tslm_b * tx1[..., None]
    rin_b = _proj(mx.concatenate([enc_b, au1[..., None] * felm1], axis=-1), m.fusion)
    rcache_b = m.ralm.init_cache()
    rs_b = m.ralm(rin_b, cache=rcache_b)
    bm._PREFILL_FP32[0] = False
    lm_b = enc_b[:, -1, :]                                       # B=1: last pos == len-1
    res_b = rs_b[:, -1, :]

    print("batch(B=1) step-0 p_stop:", sp(lm_b))
    print("prefill hidden diff (single vs batch-machinery):",
          float(mx.max(mx.abs(h1 - h_b))))

    # first AR step: same seed -> same z; compare lm_hidden after barbet.step
    mx.random.seed(100)
    dit_hidden_1 = mx.concatenate([_proj(lm1, m.lm_dit), _proj(res1, m.res_dit)], axis=-1)
    cond1 = mx.transpose(af1[:, -1, ...], (0, 2, 1))
    z1 = mx.random.normal((1, m.feat_dim, m.patch_size))
    pred1 = solve_euler(m.dit, z1, m._t_span(9), dit_hidden_1, cond1, 2.8)
    pf1 = mx.transpose(pred1, (0, 2, 1))
    le1 = m.locenc(pf1[:, None])
    ct1 = _proj(le1, m.enc_tslm)
    cl1 = _proj(le1, m.enc_lm)
    bs_single = m.barbet.step(ct1[:, 0, :], L, bc1)
    lmv_single = m.fsq(m.adapter(bs_single[:, None, :]))[:, 0, :]
    print("single next p_stop:", sp(lmv_single))

    mx.random.seed(100)
    dit_hidden_b = mx.concatenate([_proj(lm_b, m.lm_dit), _proj(res_b, m.res_dit)], axis=-1)
    pred_b = solve_euler(m.dit, z1, m._t_span(9), dit_hidden_b,
                         mx.transpose(af1[:, -1, ...], (0, 2, 1)), 2.8)
    pf_b = mx.transpose(pred_b, (0, 2, 1))
    le_b = m.locenc(pf_b[:, None])
    ct_b = _proj(le_b, m.enc_tslm)
    cl_b = _proj(le_b, m.enc_lm)
    bs_batch = m.barbet.step(ct_b[:, 0, :], mx.array([L]), bcache_b)
    lmv_batch = m.fsq(m.adapter(bs_batch[:, None, :]))[:, 0, :]
    print("batch  next p_stop:", sp(lmv_batch))
    print("barbet.step diff (int-pos vs arr-pos):",
          float(mx.max(mx.abs(bs_single - bs_batch))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
