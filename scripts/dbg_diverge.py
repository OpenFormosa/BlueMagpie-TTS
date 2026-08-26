#!/usr/bin/env python3
"""Step-by-step divergence finder: single path vs batch path, B=1, same seed.

Dumps max|diff| of pred_feat at every AR step to locate the FIRST step where
the two paths diverge, plus the stop-prob trajectory of both.
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
N = 30  # steps to trace


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    slot = "centroid"
    tt_t, af_t, txm_t, aum_t, spk_t = lite._build_inputs(TEXT, None, None, slot)
    L = int(tt_t.shape[0])
    tt1 = mx.array(tt_t.numpy().astype(np.int32))[None]
    af1 = mx.array(af_t.float().numpy())[None]
    tx1 = mx.array(txm_t.float().numpy())[None]
    au1 = mx.array(aum_t.float().numpy())[None]
    sc1 = mx.array(c.reshape(1, -1).float().numpy())
    sm1 = mx.array(spk_t.float().numpy())[None]

    # ================= SINGLE PATH (inference(), no compile) =================
    fl1 = m.locenc(af1)
    comb = tx1[..., None] * mx.take(m.embed, tt1, axis=0) + au1[..., None] * _proj(fl1, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc1, nw, eps), pw, pb)
    comb = comb + sm1[..., None] * spk_vec[:, None, :]
    bc_s = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    h_s = m.barbet(comb, cache=bc_s)
    tslm_s = m.adapter(h_s)
    enc_s = m.fsq(tslm_s) * au1[..., None] + tslm_s * tx1[..., None]
    felm = _proj(fl1, m.enc_lm)
    rin_s = _proj(mx.concatenate([enc_s, au1[..., None] * felm], axis=-1), m.fusion)
    rc_s = m.ralm.init_cache()
    rs_s = m.ralm(rin_s, cache=rc_s)
    bm._PREFILL_FP32[0] = False
    lm_s, res_s = enc_s[:, -1, :], rs_s[:, -1, :]
    cond_s = af1[:, -1, ...]
    pos_s = L

    # ================= BATCH PATH (valid_lens=[L]) =================
    bc_b = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    h_b = m.barbet(comb, cache=bc_b, valid_lens=[L])
    tslm_b = m.adapter(h_b)
    enc_b = m.fsq(tslm_b) * au1[..., None] + tslm_b * tx1[..., None]
    rin_b = _proj(mx.concatenate([enc_b, au1[..., None] * felm], axis=-1), m.fusion)
    rc_b = m.ralm.init_cache()
    rs_b = m.ralm(rin_b, cache=rc_b)
    bm._PREFILL_FP32[0] = False
    lm_b, res_b = enc_b[:, -1, :], rs_b[:, -1, :]
    cond_b = af1[:, -1, ...]
    pos_arr = np.array([L], dtype=np.int32)

    print(f"prefill: hidden diff={float(mx.max(mx.abs(h_s - h_b))):.2e} "
          f"enc diff={float(mx.max(mx.abs(enc_s - enc_b))):.2e} "
          f"res diff={float(mx.max(mx.abs(rs_s - rs_b))):.2e}")

    t_span = m._t_span(9)
    b = 1
    print("step | pred_feat diff | stop(single,batch)")
    for i in range(N):
        # same noise for both paths this step
        mx.random.seed(1000 + i)
        z_s = mx.random.normal((1, m.feat_dim, m.patch_size))
        mx.random.seed(1000 + i)
        z_b = mx.random.normal((b, m.feat_dim, m.patch_size))

        dh_s = mx.concatenate([_proj(lm_s, m.lm_dit), _proj(res_s, m.res_dit)], axis=-1)
        dh_b = mx.concatenate([_proj(lm_b, m.lm_dit), _proj(res_b, m.res_dit)], axis=-1)
        in_diff = float(mx.max(mx.abs(dh_s - dh_b)))

        pf_s = mx.transpose(solve_euler(m.dit, z_s, t_span, dh_s,
                                        mx.transpose(cond_s, (0, 2, 1)), 2.8), (0, 2, 1))
        pf_b = mx.transpose(solve_euler(m.dit, z_b, t_span, dh_b,
                                        mx.transpose(cond_b, (0, 2, 1)), 2.8), (0, 2, 1))
        d_pf = float(mx.max(mx.abs(pf_s - pf_b)))

        le_s = m.locenc(pf_s[:, None])
        le_b = m.locenc(pf_b[:, None])
        ct_s, cl_s = _proj(le_s, m.enc_tslm), _proj(le_s, m.enc_lm)
        ct_b, cl_b = _proj(le_b, m.enc_tslm), _proj(le_b, m.enc_lm)

        def sp(lm_h):
            sl = bm._lin(bm._silu(bm._lin(lm_h, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
            return float(mx.softmax(sl.astype(mx.float32), axis=-1)[0, 1])

        if i % 5 == 0 or d_pf > 1e-3:
            print(f"{i:4d} | {d_pf:.3e} (dit_in {in_diff:.3e}) | "
                  f"({sp(lm_s):.3f}, {sp(lm_b):.3f})")

        bs_s = m.barbet.step(ct_s[:, 0, :], pos_s, bc_s)
        lm_s = m.fsq(m.adapter(bs_s[:, None, :]))[:, 0, :]
        cr_s = _proj(mx.concatenate([lm_s, cl_s[:, 0, :]], axis=-1), m.fusion)
        res_s = m.ralm.step(cr_s, pos_s, rc_s)

        bs_b = m.barbet.step(ct_b[:, 0, :], mx.array(pos_arr), bc_b)
        lm_b = m.fsq(m.adapter(bs_b[:, None, :]))[:, 0, :]
        cr_b = _proj(mx.concatenate([lm_b, cl_b[:, 0, :]], axis=-1), m.fusion)
        res_b = m.ralm.step(cr_b, mx.array(pos_arr), rc_b)

        cond_s = cond_b = pf_s  # keep conditions identical going forward
        pos_s += 1
        pos_arr += 1

        if in_diff == 0.0 and d_pf == 0.0 and i > 3:
            pass
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
