#!/usr/bin/env python3
"""Sweep cfg-zero-star zero_init_steps: 1 (current) vs 2..4 of 9.

Each extra skipped step saves ~11% of DiT FLOPs. Measure per-step wall time
and shared-noise latent drift + stop-step stability vs the 1-step baseline.
"""
import os
import sys
import time

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


def ar_loop(m, lite, c, zero_steps, noises):
    """Run the single-path AR loop with a custom number of skipped steps."""
    slot = "centroid"
    tt_t, af_t, txm_t, aum_t, spk_t = lite._build_inputs(TEXT, None, None, slot)
    tt = mx.array(tt_t.numpy().astype(np.int32))[None]
    af = mx.array(af_t.float().numpy())[None]
    tx = mx.array(txm_t.float().numpy())[None]
    au = mx.array(aum_t.float().numpy())[None]
    sc = mx.array(c.reshape(1, -1).float().numpy())
    sm = mx.array(spk_t.float().numpy())[None]

    fl = m.locenc(af)
    comb = tx[..., None] * bm.embed_take(m.embed, tt) + au[..., None] * _proj(fl, m.enc_tslm)
    nw, pw, pb, eps = m.spk
    spk_vec = bm._lin(bm._rms_norm(sc, nw, eps), pw, pb)
    comb = comb + sm[..., None] * spk_vec[:, None, :]
    bcache = m.barbet.init_cache()
    bm._PREFILL_FP32[0] = True
    h = m.barbet(comb, cache=bcache)
    tslm_h = m.adapter(h)
    enc = m.fsq(tslm_h) * au[..., None] + tslm_h * tx[..., None]
    felm = _proj(fl, m.enc_lm)
    rin = _proj(mx.concatenate([enc, au[..., None] * felm], axis=-1), m.fusion)
    rcache = m.ralm.init_cache()
    rs = m.ralm(rin, cache=rcache)
    bm._PREFILL_FP32[0] = False
    lm_h, res_h = enc[:, -1, :], rs[:, -1, :]
    cond = af[:, -1, ...]
    pos = int(tt.shape[1])
    t_span = m._t_span(9)

    patches = []
    t0 = time.time()
    n_dit_calls = 0
    for i in range(30):
        dh = mx.concatenate([_proj(lm_h, m.lm_dit), _proj(res_h, m.res_dit)], axis=-1)
        mx.random.seed(1000 + i)
        z = noises[i % len(noises)]
        pred = solve_euler(m.dit, z, t_span, dh, mx.transpose(cond, (0, 2, 1)), 2.8,
                           use_cfg_zero_star=True, zero_init_steps=zero_steps)
        n_dit_calls += max(0, 9 - 1 - zero_steps)
        pf = mx.transpose(pred, (0, 2, 1))
        le = m.locenc(pf[:, None])
        ct, cl = _proj(le, m.enc_tslm), _proj(le, m.enc_lm)
        patches.append(pf)
        cond = pf

        sl = bm._lin(bm._silu(bm._lin(lm_h, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
        p_stop = float(mx.softmax(sl.astype(mx.float32), axis=-1)[0, 1])
        if i > 2:
            hits = getattr(ar_loop, "_hits", 0)
            hits = hits + 1 if p_stop >= 0.65 else 0
            ar_loop._hits = hits
            if hits >= 2 or (i + 1) >= int((len(TEXT) * 1.5 + 6) * 1.5):
                break
        else:
            ar_loop._hits = 0

        bs = m.barbet.step(ct[:, 0, :], pos, bcache)
        lm_h = m.fsq(m.adapter(bs[:, None, :]))[:, 0, :]
        cr = _proj(mx.concatenate([lm_h, cl[:, 0, :]], axis=-1), m.fusion)
        res_h = m.ralm.step(cr, pos, rcache)
        pos += 1
    wall = time.time() - t0
    ar_loop._hits = 0
    return mx.concatenate(patches, axis=0), wall


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    # warm
    _warm_mu = mx.zeros((1, 64, 1024))
    solve_euler(m.dit, mx.random.normal((1, m.feat_dim, m.patch_size)),
                m._t_span(9), _warm_mu, mx.zeros((1, m.feat_dim, m.patch_size)), 2.8)

    mx.random.seed(99)
    noises = [mx.random.normal((1, m.feat_dim, m.patch_size)) for _ in range(40)]

    ref_lat, ref_wall = ar_loop(m, lite, c, 1, noises)
    print(f"zero_steps=1 (current): {ref_lat.shape[0]} patches, {ref_wall:.1f}s")
    for zs in (2, 3, 4):
        lat, wall = ar_loop(m, lite, c, zs, noises)
        k = min(ref_lat.shape[0], lat.shape[0])
        d = float(np.abs(np.array(ref_lat)[:k] - np.array(lat)[:k]).max())
        corr = float(np.corrcoef(np.array(ref_lat)[:k].ravel(), np.array(lat)[:k].ravel())[0, 1])
        speedup = (9 - 1) / (9 - 1 - zs) if zs < 8 else float("inf")
        print(f"zero_steps={zs}: {lat.shape[0]} patches, {wall:.1f}s "
              f"(DiT calls {-100*zs/8:+.0f}%) | latent corr={corr:.4f} max|diff|={d:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
