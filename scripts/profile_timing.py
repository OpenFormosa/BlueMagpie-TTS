#!/usr/bin/env python3
"""Profile where generation time goes: prefill vs AR loop vs AudioVAE decode.

Usage: python scripts/profile_timing.py [LITE_PKL] [TEXT]
"""
import os
import sys
import time

import mlx.core as mx
import torch

MODEL_DIR = os.path.join(
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


def main() -> int:
    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx.model_mlx import _proj, _lin, _silu, to_mx

    lite_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "bluemagpie-lite-fp32.pkl"
    text = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else TEXT

    lite, m = load_model(MODEL_DIR, lite_path)
    table = torch.load(os.path.join(MODEL_DIR, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    centroid = table["centroids"][table["speaker_ids"].index("female_voice")]
    lite._runtime_dtype  # noqa

    text_token, audio_feat, text_mask, audio_mask, spk_mask = lite._build_inputs(text)
    tt = mx.array(text_token.cpu().numpy())[None]
    af = to_mx(audio_feat.float())[None]
    txm = to_mx(text_mask.float())[None]
    aum = to_mx(audio_mask.float())[None]

    # ---- prefill (mirror of model_mlx.inference) ----
    t0 = time.time()
    feat_locenc = m.locenc(af)
    feat_embed_tslm = _proj(feat_locenc, m.enc_tslm)
    feat_embed_lm = _proj(feat_locenc, m.enc_lm)
    text_embed = mx.take(m.embed, tt, axis=0)
    combined = txm[..., None] * text_embed + aum[..., None] * feat_embed_tslm
    bcache = m.barbet.init_cache()
    barbet_hidden = m.barbet.prefill(combined, bcache)
    mx.eval(barbet_hidden)
    t_prefill = time.time() - t0

    t0 = time.time()
    tslm_hidden = m.adapter(barbet_hidden)
    enc_outputs = m.fsq(tslm_hidden) * aum[..., None] + tslm_hidden * txm[..., None]
    lm_hidden = enc_outputs[:, -1, :]
    residual_inputs = _proj(mx.concatenate([enc_outputs, aum[..., None] * feat_embed_lm], axis=-1), m.fusion)
    rcache = m.ralm.init_cache()
    residual_seq = m.ralm.prefill(residual_inputs, rcache)
    residual_hidden = residual_seq[:, -1, :]
    mx.eval(residual_hidden)
    t_prefill2 = time.time() - t0

    # ---- warm DiT sampler (one-time compile) ----
    m.dit.decoder._rope(3 + 2 * m.patch_size)
    sampler = m._sampler(9, 2.8)
    z0 = mx.random.normal((1, m.feat_dim, m.patch_size))
    dit_hidden = mx.concatenate([_proj(lm_hidden, m.lm_dit), _proj(residual_hidden, m.res_dit)], axis=-1)
    cond0 = mx.transpose(af[:, -1, ...], (0, 2, 1))
    pred0 = sampler(z0, dit_hidden, cond0)
    mx.eval(pred0)
    t_warm = time.time() - t0

    # ---- AR loop, timed per phase ----
    prefix_feat_cond = af[:, -1, ...]
    pos = int(tt.shape[1])
    t_span = m._t_span(9)
    t_dit = t_locenc = t_step = t_eval = 0.0
    n = 5
    for i in range(n):
        t1 = time.time()
        dit_hidden = mx.concatenate([_proj(lm_hidden, m.lm_dit), _proj(residual_hidden, m.res_dit)], axis=-1)
        cond = mx.transpose(prefix_feat_cond, (0, 2, 1))
        pred = sampler(z0, dit_hidden, cond)
        z0 = mx.random.normal((1, m.feat_dim, m.patch_size))
        pred_feat = mx.transpose(pred, (0, 2, 1))
        t2 = time.time()
        curr_locenc = m.locenc(pred_feat[:, None])
        curr_tslm = _proj(curr_locenc, m.enc_tslm)
        curr_lm = _proj(curr_locenc, m.enc_lm)
        t3 = time.time()
        stop_logits = _lin(_silu(_lin(lm_hidden, m.stop_proj[0], m.stop_proj[1])), m.stop_head_w)
        stop = int(mx.argmax(stop_logits, axis=-1)[0])
        barbet_step = m.barbet.step(curr_tslm[:, 0, :], pos, bcache)
        lm_hidden = m.fsq(m.adapter(barbet_step[:, None, :]))[:, 0, :]
        curr_residual = _proj(mx.concatenate([lm_hidden, curr_lm[:, 0, :]], axis=-1), m.fusion)
        residual_hidden = m.ralm.step(curr_residual, pos, rcache)
        pos += 1
        prefix_feat_cond = pred_feat
        t4 = time.time()
        mx.eval([pred_feat, lm_hidden, residual_hidden])
        t5 = time.time()
        t_dit += t2 - t1
        t_locenc += t3 - t2
        t_step += t4 - t3
        t_eval += t5 - t4

    latents = mx.concatenate([prefix_feat_cond], axis=0) if False else None  # noqa
    t0 = time.time()
    audio = m.decode_latents(mx.concatenate([pred_feat] * 20, axis=0))  # ~20 patches decode timing
    mx.eval(audio)
    t_decode20 = time.time() - t0

    print(f"prefill barbet        : {t_prefill:.3f}s")
    print(f"prefill ralm+adapter  : {t_prefill2:.3f}s")
    print(f"sampler warm (1-time) : {t_warm:.3f}s")
    print(f"AR loop {n} patches   : {(t_dit + t_locenc + t_step + t_eval):.3f}s")
    print(f"  DiT+proj            : {t_dit / n * 1000:.0f} ms/patch")
    print(f"  locenc+proj         : {t_locenc / n * 1000:.0f} ms/patch")
    print(f"  LM steps(barbet+ralm): {t_step / n * 1000:.0f} ms/patch")
    print(f"  mx.eval             : {t_eval / n * 1000:.0f} ms/patch")
    print(f"AudioVAE decode 20pat : {t_decode20:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())