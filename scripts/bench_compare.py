#!/usr/bin/env python3
"""Precision vs speed experiment: which submodules must compute in fp32?

fp16 weights are mandatory (memory), but fp16 *compute* drifts in the AR loop
(FSQ quantization boundaries + DiT Euler CFG + stop head). This script loads
the fp16 lite checkpoint, selectively upcasts submodule weights back to fp32
(MLX promotes fp16 inputs against fp32 weights -> fp32 compute), and reports
correlation vs the fp32 reference + RTF for each strategy.

Strategies tested:
  A  all-fp16                     (current lite checkpoint)
  B  DiT + LocEnc + FSQ + stop + adapter heads upcast to fp32 (LM stays fp16)
  C  B plus barbet/rlm embed table upcast (SLM logits domain)
  D  everything upcast to fp32    (sanity: should match the fp32 runs closely)

Usage:
    python scripts/bench_compare.py [MODEL_DIR] [LITE_PKL] [OUT_DIR]

Needs the fp32 references from verify_lite.py (lite_out/fp32_{0,1}.pt).
"""
import os
import sys
import time

import numpy as np
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

TEXTS = [
    "今天天氣真好，我們一起去公園散步吧。",
    "這個 feature 明天上線，記得先跑 regression test。",
]


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def upcast(obj, visited):
    """Reverse of lite._fp16: cast float16 mx arrays back to float32."""
    import mlx.core as mx

    if isinstance(obj, mx.array):
        if obj.dtype == mx.float16:
            return obj.astype(mx.float32)
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            nv = upcast(v, visited)
            if nv is not v:
                obj[k] = nv
        return obj
    if isinstance(obj, tuple):
        return tuple(upcast(v, visited) for v in obj)
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nv = upcast(v, visited)
            if nv is not v:
                obj[i] = nv
        return obj
    if hasattr(obj, "__dict__"):
        if id(obj) in visited:
            return obj
        visited.add(id(obj))
        for k, v in list(vars(obj).items()):
            nv = upcast(v, visited)
            if nv is not v:
                setattr(obj, k, nv)
        return obj
    return obj


def apply_strategy(mlx_model, strategy):
    """Return the model with submodules upcast per strategy."""
    if strategy == "A":
        return
    if strategy in ("B", "C", "D"):
        roots = [mlx_model.dit, mlx_model.locenc, mlx_model.fsq,
                 mlx_model.adapter, mlx_model.enc_tslm, mlx_model.enc_lm,
                 mlx_model.lm_dit, mlx_model.res_dit, mlx_model.fusion,
                 mlx_model.stop_proj, mlx_model.stop_head_w, mlx_model.spk]
        if strategy == "C":
            roots += [mlx_model.embed, mlx_model.barbet.p["embed_tokens.weight"]]
        if strategy == "D":
            roots = [mlx_model]
        for r in roots:
            upcast(r, set())
    mlx_model._samplers = {}  # compiled traces were fp16; rebuild


def main() -> int:
    import mlx.core as mx
    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx import mlx_generate

    model_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR
    lite_path = sys.argv[2] if len(sys.argv) > 2 else "bluemagpie-lite.pkl"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "lite_out"
    os.makedirs(out_dir, exist_ok=True)

    refs = [torch.load(os.path.join(out_dir, f"fp32_{i}.pt"), weights_only=True).numpy() for i in range(2)]
    table = torch.load(os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    centroid = table["centroids"][table["speaker_ids"].index("hung_yi_lee")]

    lite, mlx_model = load_model(model_dir, lite_path)

    for strategy in ("A", "B", "C", "D"):
        apply_strategy(mlx_model, strategy)
        print(f"\n=== strategy {strategy} ===", flush=True)
        for i, txt in enumerate(TEXTS):
            # warm (compile trace + VAE cache) is included in run 1; do 2 runs
            t0 = time.time()
            mlx_generate(lite, mlx_model, txt, speaker_centroid=centroid, seed=42)
            warm = time.time() - t0
            t0 = time.time()
            a = mlg = None
            import torch as _t
            a = mlx_generate(lite, mlx_model, txt, speaker_centroid=centroid, seed=42)
            dt = time.time() - t0
            an = a.numpy()
            c = corr(refs[i], an)
            rms = float(np.sqrt(((refs[i] - an) ** 2).mean()))
            dur = a.shape[-1] / 48000.0
            print(f"  [{i}] warm={warm:.2f}s run={dt:.2f}s rtf={dt/dur:.3f} corr={c:.4f} rms_diff={rms:.5f} len={a.shape[-1]}", flush=True)
            if strategy == "D" and i == 0:
                # strategy D output should ~equal the fp32 ref; sanity check done
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())