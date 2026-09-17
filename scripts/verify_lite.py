#!/usr/bin/env python3
"""Verify the fp16 lite path: quality vs fp32, RTF, and streaming alignment.

Runs in three stages (each needs the previous):
  1. fp32 baseline  — stock MLX path (torch model + fp32 MLX), same seed
  2. fp16 lite      — load_model + mlx_generate, same seed
  3. streaming      — mlx_generate_streaming chunks vs non-streaming audio

Prints: waveform correlation (fp32 vs fp16), RTF for each backend, and the
streaming chunk-length/alignment summary. Requires the lite checkpoint
(scripts/convert_lite.py) to have been built.

Usage:
    python scripts/verify_lite.py [MODEL_DIR] [LITE_PKL] [OUT_DIR]
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


def rtf(waveform, seconds):
    dur = waveform.shape[-1] / 48000.0
    return seconds / dur


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else MODEL_DIR
    lite_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "bluemagpie-lite.pkl"
    out_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "lite_out"
    os.makedirs(out_dir, exist_ok=True)

    # ---- stage 1: fp32 stock path (skipped if fp32 refs already exist) ----
    from transformers import PreTrainedTokenizerFast
    from bluemagpie import BlueMagpieModel

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
    table = torch.load(os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    centroid = table["centroids"][table["speaker_ids"].index("hung_yi_lee")]
    refs = {}
    have_refs = all(os.path.exists(os.path.join(out_dir, f"fp32_{i}.pt")) for i in range(len(TEXTS)))
    if have_refs:
        for i in range(len(TEXTS)):
            refs[i] = torch.load(os.path.join(out_dir, f"fp32_{i}.pt"), weights_only=True).numpy()
        print("[1/3] fp32 refs found on disk, skipping stock path", flush=True)
    else:
        print("[1/3] fp32 stock path: loading torch model ...", flush=True)
        model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, device="cpu")

        from bluemagpie.mlx import BlueMagpieMLX, mlx_generate

        mlx32 = BlueMagpieMLX(model)  # fp32 (~15 GB peak with the torch model)
        for i, txt in enumerate(TEXTS):
            t0 = time.time()
            a32 = mlx_generate(model, mlx32, txt, speaker_centroid=centroid, seed=42)
            dt = time.time() - t0
            refs[i] = a32.numpy()
            print(f"  fp32  [{i}] {dt:.2f}s rtf={rtf(a32, dt):.3f} len={a32.shape[-1]}", flush=True)
            torch.save(a32, os.path.join(out_dir, f"fp32_{i}.pt"))
        del mlx32, model  # release ~15 GB before the fp16 stage
        import gc

        gc.collect()

    # ---- stage 2: fp16 lite path ----
    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx import mlx_generate as mlx_generate_lite

    print("[2/3] fp16 lite path: loading lite checkpoint ...", flush=True)
    lite, mlx16 = load_model(model_dir, lite_path)
    # mlx_generate(lite, ...) uses lite._encode_wav / _build_inputs for input assembly
    wall = time.time()
    for i, txt in enumerate(TEXTS):
        t0 = time.time()
        a16 = mlx_generate_lite(lite, mlx16, txt, speaker_centroid=centroid, seed=42)
        dt = time.time() - t0
        a16n = a16.numpy()
        c = corr(refs[i], a16n)
        rms_diff = float(np.sqrt(((refs[i] - a16n) ** 2).mean()))
        print(f"  fp16  [{i}] {dt:.2f}s rtf={rtf(a16, dt):.3f} len={a16.shape[-1]} corr={c:.4f} rms_diff={rms_diff:.6f}", flush=True)
        torch.save(a16, os.path.join(out_dir, f"fp16_{i}.pt"))

    # ---- stage 3: streaming alignment ----
    from bluemagpie.mlx import mlx_generate_streaming

    print("[3/3] streaming vs non-streaming ...", flush=True)
    for i, txt in enumerate(TEXTS):
        chunks = list(mlx_generate_streaming(lite, mlx16, txt, speaker_centroid=centroid,
                                             seed=42, chunk_patches=4))
        lens = [c.shape[-1] for c in chunks]
        total = int(sum(lens))
        ref_len = int(refs[i].shape[-1])
        # streaming decode trims a few samples at chunk boundaries (same as
        # torch's StreamingVAEDecoder); compare against the fp16 reference
        # with the tail trimmed to the shorter of the two.
        n = min(total, ref_len)
        cat = np.concatenate([c.numpy() for c in chunks])[:n]
        c = corr(refs[i][:n], cat)
        print(f"  stream[{i}] chunks={len(chunks)} lens={lens} total={total} ref={ref_len} corr(head {n})={c:.4f}", flush=True)

    print(f"total wall: {time.time() - wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())