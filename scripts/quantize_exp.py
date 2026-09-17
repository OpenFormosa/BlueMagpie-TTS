#!/usr/bin/env python3
"""MLX affine quantization experiment for the BlueMagpie MLX model.

Applies mx.quantize (grouped affine: one fp32 scale/bias per group, weights
stored packed 4/8-bit) to every eligible 2-D Linear weight under
BlueMagpieMLX — embeddings, 1-D norm weights and 3-D conv weights excluded —
replacing each fp32 array in place. The fused mx.quantized_matmul kernel is
dispatched through a _lin hook (barbet_mlx._QUANTIZED), so peak memory stays
low (no dequantized fp32 blobs).

Reports correlation vs the fp32 reference, RTF, and weight memory saved.

Usage:
    python scripts/quantize_exp.py [LITE_PKL] [BITS] [GROUP] [TEXT]
    bits: 4 or 8 (default 4); group: 32 or 64 (default 64)
    Needs lite_out/fp32_1.pt (run scripts/verify_lite.py first).
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

TEXT = "這個 feature 明天上線，記得先跑 regression test。"


def corr(a, b):
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def quantize_model(mlx_model, bits: int, group: int, protect: tuple = ()) -> dict:
    """Quantize every eligible 2-D weight in place; return stats.

    ``protect``: path substrings that must stay fp32 (AR-divergence-critical
    projections — adapter, FSQ, locenc, DiT IO, AR projections, stop head).
    """
    import mlx.core as mx
    from bluemagpie.mlx.barbet_mlx import register_quantized

    stats = {"quantized": 0, "skipped_embed": 0, "protected": 0, "bytes_saved": 0}

    def walk(obj, visited, path):
        if isinstance(obj, mx.array):
            a = obj
            if "embed" in path:
                stats["skipped_embed"] += 1
                return obj
            if a.ndim == 2 and a.shape[-1] % group == 0:
                if any(p in path for p in protect):
                    stats["protected"] += 1
                    return obj
                wq, sc, bi = mx.quantize(a, group_size=group, bits=bits)
                register_quantized(wq, sc, bi, group, bits)
                stats["quantized"] += 1
                stats["bytes_saved"] += a.size * 4 - (wq.size * 4 + sc.size * 4)
                return wq  # replaced in the parent slot; fp32 array dropped
            return obj
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                nv = walk(v, visited, f"{path}.{k}" if path else k)
                if nv is not v:
                    obj[k] = nv
            return obj
        if isinstance(obj, tuple):
            return tuple(walk(v, visited, f"{path}[{i}]") for i, v in enumerate(obj))
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                nv = walk(v, visited, f"{path}[{i}]")
                if nv is not v:
                    obj[i] = nv
            return obj
        if hasattr(obj, "__dict__"):
            if id(obj) in visited:
                return obj
            visited.add(id(obj))
            for k, v in list(vars(obj).items()):
                nv = walk(v, visited, f"{path}.{k}" if path else k)
                if nv is not v:
                    setattr(obj, k, nv)
            return obj
        return obj

    walk(mlx_model, set(), "")
    return stats


PROTECT_DEFAULT = (
    "adapter.p",          # FSQ-boundary-sensitive (barbet hidden -> quantized code)
    "fsq.",               # quantizer IO (boundary flip = different content)
    "locenc.",            # patch encoder IO
    ".dit.in_", ".dit.cond_", ".dit.out_", ".dit.t1", ".dit.t2", ".dit.d1", ".dit.d2",  # DiT IO/emb
    "enc_tslm", "enc_lm", "lm_dit", "res_dit", "fusion", "stop_proj", "stop_head_w", "spk",
)


def main() -> int:
    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx import mlx_generate

    lite_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else "bluemagpie-lite-fp32.pkl"
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    group = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    text = sys.argv[4] if len(sys.argv) > 4 else TEXT

    table = torch.load(os.path.join(MODEL_DIR, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    centroid = table["centroids"][table["speaker_ids"].index("female_voice")]
    ref = torch.load(os.path.join("lite_out", "fp32_1.pt"), weights_only=True).numpy()

    lite, mlx_model = load_model(MODEL_DIR, lite_path)
    protect = PROTECT_DEFAULT if "--all" not in sys.argv else ()
    stats = quantize_model(mlx_model, bits, group, protect=protect)
    print(f"quantized {stats['quantized']} weights (embed skip: {stats['skipped_embed']}, "
          f"protected fp32: {stats['protected']}), ~{stats['bytes_saved'] / 1e9:.2f} GB saved")

    # warm (DiT compile trace + first run)
    t0 = time.time()
    mlx_generate(lite, mlx_model, text, speaker_centroid=centroid, seed=42)
    warm = time.time() - t0

    t0 = time.time()
    a = mlx_generate(lite, mlx_model, text, speaker_centroid=centroid, seed=42)
    dt = time.time() - t0
    an = a.numpy()
    dur = an.shape[-1] / lite.sample_rate
    print(f"bits={bits} group={group}: warm={warm:.2f}s run={dt:.2f}s rtf={dt / dur:.3f} "
          f"corr_vs_fp32={corr(ref, an):.4f} len={an.shape[-1]} (ref {ref.shape[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())