"""End-to-end RTF benchmark on the real BlueMagpie-TTS model.

RTF = compute_time / audio_seconds (lower is better; <1 = faster than real time).
Compares torch-CPU, torch-MPS, and the MLX (Apple-Silicon GPU) path on the same
zero-shot utterances with identical generation params.

Run: python scripts/bench_rtf.py
"""

import os
import time

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

from bluemagpie import BlueMagpieModel

TEXTS = ["今天天氣真好，我們一起出去走走吧。", "人工智慧正在快速改變我們的生活方式。"]
TIMESTEPS, CFG = 9, 2.8
SR = 48000


def _audio_seconds(wav):
    return wav.shape[-1] / SR


def _stats(wav):
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    return dict(n=a.shape[0], rms=float(np.sqrt(np.mean(a ** 2))), peak=float(np.max(np.abs(a))),
               nan=bool(np.isnan(a).any()))


def bench_torch(model, device, texts):
    model = model.to(device)
    # The RALM's StaticKVCache is pre-allocated at __init__ (bf16); re-setup it in
    # fp32 on the target device so it matches the fp32 params.
    model.residual_lm.setup_cache(1, model.config.max_length, torch.device(device), torch.float32)
    gen = lambda t: model.generate(target_text=t, inference_timesteps=TIMESTEPS, cfg_value=CFG)
    with torch.no_grad():
        a0 = gen(texts[0])  # warm
        if device == "mps":
            torch.mps.synchronize()
        rows = []
        for t in texts:
            t0 = time.perf_counter()
            wav = gen(t)
            if device == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            sec = _audio_seconds(wav)
            rows.append((dt, sec, _stats(wav)))
    return rows


def bench_mlx(model, texts):
    import mlx.core as mx
    from bluemagpie.mlx import BlueMagpieMLX, mlx_generate

    mlxm = BlueMagpieMLX(model)  # converts weights once
    mlx_generate(model, mlxm, texts[0], inference_timesteps=TIMESTEPS, cfg_value=CFG, seed=0)  # warm (compile)
    rows = []
    for i, t in enumerate(texts):
        t0 = time.perf_counter()
        wav = mlx_generate(model, mlxm, t, inference_timesteps=TIMESTEPS, cfg_value=CFG, seed=i)
        dt = time.perf_counter() - t0
        rows.append((dt, _audio_seconds(wav), _stats(wav)))
    return rows


def _print(name, rows):
    print(f"\n=== {name} ===")
    for i, (dt, sec, st) in enumerate(rows):
        print(f"  text{i}: {dt:6.2f}s gen | {sec:5.2f}s audio | RTF {dt/sec:5.2f} | "
              f"rms {st['rms']:.3f} peak {st['peak']:.2f} nan={st['nan']}")
    rtfs = [dt / sec for dt, sec, _ in rows]
    print(f"  mean RTF: {np.mean(rtfs):.2f}")
    return float(np.mean(rtfs))


def main():
    model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS")
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
    print("loading model (cpu)...")
    model = BlueMagpieModel.from_local(model_dir, tokenizer=tok, training=False, device="cpu")
    model = model.float()  # fp32 everywhere for a fair compute comparison (bf16 glitches on MPS)

    results = {}
    # MLX first (model on cpu; input assembly uses cpu tokenizer).
    print("benchmarking MLX (Apple-Silicon GPU)...")
    results["mlx-gpu"] = _print("MLX (Apple-Silicon GPU, fp32)", bench_mlx(model, TEXTS))

    print("benchmarking torch-MPS...")
    try:
        results["torch-mps"] = _print("torch-MPS (fp32)", bench_torch(model, "mps", TEXTS))
    except Exception as e:
        print("  torch-MPS failed:", str(e)[:120])
    model = model.to("cpu")

    print("benchmarking torch-CPU...")
    try:
        results["torch-cpu"] = _print("torch-CPU", bench_torch(model, "cpu", TEXTS))
    except Exception as e:
        print("  torch-CPU failed:", str(e)[:120])

    print("\n=== summary (mean RTF, lower=faster) ===")
    for k, v in results.items():
        print(f"  {k:10s}: {v:.2f}")
    if "torch-mps" in results and "mlx-gpu" in results:
        print(f"  MLX-GPU vs torch-MPS speedup: {results['torch-mps']/results['mlx-gpu']:.2f}x")
    if "torch-cpu" in results and "mlx-gpu" in results:
        print(f"  MLX-GPU vs torch-CPU speedup: {results['torch-cpu']/results['mlx-gpu']:.2f}x")


if __name__ == "__main__":
    main()
