#!/usr/bin/env python3
"""Generate demo wavs from the fp32 lite checkpoint (normal + streaming).

Usage:
    python scripts/demo_speak.py [MODEL_DIR] [LITE_PKL] [OUT_DIR]
"""
import os
import sys
import time

import soundfile as sf
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
    "各位觀眾朋友大家好，歡迎收看今天的晚間新聞。",
]


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else MODEL_DIR
    lite_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "bluemagpie-lite-fp32.pkl"
    out_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "lite_out"
    os.makedirs(out_dir, exist_ok=True)

    from bluemagpie.mlx.lite import load_model
    from bluemagpie.mlx import mlx_generate, mlx_generate_streaming

    print(f"loading lite: {lite_path}", flush=True)
    t0 = time.time()
    lite, mlx_model = load_model(model_dir, lite_path)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    table = torch.load(os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    centroid = table["centroids"][table["speaker_ids"].index("hung_yi_lee")]

    for i, txt in enumerate(TEXTS):
        # normal (warm on first call — includes DiT compile trace)
        t0 = time.time()
        audio = mlx_generate(lite, mlx_model, txt, speaker_centroid=centroid, seed=7)
        dt = time.time() - t0
        path = os.path.join(out_dir, f"demo_fp32_{i}.wav")
        sf.write(path, audio.numpy(), lite.sample_rate)
        print(f"[{i}] normal: {dt:.2f}s -> {path}", flush=True)

        # streaming: report per-chunk latency + total
        t0 = time.time()
        chunks = []
        first = None
        for ci, c in enumerate(mlx_generate_streaming(lite, mlx_model, txt,
                                                      speaker_centroid=centroid, seed=7,
                                                      chunk_patches=4)):
            if first is None:
                first = time.time() - t0
            chunks.append(c)
        dt = time.time() - t0
        cat = torch.cat(chunks)
        path = os.path.join(out_dir, f"demo_stream_{i}.wav")
        sf.write(path, cat.numpy(), lite.sample_rate)
        dur = cat.shape[-1] / lite.sample_rate
        print(f"[{i}] stream: {dt:.2f}s total (first chunk @ {first:.2f}s, {len(chunks)} chunks, {dur:.1f}s audio)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())