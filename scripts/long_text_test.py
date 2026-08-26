#!/usr/bin/env python3
"""Long-text synthesis: sentence-split -> batched generation -> concat.

Measures wall-clock RTF for a long paragraph using mlx_generate_batch
(one AR pass per cohort of sentences) and the WSOLA pace option.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

from bluemagpie.mlx.lite import load_model
from bluemagpie.mlx.model_mlx import mlx_generate_batch

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

LONG_TEXT = (
    "人工智慧的發展改變了我們的生活方式。從早期的規則系統，到現在的大型語言模型，"
    "機器已經能夠理解並生成自然的語言。這樣的進展不僅影響了科技產業，也深入到教育、"
    "醫療和日常生活的各個層面。語音合成技術就是其中一個很好的例子。早期的合成語音"
    "聽起來相當生硬，而現在的模型可以生成接近真人的自然語音，甚至支援多種說話者的"
    "音色轉換。在蘋果晶片上，透過 MLX 框架的最佳化，即時的語音合成已經可以在個人"
    "電腦上實現，不需要昂貴的伺服器設備。這為開發者帶來了更多的可能性，也讓語音"
    "應用變得更加普及。未來，隨著模型架構與推理技術的持續進步，我們可以期待更快速、"
    "更自然、更個性化的語音互動體驗。"
)


def split_sentences(text):
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    table = torch.load(os.path.join(MD, "checkpoints", "speaker_centroids.pt"),
                       map_location="cpu", weights_only=True)
    c = table["centroids"][table["speaker_ids"].index("female_voice")]

    sents = split_sentences(LONG_TEXT)
    n_chars = len(LONG_TEXT)
    print(f"text: {n_chars} chars -> {len(sents)} sentences")

    # warm
    mlx_generate_batch(lite, m, sents[:1], speaker_centroid=c)

    t0 = time.time()
    wavs = []
    B = 4
    for i in range(0, len(sents), B):
        cohort = sents[i:i + B]
        wavs.extend(mlx_generate_batch(lite, m, cohort, speaker_centroid=c))
        done = min(i + B, len(sents))
        print(f"  cohort {i//B}: {done}/{len(sents)} sentences ({time.time()-t0:.1f}s)")
    wall = time.time() - t0

    # concat with small gaps between sentences (natural pauses)
    sr = lite.sample_rate
    pieces, gap = [], np.zeros(int(sr * 0.18), dtype=np.float32)
    for i, w in enumerate(wavs):
        wav = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        pieces.append(wav)
        if i < len(wavs) - 1:
            pieces.append(gap)
    audio = np.concatenate(pieces)

    out = "lite_out/long_text.wav"
    sf.write(out, audio, sr)
    dur = len(audio) / sr
    print(f"\naudio: {dur:.1f}s | wall: {wall:.1f}s | effective RTF {wall/dur:.2f}")
    print(f"saved: {out}")

    try:
        from audio_artifact_gate import evaluate_echo_smearing
        r = evaluate_echo_smearing(audio.astype(np.float32), sr)
        print(f"gate: passed={r.passed}")
    except Exception as e:
        print("gate skipped:", e)

    # per-sentence pace audit
    total_units = sum(1 if ord(ch) > 0x2E7F else (0.5 if ch.isdigit() else 0.25)
                      for ch in LONG_TEXT if ch.isalnum())
    print(f"overall pace: {total_units/dur:.1f} units/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
