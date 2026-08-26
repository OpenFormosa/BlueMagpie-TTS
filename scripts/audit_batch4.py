#!/usr/bin/env python3
"""B=4 quality audit: per-row duration sanity vs text length."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

TEXTS = [
    "今天天氣真好，我們一起去公園散步吧。",
    "這個 feature 明天上線，記得先跑 regression test。",
    "會議改到下午三點，請大家準時參加。",
    "晚餐想吃什麼？附近新開了一間拉麵店。",
]

for i in range(4):
    wav, sr = sf.read(f"lite_out/batch4_{i}.wav")
    dur = len(wav) / sr
    units = sum(1 if ord(ch) > 0x2E7F else (0.5 if ch.isdigit() else 0.25)
                for ch in TEXTS[i] if ch.isalnum())
    cps = units / dur
    rms = float(np.sqrt((wav ** 2).mean()))
    flag = "OK" if 2.5 <= cps <= 8.0 and rms < 0.3 else "CHECK"
    print(f"[{i}] dur={dur:5.1f}s units={units:4.1f} pace={cps:4.1f}u/s rms={rms:.3f} {flag}")
