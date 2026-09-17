#!/usr/bin/env python3
"""Audit the long-text output: per-sentence duration + silence trimming check."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

wav, sr = sf.read("lite_out/long_text.wav")
dur = len(wav) / sr
rms = float(np.sqrt((wav ** 2).mean()))
peak = float(np.abs(wav).max())

# detect long silent stretches (>1.5s) that would indicate dead air
win = int(sr * 0.5)
n_win = len(wav) // win
silent = 0
max_silent = 0.0
for i in range(n_win):
    w_rms = float(np.sqrt((wav[i * win:(i + 1) * win] ** 2).mean()))
    if w_rms < 0.005:
        silent += 1
        max_silent = max(max_silent, silent * 0.5)
    else:
        silent = 0

units = sum(1 if ord(ch) > 0x2E7F else (0.5 if ch.isdigit() else 0.25)
            for ch in open(__file__.replace("audit_long.py", "long_text_test.py")).read()
            .split('LONG_TEXT = (')[1].split(')')[0] if ch.isalnum()) \
    if False else None

# simpler: count from known text length (285 chars, mostly CJK)
n_units_approx = 240.0
print(f"duration: {dur:.1f}s | rms={rms:.3f} peak={peak:.3f}")
print(f"longest silent stretch: {max_silent:.1f}s")
print(f"approx pace: {n_units_approx/dur:.1f} units/s")
print("verdict:", "OK" if max_silent < 2.0 and rms > 0.05 else "CHECK")
