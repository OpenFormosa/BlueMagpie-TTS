#!/usr/bin/env python3
"""Diagnose batch row quality: rms/energy profile per output."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

for i in range(4):
    wav, sr = sf.read(f"lite_out/batch_{i}_b.wav")
    rms = float(np.sqrt((wav ** 2).mean()))
    peak = float(np.abs(wav).max())
    tail = float(np.sqrt((wav[-sr // 2:] ** 2).mean())) if len(wav) > sr // 2 else 0.0
    head = float(np.sqrt((wav[: sr // 4] ** 2).mean()))
    print(f"[{i}] dur={len(wav)/sr:.1f}s rms={rms:.3f} peak={peak:.3f} head={head:.3f} tail={tail:.3f}")
