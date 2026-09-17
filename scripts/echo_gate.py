#!/usr/bin/env python3
"""Echo / smearing artifact detector (cepstral, no-reference, numpy-only).

Re-implements the approach documented in the official BlueMagpie demo's
audio_artifact_gate: look for a *time-invariant* comb/echo signature that
survives across several active analysis windows —

* a narrow long-quefrency cepstral peak  -> delayed-copy (echo) evidence;
* a dense set of long-quefrency peaks    -> reverberant smearing evidence.

Content-dependent pitch/formants move between windows while a fixed echo path
does not, so taking the **median** cepstrum across windows suppresses false
positives. Deterministic for identical input.

Usage:
    python scripts/echo_gate.py WAV [WAV ...]     # score files
    (or import echo_score / passes_gate from other scripts)
"""
import sys

import numpy as np

# Config mirrors the official demo's defaults where applicable.
WINDOW_SECONDS = 0.60
MAX_WINDOWS = 9
MIN_WINDOWS = 2
MIN_DELAY_MS = 24.0
MAX_DELAY_MS = 180.0
FREQ_SMOOTH_HZ = 70.0
RELATIVE_ACTIVE_RMS = 0.12
# Pass thresholds (calibrated on local fp32 outputs; fail-closed).
DELAYED_COPY_PASS = 0.35
SMEARING_PASS = 0.45


def _active_starts(x, sr):
    win = int(WINDOW_SECONDS * sr)
    hop = win // 2
    n = (len(x) - win) // hop + 1
    if n < 1:
        return []
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2)) for i in range(n)])
    top = float(rms.max())
    if top < 1e-5:
        return []
    act = np.where(rms >= RELATIVE_ACTIVE_RMS * top)[0]
    if len(act) == 0:
        return []
    # up to MAX_WINDOWS evenly spaced active windows
    idx = np.linspace(0, len(act) - 1, min(MAX_WINDOWS, len(act))).round().astype(int)
    starts = sorted({int(a) * hop for a in act[idx]})
    return starts[:MAX_WINDOWS] if len(starts) >= MIN_WINDOWS else []


def _window_cepstrum(x, sr):
    win = x - x.mean()
    spec = np.fft.rfft(win * np.hanning(len(win)))
    logmag = np.log(np.abs(spec) + 1e-10)
    # Whiten: subtract a wide moving average along frequency. This kills the
    # slow spectral envelope (pitch/formants) so the remaining structure is
    # comb-like — exactly where an echo path shows up as a sharp cepstral peak.
    bin_hz = sr / len(win)
    k_wide = max(1, int(400.0 / bin_hz))          # ~400 Hz envelope window
    # edge-pad so the moving average doesn't collapse at the array borders
    # (zero padding would create artificial whitened spikes -> fake cepstral
    # peaks at tiny quefrencies)
    padded = np.pad(logmag, k_wide, mode="edge")
    kernel_w = np.ones(2 * k_wide + 1) / (2 * k_wide + 1)
    whitened = logmag - np.convolve(padded, kernel_w, mode="valid")
    # light smoothing to merge residual harmonics
    k = max(1, int(FREQ_SMOOTH_HZ / bin_hz))
    kernel_s = np.ones(2 * k + 1) / (2 * k + 1)
    whitened = np.convolve(whitened, kernel_s, mode="same")
    return np.fft.irfft(whitened)


def analyze(x: np.ndarray, sr: int) -> dict:
    """Return {'delayed_copy': 0..1, 'smearing': 0..1, 'delay_ms': float}."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < int(WINDOW_SECONDS * sr):
        return {"delayed_copy": 0.0, "smearing": 0.0, "delay_ms": 0.0, "windows": 0}
    starts = _active_starts(x, sr)
    if len(starts) < MIN_WINDOWS:
        return {"delayed_copy": 0.0, "smearing": 0.0, "delay_ms": 0.0, "windows": 0}
    win_len = int(WINDOW_SECONDS * sr)
    cepts = np.stack([_window_cepstrum(x[s:s + win_len], sr) for s in starts])
    med = np.median(cepts, axis=0)

    lo = int(MIN_DELAY_MS / 1000 * sr)
    hi = int(MAX_DELAY_MS / 1000 * sr)
    band = med[lo:hi]
    if band.size == 0 or not np.all(np.isfinite(band)):
        return {"delayed_copy": 0.0, "smearing": 0.0, "delay_ms": 0.0, "windows": len(starts)}
    # peak *prominence* against the band's own statistics (baseline-robust)
    med_b = float(np.median(band))
    mad = float(np.median(np.abs(band - med_b))) + 1e-12
    peak_i = int(np.argmax(np.abs(band)))
    peak = float(abs(band[peak_i]) - med_b)
    delayed_copy = min(1.0, max(0.0, peak / (mad * 25.0)))
    thr = mad * 6.0
    smearing = min(1.0, float((np.abs(band - med_b) > thr).mean()) * 5.0)
    return {
        "delayed_copy": round(delayed_copy, 4),
        "smearing": round(smearing, 4),
        "delay_ms": round((lo + peak_i) / sr * 1000, 1),
        "windows": len(starts),
    }


def echo_score(x: np.ndarray, sr: int) -> float:
    d = analyze(x, sr)
    return max(d["delayed_copy"], d["smearing"])


def passes_gate(x: np.ndarray, sr: int) -> bool:
    d = analyze(x, sr)
    return d["delayed_copy"] <= DELAYED_COPY_PASS and d["smearing"] <= SMEARING_PASS


if __name__ == "__main__":
    import os

    import soundfile as sf

    for path in sys.argv[1:]:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        d = analyze(wav, sr)
        ok = d["delayed_copy"] <= DELAYED_COPY_PASS and d["smearing"] <= SMEARING_PASS
        print(f"{os.path.basename(path):40s} delayed={d['delayed_copy']:.3f} "
              f"smear={d['smearing']:.3f} delay={d['delay_ms']:.0f}ms win={d['windows']} -> {'PASS' if ok else 'FAIL'}")