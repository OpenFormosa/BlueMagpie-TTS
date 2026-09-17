"""Pitch-preserving pace control (official-demo contract).

- count_speech_units(): CJK chars count 1, ASCII digit runs compress by 2,
  other ASCII alnum runs by 4 (mirrors production.count_speech_units).
- target_pace_speed(): only ever SLOWS audio down toward TARGET_CPS
  (= 4.0 units/sec), floored at MIN_PACE_SPEED (= 0.90).
- wsola_time_stretch(): the official Space's numpy-only WSOLA engine.
"""
import math

import numpy as np

from .audio_wsola import wsola_time_stretch

TARGET_CPS = 4.0
MIN_PACE_SPEED = 0.90


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _normalize_light(text: str) -> str:
    return " ".join(str(text).split()).lower()


def count_speech_units(text: str) -> int:
    """CJK characters and compressed ASCII runs (official convention)."""
    units = 0
    ascii_buffer: list = []

    def flush_ascii():
        nonlocal units
        if not ascii_buffer:
            return
        token = "".join(ascii_buffer)
        divisor = 2 if token.isdigit() else 4
        units += max(1, math.ceil(len(token) / divisor))
        ascii_buffer.clear()

    for char in _normalize_light(text):
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
        elif _is_cjk(char):
            flush_ascii()
            units += 1
        else:
            flush_ascii()
    flush_ascii()
    return units


def target_pace_speed(audio_samples: int, sample_rate: int, text: str, *,
                      target_cps: float = TARGET_CPS,
                      min_speed: float = MIN_PACE_SPEED) -> float:
    """Stretch rate <= 1.0 that slows over-fast speech to the target pace."""
    units = count_speech_units(text)
    if audio_samples <= 0 or sample_rate <= 0 or units <= 0 or target_cps <= 0.0:
        return 1.0
    actual_seconds = float(audio_samples) / float(sample_rate)
    target_seconds = float(units) / float(target_cps)
    if actual_seconds >= target_seconds:
        return 1.0
    return min(1.0, max(float(min_speed), actual_seconds / target_seconds))


def match_pace(audio: np.ndarray, sample_rate: int, text: str, *,
               target_cps: float = TARGET_CPS) -> np.ndarray:
    """Return ``audio`` time-stretched so pace approaches ``target_cps``.

    Only slowing is applied (the official demo never speeds generation up);
    rates are bounded to [MIN_PACE_SPEED, 1.0].
    """
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    rate = target_pace_speed(signal.size, sample_rate, text, target_cps=target_cps)
    if rate >= 0.999:
        return signal
    stretched = wsola_time_stretch(signal, rate=rate, sample_rate=sample_rate)
    return np.asarray(stretched, dtype=np.float32)
