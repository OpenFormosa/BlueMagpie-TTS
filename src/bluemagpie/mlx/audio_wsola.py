"""Deterministic mono WSOLA time-scale modification for production speech.

The implementation deliberately has no optional native dependency.  It keeps
pitch by moving and overlap-adding waveform frames in the time domain, while a
bounded similarity search aligns each new frame to the already synthesized
waveform.  The public ``rate`` follows the librosa convention: values below
one make audio longer and values above one make it shorter.
"""

from __future__ import annotations

import math
import operator
from typing import Sequence

import numpy as np


WSOLA_MIN_RATE = 0.50
WSOLA_MAX_RATE = 2.00
WSOLA_FRAME_MS = 40.0
WSOLA_SYNTHESIS_HOP_MS = 10.0
WSOLA_SEARCH_MS = 6.0
WSOLA_MIN_SAMPLES = 256
WSOLA_CORRELATION_FLOOR = 0.05


def _positive_sample_count(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def _finite_positive(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _mono_waveform(audio: np.ndarray | Sequence[float]) -> np.ndarray:
    waveform = np.asarray(audio)
    if waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1:
        raise ValueError("audio must be mono")
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if waveform.size < WSOLA_MIN_SAMPLES:
        raise ValueError("audio is too short for WSOLA")
    if not np.isfinite(waveform).all():
        raise ValueError("audio must contain only finite samples")
    return waveform


def _frame_positions(target_length: int, frame_length: int, hop: int) -> tuple[int, ...]:
    final_start = target_length - frame_length
    if final_start <= 0:
        return (0,)
    positions = list(range(0, final_start + 1, hop))
    if positions[-1] != final_start:
        positions.append(final_start)
    return tuple(positions)


def _best_source_position(
    signal: np.ndarray,
    synthesized: np.ndarray,
    weight: np.ndarray,
    *,
    output_start: int,
    current_end: int,
    predicted_start: int,
    previous_start: int,
    predicted_step: int,
    frame_length: int,
    search_radius: int,
) -> int:
    """Return a monotonic, waveform-similar source position."""

    source_limit = signal.size - frame_length
    lower = max(
        0,
        predicted_start - search_radius,
        previous_start + max(1, int(round(0.35 * predicted_step))),
    )
    upper = min(source_limit, predicted_start + search_radius)
    if lower > upper:
        return int(np.clip(predicted_start, 0, source_limit))

    overlap = min(frame_length, max(0, current_end - output_start))
    if overlap < 32 or lower == upper:
        return int(np.clip(predicted_start, lower, upper))

    overlap_weights = weight[output_start : output_start + overlap]
    usable = overlap_weights > 1.0e-4
    if int(np.count_nonzero(usable)) < 32:
        return int(np.clip(predicted_start, lower, upper))
    first = int(np.flatnonzero(usable)[0])
    last = int(np.flatnonzero(usable)[-1]) + 1
    if last - first < 32:
        return int(np.clip(predicted_start, lower, upper))

    reference = (
        synthesized[output_start + first : output_start + last]
        / np.maximum(
            weight[output_start + first : output_start + last],
            np.float32(1.0e-7),
        )
    )
    # Pre-emphasis makes the search follow speech fine structure rather than
    # matching only the slowly changing loudness envelope.
    reference = np.diff(reference, prepend=reference[0]).astype(
        np.float64,
        copy=False,
    )
    reference -= float(np.mean(reference))
    reference_norm = float(np.linalg.norm(reference))
    if not math.isfinite(reference_norm) or reference_norm <= 1.0e-7:
        return int(np.clip(predicted_start, lower, upper))

    segment_length = last - first
    search = signal[
        lower + first : upper + first + segment_length
    ].astype(np.float64, copy=False)
    search = np.diff(search, prepend=search[0])
    candidate_count = upper - lower + 1
    if search.size < segment_length + candidate_count - 1:
        return int(np.clip(predicted_start, lower, upper))

    dots = np.correlate(search, reference, mode="valid")
    if dots.size != candidate_count:
        return int(np.clip(predicted_start, lower, upper))
    squared = np.square(search, dtype=np.float64)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64))
    )
    norms = np.sqrt(
        np.maximum(
            cumulative[segment_length:] - cumulative[:-segment_length],
            0.0,
        )
    )
    scores = dots / np.maximum(norms * reference_norm, 1.0e-12)
    finite = np.isfinite(scores)
    if not finite.any():
        return int(np.clip(predicted_start, lower, upper))
    scores = np.where(finite, scores, -np.inf)
    best_offset = int(np.argmax(scores))
    if float(scores[best_offset]) < WSOLA_CORRELATION_FLOOR:
        return int(np.clip(predicted_start, lower, upper))
    return lower + best_offset


def wsola_time_stretch(
    audio: np.ndarray | Sequence[float],
    *,
    rate: float,
    sample_rate: int,
    frame_ms: float = WSOLA_FRAME_MS,
    synthesis_hop_ms: float = WSOLA_SYNTHESIS_HOP_MS,
    search_ms: float = WSOLA_SEARCH_MS,
) -> np.ndarray:
    """Return a deterministic, pitch-preserving WSOLA time stretch.

    The output length is exactly ``round(input_samples / rate)``.  Invalid,
    non-finite, multi-channel or pathologically short inputs raise instead of
    falling back to a lower-quality transform.
    """

    waveform = _mono_waveform(audio)
    source_rate = _positive_sample_count(sample_rate, "sample_rate")
    stretch_rate = _finite_positive(rate, "rate")
    if not WSOLA_MIN_RATE <= stretch_rate <= WSOLA_MAX_RATE:
        raise ValueError(
            f"rate must be between {WSOLA_MIN_RATE:.2f} and {WSOLA_MAX_RATE:.2f}"
        )
    frame_duration = _finite_positive(frame_ms, "frame_ms")
    hop_duration = _finite_positive(synthesis_hop_ms, "synthesis_hop_ms")
    search_duration = _finite_positive(search_ms, "search_ms")
    if abs(stretch_rate - 1.0) < 1.0e-3:
        return waveform.copy()

    target_length = max(1, int(round(waveform.size / stretch_rate)))
    minimum_extent = min(waveform.size, target_length)
    if minimum_extent < WSOLA_MIN_SAMPLES:
        raise ValueError("stretched audio is too short for WSOLA")

    default_frame = max(
        64,
        int(round(frame_duration * source_rate / 1000.0)),
    )
    # Keep at least two synthesis frames even for unusually short utterances.
    frame_length = min(default_frame, max(128, minimum_extent // 2))
    frame_length = min(frame_length, minimum_extent)
    default_hop = max(
        16,
        int(round(hop_duration * source_rate / 1000.0)),
    )
    synthesis_hop = min(default_hop, max(16, frame_length // 3))
    if synthesis_hop >= frame_length:
        synthesis_hop = max(1, frame_length // 4)
    search_radius = max(
        1,
        int(round(search_duration * source_rate / 1000.0)),
    )

    output_positions = _frame_positions(
        target_length,
        frame_length,
        synthesis_hop,
    )
    source_limit = waveform.size - frame_length
    output_limit = target_length - frame_length
    window = np.hanning(frame_length + 2)[1:-1].astype(np.float32)
    synthesized = np.zeros(target_length, dtype=np.float32)
    weight = np.zeros(target_length, dtype=np.float32)

    previous_source = -1
    previous_predicted = 0
    current_end = 0
    for frame_index, output_start in enumerate(output_positions):
        if frame_index == 0:
            predicted_source = 0
            source_start = 0
        elif frame_index + 1 == len(output_positions):
            predicted_source = source_limit
            source_start = source_limit
        else:
            predicted_source = int(
                round(output_start * source_limit / max(1, output_limit))
            )
            predicted_step = max(1, predicted_source - previous_predicted)
            source_start = _best_source_position(
                waveform,
                synthesized,
                weight,
                output_start=output_start,
                current_end=current_end,
                predicted_start=predicted_source,
                previous_start=previous_source,
                predicted_step=predicted_step,
                frame_length=frame_length,
                search_radius=search_radius,
            )

        output_end = output_start + frame_length
        source_end = source_start + frame_length
        synthesized[output_start:output_end] += (
            waveform[source_start:source_end] * window
        )
        weight[output_start:output_end] += window
        current_end = max(current_end, output_end)
        previous_source = source_start
        previous_predicted = predicted_source

    valid = weight > 1.0e-7
    if not valid.all():
        raise RuntimeError("WSOLA overlap-add left uncovered output samples")
    output = synthesized / weight
    if not np.isfinite(output).all():
        raise RuntimeError("WSOLA produced non-finite output")
    return np.ascontiguousarray(output, dtype=np.float32)
