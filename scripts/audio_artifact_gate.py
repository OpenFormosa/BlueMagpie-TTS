"""Deterministic, no-reference echo and smearing gate.

The gate is intentionally conservative.  It looks for a *time-invariant*
comb/echo signature that survives across several active analysis windows:

* a narrow long-quefrency cepstral peak is evidence of a delayed copy; and
* a dense set of long-quefrency peaks is a proxy for reverberant smearing.

Content-dependent pitch and formants move between windows, while a fixed echo
path does not.  Taking the median cepstrum across windows therefore avoids most
of the false positives produced by a plain waveform autocorrelation.  This is
not a room-acoustics measurement and should be used as a fail-closed candidate
gate, not as a perceptual quality score.

Only NumPy is required and the analysis is deterministic for identical input.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from latency_timing import timed_latency_stage


@dataclass(frozen=True)
class EchoSmearingGateConfig:
    """Configuration for :func:`evaluate_echo_smearing`.

    The defaults reject only pronounced delayed-copy or dense-comb artifacts.
    Scores are normalized to ``[0, 1]`` before the two gate thresholds are
    applied.
    """

    min_sample_rate: int = 8_000
    max_sample_rate: int = 192_000
    min_duration_seconds: float = 1.0
    window_seconds: float = 0.60
    candidate_window_count: int = 27
    max_analysis_windows: int = 9
    min_analysis_windows: int = 3
    min_rms: float = 1.0e-5
    relative_active_rms: float = 0.12
    min_delay_ms: float = 24.0
    max_delay_ms: float = 180.0
    frequency_smoothing_hz: float = 70.0
    # A raw median-cepstral peak ratio of 8.965 maps to the default rejection
    # boundary.  Calibration controls topped out at 6.61, while the deployed
    # echo-like smoke output measured 10.34.
    delayed_ratio_floor: float = 5.5
    delayed_ratio_ceiling: float = 16.0
    smear_q99_ratio_floor: float = 3.0
    smear_q99_ratio_ceiling: float = 6.0
    smear_density_floor: float = 0.008
    smear_density_ceiling: float = 0.035
    delayed_copy_threshold: float = 0.33
    smearing_threshold: float = 0.40


DEFAULT_ECHO_SMEARING_GATE_CONFIG = EchoSmearingGateConfig()


@dataclass(frozen=True)
class EchoSmearingDiagnostics:
    """Finite diagnostics and the fail-closed gate decision."""

    passed: bool
    rejection_reasons: tuple[str, ...]
    delayed_copy_score: float
    smearing_score: float
    dominant_delay_ms: float
    cepstral_peak_ratio: float
    cepstral_q99_ratio: float
    cepstral_dense_fraction: float
    duration_seconds: float
    input_rms: float
    analysis_window_count: int


def _bounded(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value) or upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _finite_nonnegative(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(converted) or converted < 0.0:
        return 0.0
    return converted


def _rejected(
    reason: str,
    *,
    duration_seconds: float = 0.0,
    input_rms: float = 0.0,
) -> EchoSmearingDiagnostics:
    return EchoSmearingDiagnostics(
        passed=False,
        rejection_reasons=(reason,),
        delayed_copy_score=0.0,
        smearing_score=0.0,
        dominant_delay_ms=0.0,
        cepstral_peak_ratio=0.0,
        cepstral_q99_ratio=0.0,
        cepstral_dense_fraction=0.0,
        duration_seconds=_finite_nonnegative(duration_seconds),
        input_rms=_finite_nonnegative(input_rms),
        analysis_window_count=0,
    )


def _valid_config(config: EchoSmearingGateConfig) -> bool:
    integer_fields = (
        config.min_sample_rate,
        config.max_sample_rate,
        config.candidate_window_count,
        config.max_analysis_windows,
        config.min_analysis_windows,
    )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in integer_fields
    ):
        return False
    if not (
        1 <= config.min_sample_rate <= config.max_sample_rate
        and config.candidate_window_count >= config.max_analysis_windows
        and config.max_analysis_windows >= config.min_analysis_windows >= 1
    ):
        return False

    finite_fields = (
        config.min_duration_seconds,
        config.window_seconds,
        config.min_rms,
        config.relative_active_rms,
        config.min_delay_ms,
        config.max_delay_ms,
        config.frequency_smoothing_hz,
        config.delayed_ratio_floor,
        config.delayed_ratio_ceiling,
        config.smear_q99_ratio_floor,
        config.smear_q99_ratio_ceiling,
        config.smear_density_floor,
        config.smear_density_ceiling,
        config.delayed_copy_threshold,
        config.smearing_threshold,
    )
    try:
        finite = all(math.isfinite(float(value)) for value in finite_fields)
    except (TypeError, ValueError, OverflowError):
        return False
    if not finite:
        return False
    return bool(
        config.min_duration_seconds > 0.0
        and config.window_seconds > 0.0
        and config.min_rms > 0.0
        and 0.0 < config.relative_active_rms <= 1.0
        and 0.0 < config.min_delay_ms < config.max_delay_ms
        and config.frequency_smoothing_hz > 0.0
        and config.delayed_ratio_floor < config.delayed_ratio_ceiling
        and config.smear_q99_ratio_floor < config.smear_q99_ratio_ceiling
        and config.smear_density_floor < config.smear_density_ceiling
        and 0.0 <= config.delayed_copy_threshold <= 1.0
        and 0.0 <= config.smearing_threshold <= 1.0
    )


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Return an edge-padded centered moving average in linear time."""

    width = max(1, min(int(width), int(values.size)))
    if width % 2 == 0:
        width = max(1, width - 1)
    if width == 1:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(padded, dtype=np.float64))
    )
    return (cumulative[width:] - cumulative[:-width]) / float(width)


def _analysis_starts(
    waveform: np.ndarray,
    window_samples: int,
    config: EchoSmearingGateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Select deterministic high-energy windows distributed over the input."""

    last_start = waveform.size - window_samples
    candidate_count = min(
        config.candidate_window_count,
        max(1, last_start // max(1, window_samples // 3) + 1),
    )
    starts = np.unique(
        np.linspace(0, last_start, num=candidate_count, dtype=np.int64)
    )
    rms_values_list: list[float] = []
    for start in starts:
        window = np.asarray(
            waveform[int(start) : int(start) + window_samples],
            dtype=np.float64,
        )
        centered = window - float(np.mean(window))
        rms_values_list.append(
            math.sqrt(
                float(
                    np.mean(
                        np.square(centered, dtype=np.float64),
                        dtype=np.float64,
                    )
                )
            )
        )
    rms_values = np.asarray(rms_values_list, dtype=np.float64)
    active_floor = max(config.min_rms, config.relative_active_rms * float(rms_values.max()))
    active_indices = np.flatnonzero(rms_values >= active_floor)
    if active_indices.size > config.max_analysis_windows:
        # Keep the strongest windows.  Sorting their positions afterwards
        # makes the output independent of NumPy's tie ordering.
        ranked = sorted(
            active_indices.tolist(),
            key=lambda index: (-float(rms_values[index]), int(starts[index])),
        )
        active_indices = np.asarray(
            sorted(ranked[: config.max_analysis_windows]), dtype=np.int64
        )
    return starts[active_indices], rms_values[active_indices]


def _window_cepstrum(
    window: np.ndarray,
    sample_rate: int,
    config: EchoSmearingGateConfig,
    max_delay_samples: int,
) -> np.ndarray:
    centered = np.asarray(window, dtype=np.float64) - float(np.mean(window))
    emphasized = np.empty_like(centered)
    emphasized[0] = centered[0]
    emphasized[1:] = centered[1:] - 0.97 * centered[:-1]
    emphasized *= np.hanning(emphasized.size)

    fft_size = 1 << max(1, (2 * emphasized.size - 1).bit_length())
    magnitude = np.abs(np.fft.rfft(emphasized, n=fft_size))
    magnitude_floor = max(
        np.finfo(np.float64).tiny,
        float(magnitude.max()) * 1.0e-6,
    )
    log_magnitude = np.log(np.maximum(magnitude, magnitude_floor))
    bin_hz = sample_rate / float(fft_size)
    smoothing_bins = max(3, int(round(config.frequency_smoothing_hz / bin_hz)))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1
    residual = log_magnitude - _moving_average(log_magnitude, smoothing_bins)
    residual -= float(np.mean(residual))
    cepstrum = np.abs(np.fft.irfft(residual, n=fft_size))
    return np.asarray(cepstrum[: max_delay_samples + 1], dtype=np.float64)


@timed_latency_stage("echo")
def evaluate_echo_smearing(
    audio: Any,
    sample_rate: Any,
    *,
    config: EchoSmearingGateConfig = DEFAULT_ECHO_SMEARING_GATE_CONFIG,
) -> EchoSmearingDiagnostics:
    """Measure delayed-copy and smearing proxies and fail closed.

    Invalid, silent, clipped-to-nonfinite, too-short, or analytically
    insufficient inputs return a rejected result rather than raising.  Every
    floating-point field in the result is finite.
    """

    if not isinstance(config, EchoSmearingGateConfig) or not _valid_config(config):
        return _rejected("invalid_config")
    if isinstance(sample_rate, (bool, np.bool_)):
        return _rejected("invalid_sample_rate")
    try:
        rate = operator.index(sample_rate)
    except (TypeError, ValueError, OverflowError):
        return _rejected("invalid_sample_rate")
    rate = int(rate)
    if not config.min_sample_rate <= rate <= config.max_sample_rate:
        return _rejected("invalid_sample_rate")

    try:
        waveform = np.asarray(audio)
    except (TypeError, ValueError, OverflowError):
        return _rejected("invalid_audio")
    if (
        waveform.ndim != 1
        or waveform.size == 0
        or waveform.dtype.kind not in "fiu"
    ):
        return _rejected("invalid_audio")
    try:
        waveform = waveform.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError):
        return _rejected("invalid_audio")
    duration = float(waveform.size / rate)
    if not np.isfinite(waveform).all():
        return _rejected("nonfinite_audio", duration_seconds=duration)
    input_rms = math.sqrt(
        float(np.mean(np.square(waveform, dtype=np.float64), dtype=np.float64))
    )
    if not math.isfinite(input_rms) or input_rms < config.min_rms:
        return _rejected(
            "silent_audio",
            duration_seconds=duration,
            input_rms=input_rms,
        )
    if duration < config.min_duration_seconds:
        return _rejected(
            "insufficient_duration",
            duration_seconds=duration,
            input_rms=input_rms,
        )

    window_samples = max(8, int(round(config.window_seconds * rate)))
    min_delay_samples = max(1, int(round(config.min_delay_ms * rate / 1000.0)))
    max_delay_samples = int(round(config.max_delay_ms * rate / 1000.0))
    if (
        waveform.size < window_samples
        or max_delay_samples <= min_delay_samples
        or max_delay_samples >= window_samples // 2
    ):
        return _rejected(
            "invalid_analysis_geometry",
            duration_seconds=duration,
            input_rms=input_rms,
        )

    starts, _window_rms = _analysis_starts(waveform, window_samples, config)
    if starts.size < config.min_analysis_windows:
        return _rejected(
            "insufficient_active_windows",
            duration_seconds=duration,
            input_rms=input_rms,
        )

    cepstra = np.asarray(
        [
            _window_cepstrum(
                waveform[int(start) : int(start) + window_samples],
                rate,
                config,
                max_delay_samples,
            )
            for start in starts
        ],
        dtype=np.float64,
    )
    aggregate = np.median(
        cepstra[:, min_delay_samples : max_delay_samples + 1],
        axis=0,
    )
    if aggregate.size == 0 or not np.isfinite(aggregate).all():
        return _rejected(
            "nonfinite_analysis",
            duration_seconds=duration,
            input_rms=input_rms,
        )

    baseline = max(float(np.median(aggregate)), np.finfo(np.float64).eps)
    peak_index = int(np.argmax(aggregate))
    peak_ratio = float(aggregate[peak_index] / baseline)
    q99_ratio = float(np.percentile(aggregate, 99.0) / baseline)
    dense_fraction = float(np.mean(aggregate > (3.0 * baseline)))
    dominant_delay_ms = float(
        (min_delay_samples + peak_index) * 1000.0 / rate
    )
    if not all(
        math.isfinite(value)
        for value in (
            peak_ratio,
            q99_ratio,
            dense_fraction,
            dominant_delay_ms,
        )
    ):
        return _rejected(
            "nonfinite_analysis",
            duration_seconds=duration,
            input_rms=input_rms,
        )

    delayed_score = _bounded(
        peak_ratio,
        config.delayed_ratio_floor,
        config.delayed_ratio_ceiling,
    )
    q99_component = _bounded(
        q99_ratio,
        config.smear_q99_ratio_floor,
        config.smear_q99_ratio_ceiling,
    )
    density_component = _bounded(
        dense_fraction,
        config.smear_density_floor,
        config.smear_density_ceiling,
    )
    smearing_score = float(math.sqrt(q99_component * density_component))

    reasons: list[str] = []
    if delayed_score >= config.delayed_copy_threshold:
        reasons.append("delayed_copy")
    if smearing_score >= config.smearing_threshold:
        reasons.append("smearing")
    return EchoSmearingDiagnostics(
        passed=not reasons,
        rejection_reasons=tuple(reasons),
        delayed_copy_score=_finite_nonnegative(delayed_score),
        smearing_score=_finite_nonnegative(smearing_score),
        dominant_delay_ms=_finite_nonnegative(dominant_delay_ms),
        cepstral_peak_ratio=_finite_nonnegative(peak_ratio),
        cepstral_q99_ratio=_finite_nonnegative(q99_ratio),
        cepstral_dense_fraction=_finite_nonnegative(dense_fraction),
        duration_seconds=_finite_nonnegative(duration),
        input_rms=_finite_nonnegative(input_rms),
        analysis_window_count=int(starts.size),
    )


__all__ = [
    "DEFAULT_ECHO_SMEARING_GATE_CONFIG",
    "EchoSmearingDiagnostics",
    "EchoSmearingGateConfig",
    "evaluate_echo_smearing",
]
