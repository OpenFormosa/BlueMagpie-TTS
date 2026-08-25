"""Content-free, request-local latency instrumentation.

The production Space emits one fixed-schema latency line per public synthesis
request.  Only stage names, elapsed seconds, and call counts are retained; no
request text, waveform, transcript, seed, or exception data enters this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Callable, Iterator


LATENCY_STAGE_NAMES = (
    "request_total",
    "generation",
    "breeze25_asr",
    "ecapa",
    "squim",
    "echo",
    "f0",
    "assemble",
)
_TIMED_STAGE_NAMES = frozenset(LATENCY_STAGE_NAMES[1:])
LATENCY_LOG_PREFIX = "[BlueMagpie] latency "


@dataclass
class RequestLatency:
    """Mutable request-local totals with a fixed, content-free schema."""

    seconds: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in LATENCY_STAGE_NAMES}
    )
    calls: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in LATENCY_STAGE_NAMES}
    )
    active_depths: dict[str, int] = field(default_factory=dict)

    def add(self, stage: str, elapsed_seconds: float) -> None:
        if stage not in LATENCY_STAGE_NAMES:
            raise ValueError("unknown latency stage")
        self.seconds[stage] += max(0.0, float(elapsed_seconds))
        self.calls[stage] += 1


_CURRENT_REQUEST: ContextVar[RequestLatency | None] = ContextVar(
    "bluemagpie_current_request_latency",
    default=None,
)


def format_latency_log(totals: RequestLatency) -> str:
    """Return the fixed-schema log line without accepting request content."""

    fields: list[str] = []
    for stage in LATENCY_STAGE_NAMES:
        fields.append(f"{stage}_seconds={totals.seconds[stage]:.6f}")
        fields.append(f"{stage}_calls={totals.calls[stage]}")
    return LATENCY_LOG_PREFIX + " ".join(fields)


@contextmanager
def latency_request(
    *,
    emit: Callable[[str], object] | None = None,
) -> Iterator[RequestLatency]:
    """Open one request timer and emit exactly once at the outermost boundary."""

    existing = _CURRENT_REQUEST.get()
    if existing is not None:
        yield existing
        return

    totals = RequestLatency()
    token = _CURRENT_REQUEST.set(totals)
    started = perf_counter()
    try:
        yield totals
    finally:
        totals.add("request_total", perf_counter() - started)
        _CURRENT_REQUEST.reset(token)
        (emit or print)(format_latency_log(totals))


@contextmanager
def latency_stage(stage: str) -> Iterator[None]:
    """Accumulate one stage call when a public request timer is active.

    Re-entering the same stage is deliberately folded into its outer call.  It
    keeps a high-level ECAPA operation from double-counting a nested lazy load
    or lower-level speaker measurement.
    """

    if stage not in _TIMED_STAGE_NAMES:
        raise ValueError("unknown latency stage")
    totals = _CURRENT_REQUEST.get()
    if totals is None:
        yield
        return

    depth = totals.active_depths.get(stage, 0)
    totals.active_depths[stage] = depth + 1
    if depth:
        try:
            yield
        finally:
            totals.active_depths[stage] -= 1
        return

    started = perf_counter()
    try:
        yield
    finally:
        totals.add(stage, perf_counter() - started)
        totals.active_depths.pop(stage, None)


def timed_latency_stage(stage: str):
    """Decorate a synchronous function as one fixed latency stage."""

    if stage not in _TIMED_STAGE_NAMES:
        raise ValueError("unknown latency stage")

    def decorate(function):
        @wraps(function)
        def timed(*args, **kwargs):
            with latency_stage(stage):
                return function(*args, **kwargs)

        return timed

    return decorate


def timed_latency_request(function):
    """Decorate one public synchronous synthesis entry point."""

    @wraps(function)
    def timed(*args, **kwargs):
        with latency_request():
            return function(*args, **kwargs)

    return timed
