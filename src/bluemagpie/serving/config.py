"""Configuration for the BlueMagpie serving engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import _caps


@dataclass
class EngineConfig:
    """Knobs for :class:`BlueMagpieEngine`.

    Defaults are chosen so the engine is numerically equal to
    ``BlueMagpieModel._inference`` at batch=1: ``enforce_eager=True`` disables
    CUDA graphs / compile, and the generation defaults mirror the shipped
    ``config.json`` (``inference_timesteps=9``, ``cfg_value=2.8``).
    """

    # Batching / memory.
    max_num_seqs: int = 16          # rows in the batched cache (concurrency cap)
    max_model_len: int = 2048       # max prompt+generated positions per sequence

    # Generation defaults (shipped config.json generation_defaults).
    inference_timesteps: int = 9
    cfg_value: float = 2.8
    min_len: int = 2
    max_len: int = 2000

    # Streaming context (mirrors model._generate streaming_prefix_len).
    streaming_prefix_len: int = 4

    # Device / dtype. ``device=None`` -> auto-select (cuda > mps > cpu).
    device: Optional[str] = None
    dtype: Optional[str] = None     # None -> inherit the model's runtime dtype

    # Acceleration toggles. All are auto-gated: nothing CUDA-only ever runs off
    # CUDA regardless of these flags.
    enforce_eager: bool = True      # True => no CUDA graphs / no compile (parity mode)
    enable_cuda_graph: bool = False
    graph_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # torch.compile (mode="reduce-overhead" captures CUDA graphs internally) of the
    # DiT estimator + LocEnc. CUDA-only; no-op on MPS/CPU. See serving/accel.py.
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    # Stop-flag host sync cadence. 1 == sync every step (exact parity with the
    # eager loop). >1 trades a bounded EOS overrun for fewer device syncs.
    stop_sync_every: int = 1

    def resolved_device(self, model_device) -> str:
        """Resolve the engine device, defaulting to the model's device."""
        if self.device is not None:
            return self.device
        return str(model_device)

    def cuda_graph_active(self, device) -> bool:
        """CUDA graphs only when explicitly enabled, not eager, and on CUDA."""
        return (
            self.enable_cuda_graph
            and not self.enforce_eager
            and _caps.cuda_graphs_available()
            and _caps.device_type(device) == "cuda"
        )
