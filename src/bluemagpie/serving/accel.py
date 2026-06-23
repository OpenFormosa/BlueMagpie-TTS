"""Optional CUDA-graph / torch.compile acceleration for the serving engine.

``torch.compile(mode="reduce-overhead")`` captures CUDA graphs internally, so it
is the robust way to get the CUDA-graph launch-overhead win without hand-rolling
capture/replay — this is exactly what VoxCPM2's own ``optimize()`` does.

We compile only the FLOP-heavy, shape-stable modules:
- ``feat_decoder.estimator`` — the LocDiT transformer (the per-patch FLOP
  dominator; called ~inference_timesteps×2 times per patch at a fixed shape), and
- ``feat_encoder`` — the LocEnc, run once per patch at a fixed shape.

The autoregressive LM decode kernels are intentionally NOT compiled: their KV
length grows every step and they read it host-side (``int(positions.max())``),
which forces a graph break / recompile. Compiling them cleanly needs a
fixed-shape / on-device-length refactor (a documented follow-up).

Everything here is **hard-gated on CUDA** and a no-op on MPS/CPU, so importing
and constructing the engine never requires CUDA.
"""

from __future__ import annotations

import sys

from . import _caps


def optimize_for_inference(model, *, compile_dit: bool = True, compile_encoder: bool = True,
                           mode: str = "reduce-overhead") -> bool:
    """In-place ``torch.compile`` the hot modules. Returns whether it ran.

    No-op (returns ``False``) off CUDA, since ``reduce-overhead`` relies on CUDA
    graphs. Mutates ``model.feat_decoder.estimator`` / ``model.feat_encoder`` in
    place, so the model's own ``generate`` benefits too.
    """
    if not _caps.has_cuda() or _caps.device_type(model._runtime_device()) != "cuda":
        return False
    import torch

    try:
        if compile_encoder:
            model.feat_encoder = torch.compile(model.feat_encoder, mode=mode, fullgraph=False)
        if compile_dit:
            model.feat_decoder.estimator = torch.compile(model.feat_decoder.estimator, mode=mode, fullgraph=False)
    except Exception as e:  # torch.compile is best-effort; never fail generation over it
        print(f"[bluemagpie.serving] torch.compile disabled: {e}", file=sys.stderr)
        return False
    return True
