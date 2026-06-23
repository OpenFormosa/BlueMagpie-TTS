"""Runtime capability detection for the serving engine.

Every accelerated path the engine can take is *optional* and *auto-detected* so
the same code runs on CUDA, Apple-Silicon/MPS, and CPU with no extra
dependencies. Nothing here imports a CUDA-only package at module load; probes
are cheap and cached.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def has_cuda() -> bool:
    return torch.cuda.is_available()


@functools.lru_cache(maxsize=None)
def has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@functools.lru_cache(maxsize=None)
def cuda_graphs_available() -> bool:
    """CUDA graph capture is only meaningful on a real CUDA device."""
    return has_cuda() and hasattr(torch.cuda, "CUDAGraph")


@functools.lru_cache(maxsize=None)
def mamba_ssm_available() -> bool:
    """The optional mamba_ssm selective-scan kernel (CUDA/Triton only).

    Barbet already ships a pure-PyTorch selective-scan fallback, so this only
    gates the faster kernel path; it is never required.
    """
    if not has_cuda():
        return False
    try:
        import mamba_ssm.ops.triton.ssd_combined  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=None)
def sdpa_supports_gqa() -> bool:
    """Whether F.scaled_dot_product_attention accepts ``enable_gqa=``.

    Added in torch 2.5. When absent we repeat_interleave the KV heads by hand,
    which is correct on every device/version.
    """
    import inspect

    try:
        sig = inspect.signature(torch.nn.functional.scaled_dot_product_attention)
    except (ValueError, TypeError):
        return False
    return "enable_gqa" in sig.parameters


def device_type(device) -> str:
    """Normalize a device / device-string to its type ('cuda'/'mps'/'cpu')."""
    return torch.device(device).type


def supports_low_precision(device) -> bool:
    """MPS still glitches the diffusion AR loop in bf16/fp16 (see voxcpm utils).

    The engine keeps float32 on MPS unless the user opts in; CUDA/CPU keep the
    checkpoint dtype.
    """
    return device_type(device) != "mps"
