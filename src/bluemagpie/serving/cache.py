"""Per-row batched caches for the serving engine.

Unlike the model's ``BarbetCache`` / ``StaticKVCache`` — whose ``seen_tokens`` /
``current_length`` are *scalars* shared across the whole batch — these caches
key every piece of state by a per-row **slot** so sequences admitted at
different times (continuous batching) coexist at different absolute positions.

- :class:`BatchedKVCache` stores attention K/V at **absolute positions** in a
  padded ``[layers, max_seqs, n_kv_heads, max_len, head_dim]`` buffer. No
  rolling-window trim: the per-row additive mask (built by the kernel) enforces
  causality *and* the sliding-window floor, which is mathematically identical to
  Barbet's trimmed window but lets us keep untrimmed K/V we wrote ourselves.
- :class:`BatchedMambaState` stores the per-row causal-conv tail and SSM state
  for Barbet's Mamba2 layers (constant shape — no positions involved).
- :class:`SlotManager` is a free-list of row indices.

Buffers are allocated in the model's runtime dtype/device (e.g. float32 on MPS),
so nothing here introduces a device- or precision-specific code path.
"""

from __future__ import annotations

from typing import Optional

import torch


class SlotManager:
    """Free-list of cache row indices in ``[0, max_seqs)``."""

    def __init__(self, max_seqs: int) -> None:
        self.max_seqs = max_seqs
        self._free = list(range(max_seqs))

    def acquire(self) -> int:
        if not self._free:
            raise RuntimeError(f"no free cache slot (max_num_seqs={self.max_seqs} reached)")
        return self._free.pop(0)

    def release(self, slot: int) -> None:
        # Keep the free-list ordered/unique so a released slot is reusable but
        # never double-freed.
        if slot in self._free:
            raise RuntimeError(f"slot {slot} double-freed")
        self._free.append(slot)
        self._free.sort()

    @property
    def num_free(self) -> int:
        return len(self._free)


class BatchedKVCache:
    """Absolute-position padded K/V for one set of attention layers.

    Shapes: ``k``/``v`` are ``[num_layers, max_seqs, n_kv_heads, max_len,
    head_dim]``. Writes scatter one token per active row at its own position;
    reads gather the valid prefix for the active rows.
    """

    def __init__(
        self,
        num_layers: int,
        max_seqs: int,
        n_kv_heads: int,
        head_dim: int,
        max_len: int,
        device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.max_len = max_len
        shape = (num_layers, max_seqs, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def write(self, layer: int, slots: torch.Tensor, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter ``k``/``v`` (``[n_active, n_kv_heads, head_dim]``) at each row's position."""
        # Paired advanced indices (slots, positions) select one (row, pos) per
        # active sequence; the ``:`` axes keep n_kv_heads/head_dim.
        self.k[layer, slots, :, positions, :] = k
        self.v[layer, slots, :, positions, :] = v

    def read(self, layer: int, slots: torch.Tensor, kv_len: int):
        """Gather ``[n_active, n_kv_heads, kv_len, head_dim]`` for the active rows."""
        k = self.k[layer, slots, :, :kv_len, :]
        v = self.v[layer, slots, :, :kv_len, :]
        return k, v


class BatchedMambaState:
    """Per-row causal-conv tail + SSM state for Barbet's Mamba2 layers."""

    def __init__(
        self,
        num_layers: int,
        max_seqs: int,
        conv_channels: int,
        d_conv: int,
        n_heads: int,
        head_dim: int,
        d_state: int,
        device,
        dtype: torch.dtype,
    ) -> None:
        self.conv = torch.zeros(num_layers, max_seqs, conv_channels, d_conv, device=device, dtype=dtype)
        self.ssm = torch.zeros(num_layers, max_seqs, n_heads, head_dim, d_state, device=device, dtype=dtype)

    def reset_slot(self, slot: int) -> None:
        """Zero a row's state on (re)admission so a reused slot starts clean."""
        self.conv[:, slot].zero_()
        self.ssm[:, slot].zero_()


def build_allowed_mask(positions: torch.Tensor, kv_len: int, window: Optional[int]) -> torch.Tensor:
    """Per-row boolean attention mask ``[n_active, kv_len]`` (True == attend).

    A query at absolute ``positions[r]`` attends to keys ``[0, positions[r]]``
    (causal + valid prefix); a sliding layer additionally floors at
    ``positions[r] - window + 1``. The kernel applies this with
    ``masked_fill(~allowed, finfo.min)`` exactly like ``BarbetAttention.forward``,
    which is mathematically identical to the model's trimmed sliding window.
    """
    idx = torch.arange(kv_len, device=positions.device)
    allowed = idx[None, :] <= positions[:, None]
    if window is not None and window > 0:
        allowed = allowed & (idx[None, :] >= (positions[:, None] - window + 1))
    return allowed
