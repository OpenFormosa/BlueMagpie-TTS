"""Batched streaming AudioVAE decode for the serving engine.

The vendored ``StreamingVAEDecoder`` keeps causal-convolution state per decoder
module (keyed by ``id(mod)``) with a **leading batch dim** — so feeding it a
batched ``[N, D, patch]`` latent already gives each row an independent causal
state. The only thing missing for continuous batching is keeping that batch dim
aligned to the *current* active-row set: rows that finish are sliced out, rows
that join get a fresh zero state row appended (identical to the decoder's
cold-start zero padding). :class:`BatchedStreamingVAE` does exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class StreamChunk:
    """One streamed step for one request."""

    request_id: int
    latents: torch.Tensor             # [p, d] newly generated patch
    audio: Optional[torch.Tensor] = None   # [chunk] 48 kHz waveform (if VAE present)
    finished: bool = False
    sample_rate: Optional[int] = None


class BatchedStreamingVAE:
    """Drives one streaming AudioVAE decode across a dynamic batch of rows."""

    def __init__(self, vae) -> None:
        self.dec = vae.streaming_decode()
        self.dec.__enter__()
        self._order: Optional[List] = None

    def close(self) -> None:
        self.dec.__exit__(None, None, None)

    def _reconcile(self, order: Sequence) -> None:
        """Realign every per-module conv state's batch dim to ``order``.

        ``order`` is a sequence of stable per-row keys (cache slots). Rows absent
        from the new order are dropped; new rows get a zero state row (the same
        cold start the decoder uses when a key is first seen).
        """
        states = self.dec._states
        if states and self._order is not None:
            old = {k: i for i, k in enumerate(self._order)}
            for key, st in list(states.items()):
                rows = []
                for o in order:
                    i = old.get(o)
                    if i is not None and i < st.shape[0]:
                        rows.append(st[i : i + 1])
                    else:
                        rows.append(torch.zeros(1, st.shape[1], st.shape[2], device=st.device, dtype=st.dtype))
                states[key] = torch.cat(rows, dim=0) if rows else st[:0]
        self._order = list(order)

    def decode(self, order: Sequence, z: torch.Tensor) -> torch.Tensor:
        """Decode one batched chunk. ``z``: ``[N, D, patch]`` in ``order``.

        Returns audio ``[N, 1, chunk]`` (one row per ``order`` entry).
        """
        self._reconcile(order)
        return self.dec.decode_chunk(z)
