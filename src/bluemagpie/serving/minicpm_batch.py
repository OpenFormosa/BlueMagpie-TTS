"""Batched single-step kernel for the RALM (MiniCPM, residual acoustic LM).

The RALM is an 8-layer pure-attention MiniCPM with ``residual_lm_no_rope=True``
(no positional embedding) and ``vocab_size=0`` (identity embedding). This
re-implements one decode step for a batch of rows at ragged positions, mirroring
``MiniCPMAttention.forward_step`` / ``MiniCPMDecoderLayer.forward_step`` but with
per-row positions and our :class:`BatchedKVCache`. Attention uses SDPA (with a
manual GQA expand so we don't depend on torch>=2.5's ``enable_gqa``).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from .cache import BatchedKVCache, build_allowed_mask


class BatchedRALM:
    """Owns the per-row KV cache and runs batched prefill/decode for the RALM."""

    def __init__(self, residual_lm, max_seqs: int, max_len: int, device, dtype: torch.dtype) -> None:
        self.model = residual_lm                      # MiniCPMModel
        self.config = residual_lm.config
        self.layers = list(residual_lm.layers)
        self.device = device
        self.dtype = dtype
        self.no_rope = residual_lm.rope_emb is None

        a = self.layers[0].self_attn
        self.num_heads = a.num_heads
        self.num_kv_heads = a.num_key_value_heads
        self.num_kv_groups = a.num_key_value_groups
        self.head_dim = a.head_dim
        self.kv = BatchedKVCache(
            num_layers=len(self.layers),
            max_seqs=max_seqs,
            n_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_len=max_len,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------ #
    def _attn_step(self, layer_idx: int, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attn = self.layers[layer_idx].self_attn  # MiniCPMAttention
        n = x.shape[0]
        hd = self.head_dim

        q = attn.q_proj(x).view(n, 1, self.num_heads, hd).transpose(1, 2)      # [n, H, 1, d]
        k = attn.k_proj(x).view(n, 1, self.num_kv_heads, hd).transpose(1, 2)
        v = attn.v_proj(x).view(n, 1, self.num_kv_heads, hd).transpose(1, 2)
        # RALM has no_rope -> no positional embedding (position_emb is None).

        self.kv.write(layer_idx, slots, positions, k[:, :, 0, :], v[:, :, 0, :])
        kv_len = int(positions.max().item()) + 1
        K, V = self.kv.read(layer_idx, slots, kv_len)                           # [n, n_kv, kv_len, d]
        K = K.repeat_interleave(self.num_kv_groups, dim=1)                      # portable GQA
        V = V.repeat_interleave(self.num_kv_groups, dim=1)

        allowed = build_allowed_mask(positions, kv_len, window=None)            # causal + valid, no window
        attn_mask = allowed[:, None, None, :]                                   # [n, 1, 1, kv_len] (bool)
        q = q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        out = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask)      # [n, H, 1, d]
        out = out.transpose(1, 2).contiguous().view(n, self.num_heads * hd)
        return attn.o_proj(out)

    def _layer_step(self, layer_idx: int, h: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        layer = self.layers[layer_idx]
        residual = h
        normed = layer.input_layernorm(h)
        mixed = self._attn_step(layer_idx, normed, slots, positions)
        if layer.use_mup:
            h = residual + mixed * (layer.scale_depth / math.sqrt(layer.num_hidden_layers))
        else:
            h = residual + mixed
        residual = h
        normed = layer.post_attention_layernorm(h)
        mlp = layer.mlp(normed)
        if layer.use_mup:
            h = residual + mlp * (layer.scale_depth / math.sqrt(layer.num_hidden_layers))
        else:
            h = residual + mlp
        return h

    # ------------------------------------------------------------------ #
    def decode_step(self, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One RALM step for ``n`` rows. ``x``: [n, H] -> hidden [n, H]."""
        h = x
        for layer_idx in range(len(self.layers)):
            h = self._layer_step(layer_idx, h, slots, positions)
        return self.model.norm(h)

    def prefill(self, embeds: torch.Tensor, slot: int, start: int = 0) -> torch.Tensor:
        """Run a prompt (``[L, H]``) for one row, returning hiddens ``[L, H]``."""
        return self.prefill_batch([embeds], [slot])[0]

    def prefill_batch(self, embeds_list: List[torch.Tensor], slots: List[int]) -> List[torch.Tensor]:
        """Prefill several prompts together, batched per position (see BatchedBarbet)."""
        lengths = [e.shape[0] for e in embeds_list]
        outs: List[List[torch.Tensor]] = [[None] * L for L in lengths]
        t_max = max(lengths) if lengths else 0
        for t in range(t_max):
            act = [k for k in range(len(slots)) if t < lengths[k]]
            x = torch.stack([embeds_list[k][t] for k in act], dim=0)
            slot_t = torch.tensor([slots[k] for k in act], device=self.device, dtype=torch.long)
            pos = torch.full((len(act),), t, device=self.device, dtype=torch.long)
            h = self.decode_step(x, slot_t, pos)
            for j, k in enumerate(act):
                outs[k][t] = h[j]
        return [torch.stack(outs[k], dim=0) for k in range(len(slots))]
