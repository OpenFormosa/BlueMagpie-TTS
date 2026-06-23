"""Batched single-step Barbet kernel for the serving engine.

This re-implements *one decode step* of ``BarbetModel`` for a batch of rows at
**ragged absolute positions**, which the model's own ``forward_step`` /
``BarbetCache`` cannot do (their position is a single scalar). It reuses the
model's own weights and submodules (``q_proj``, ``rotary_emb``, ``conv_x``,
``out_proj`` …) and only re-derives the position/mask/cache plumbing, so it
stays bit-faithful to ``BarbetAttention.forward`` and the ``use_step_cache``
branch of ``BarbetMambaMixer.forward``.

Prefill is just this kernel looped over the prompt embeddings one position at a
time — ``tests/test_step_equivalence.py`` proves stepwise == full forward, so the
loop is provably equal to the model's parallel prefill while keeping the
pure-PyTorch single-step Mamba path as the *only* Mamba path (the CUDA
chunk-scan kernel, with its different final-state layout, is never reached).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from barbet.modeling_barbet import apply_rotary_pos_emb

from .cache import BatchedKVCache, BatchedMambaState, build_allowed_mask


class BatchedBarbet:
    """Owns the per-row caches and runs batched prefill/decode for Barbet."""

    def __init__(self, tslm, max_seqs: int, max_len: int, device, dtype: torch.dtype) -> None:
        self.backbone = tslm.backbone          # BarbetModel
        self.config = tslm.config
        self.device = device
        self.dtype = dtype
        self.max_seqs = max_seqs
        self.max_len = max_len

        layers = list(self.backbone.layers)
        self.layers = layers
        # Map decoder-layer index -> index within the attention / mamba caches.
        self.attn_layer_ids: List[int] = [i for i, lyr in enumerate(layers) if lyr.layer_type != "mamba"]
        self.mamba_layer_ids: List[int] = [i for i, lyr in enumerate(layers) if lyr.layer_type == "mamba"]
        self._attn_slot = {li: k for k, li in enumerate(self.attn_layer_ids)}
        self._mamba_slot = {li: k for k, li in enumerate(self.mamba_layer_ids)}
        # Per-layer sliding window (None for global-attention layers).
        self._window = {
            li: (self.config.sliding_window_size if layers[li].layer_type == "sliding_attention" else None)
            for li in self.attn_layer_ids
        }

        a_mixer = layers[self.attn_layer_ids[0]].mixer
        self.kv = BatchedKVCache(
            num_layers=len(self.attn_layer_ids),
            max_seqs=max_seqs,
            n_kv_heads=a_mixer.num_key_value_heads,
            head_dim=a_mixer.head_dim,
            max_len=max_len,
            device=device,
            dtype=dtype,
        )
        if self.mamba_layer_ids:
            m = layers[self.mamba_layer_ids[0]].mixer
            conv_channels = m.inner_size + 2 * m.num_groups * m.d_state
            self.mamba = BatchedMambaState(
                num_layers=len(self.mamba_layer_ids),
                max_seqs=max_seqs,
                conv_channels=conv_channels,
                d_conv=m.d_conv,
                n_heads=m.num_heads,
                head_dim=m.head_dim,
                d_state=m.d_state,
                device=device,
                dtype=dtype,
            )
        else:
            self.mamba = None

    # ------------------------------------------------------------------ #
    def reset_slot(self, slot: int) -> None:
        """Clear a slot's Mamba state on (re)admission (attn KV is mask-safe)."""
        if self.mamba is not None:
            self.mamba.reset_slot(slot)

    # ------------------------------------------------------------------ #
    def _attn_step(self, layer_idx: int, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        mixer = self.layers[layer_idx].mixer  # BarbetAttention
        cache_layer = self._attn_slot[layer_idx]
        window = self._window[layer_idx]
        n = x.shape[0]
        hd = mixer.head_dim

        # [n, heads, 1, dim] mirroring BarbetAttention._shape with seq_len=1.
        q = mixer.q_proj(x).view(n, 1, mixer.num_heads, hd).transpose(1, 2)
        k = mixer.k_proj(x).view(n, 1, mixer.num_key_value_heads, hd).transpose(1, 2)
        v = mixer.v_proj(x).view(n, 1, mixer.num_key_value_heads, hd).transpose(1, 2)
        q = mixer.q_norm(q)
        k = mixer.k_norm(k)
        cos, sin = mixer.rotary_emb(positions[:, None], q.dtype)  # [n, 1, dim]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Scatter the new K/V at each row's own position, then gather the prefix.
        self.kv.write(cache_layer, slots, positions, k[:, :, 0, :], v[:, :, 0, :])
        kv_len = int(positions.max().item()) + 1
        K, V = self.kv.read(cache_layer, slots, kv_len)            # [n, n_kv, kv_len, dim]
        K = K.repeat_interleave(mixer.num_key_value_groups, dim=1)  # portable GQA (no enable_gqa dep)
        V = V.repeat_interleave(mixer.num_key_value_groups, dim=1)
        q = q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        scores = torch.matmul(q, K.transpose(-1, -2)) / math.sqrt(hd)  # [n, heads, 1, kv_len]
        if self.config.qk_logit_clip:
            thr = float(self.config.qk_clip_threshold)
            scores = thr * torch.tanh(scores / thr)
        allowed = build_allowed_mask(positions, kv_len, window)        # [n, kv_len]
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed[:, None, None, :], min_value)

        if mixer.sink_logits is None:
            softmax_in = scores if scores.is_cuda else scores.float()
            probs = torch.softmax(softmax_in, dim=-1).to(q.dtype)
        else:
            sink = mixer.sink_logits.view(1, mixer.num_heads, 1, 1).float()
            max_score = torch.maximum(scores.float().max(dim=-1, keepdim=True).values, sink)
            real_exp = torch.exp(scores.float() - max_score)
            sink_exp = torch.exp(sink - max_score)
            probs = (real_exp / (real_exp.sum(dim=-1, keepdim=True) + sink_exp)).to(q.dtype)

        out = torch.matmul(probs, V)                  # [n, heads, 1, dim]
        out = out.transpose(1, 2).contiguous().view(n, -1)
        return mixer.o_proj(out)

    # ------------------------------------------------------------------ #
    def _mamba_step(self, layer_idx: int, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        mixer = self.layers[layer_idx].mixer  # BarbetMambaMixer
        cache_layer = self._mamba_slot[layer_idx]
        n = x.shape[0]

        z = mixer.in_proj_z(x)
        xx = mixer.in_proj_x(x)
        b_proj = mixer.in_proj_b(x)
        c_proj = mixer.in_proj_c(x)
        dt = mixer.in_proj_dt(x)
        conv_inputs = torch.cat([xx, b_proj, c_proj], dim=-1)  # [n, channels]

        # --- causal-conv single step (use_step_cache branch) --- #
        conv_state = self.mamba.conv[cache_layer, slots]       # [n, channels, d_conv]
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = conv_inputs
        weights = torch.cat([mixer.conv_x.weight, mixer.conv_b.weight, mixer.conv_c.weight], dim=0)
        bias = torch.cat([mixer.conv_x.bias, mixer.conv_b.bias, mixer.conv_c.bias], dim=0)
        conv_out = (conv_state * weights.squeeze(1)[None, :, :]).sum(dim=-1) + bias
        conv_out = F.silu(conv_out).to(dtype=x.dtype)          # [n, channels]
        self.mamba.conv[cache_layer, slots] = conv_state

        xx, b_proj, c_proj = torch.split(
            conv_out, [mixer.inner_size, mixer.num_groups * mixer.d_state, mixer.num_groups * mixer.d_state], dim=-1
        )

        # --- selective-scan single step (mirrors _selective_scan, seq_len=1) --- #
        dtype = xx.dtype
        xx = xx.view(n, mixer.num_heads, mixer.head_dim)
        b_proj = b_proj.view(n, mixer.num_groups, mixer.d_state)
        c_proj = c_proj.view(n, mixer.num_groups, mixer.d_state)
        z = z.view(n, mixer.num_heads, mixer.head_dim)

        state = self.mamba.ssm[cache_layer, slots].to(dtype=dtype)  # [n, heads, head_dim, d_state]
        heads_per_group = mixer.num_heads // mixer.num_groups
        group_for_head = torch.arange(mixer.num_heads, device=x.device) // heads_per_group
        a = -torch.exp(mixer.A_log.float()).to(dtype=dtype)
        d = mixer.D.to(dtype=dtype)
        dt_bias = mixer.dt_bias.to(dtype=dtype)

        dt_pos = F.softplus(dt + dt_bias)            # [n, heads]
        d_a = torch.exp(dt_pos * a)                  # [n, heads]
        b_pos = b_proj.index_select(1, group_for_head)  # [n, heads, d_state]
        c_pos = c_proj.index_select(1, group_for_head)
        state = state * d_a[:, :, None, None] + (dt_pos[:, :, None, None] * b_pos[:, :, None, :] * xx[:, :, :, None])
        y = (state * c_pos[:, :, None, :]).sum(dim=-1)   # [n, heads, head_dim]
        y = y + d[None, :, None] * xx
        self.mamba.ssm[cache_layer, slots] = state

        # rms-norm gated + out_proj, matching BarbetMambaMixer.forward's tail.
        y = y.reshape(n, 1, mixer.inner_size)
        y = mixer._rmsnorm_gated(y, z.reshape(n, 1, mixer.inner_size))
        return mixer.out_proj(y)[:, 0, :]

    # ------------------------------------------------------------------ #
    def decode_step(self, x: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One Barbet step for ``n`` rows. ``x``: [n, H] -> hidden [n, H].

        ``slots``/``positions`` are per-row LongTensors; caches are updated in
        place at each row's slot and position.
        """
        h = x
        for layer_idx, layer in enumerate(self.layers):
            residual = h
            normed = layer.input_layernorm(h)
            if layer.layer_type == "mamba":
                mixed = self._mamba_step(layer_idx, normed, slots)
            else:
                mixed = self._attn_step(layer_idx, normed, slots, positions)
            h = residual + mixed
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.backbone.norm(h)

    # ------------------------------------------------------------------ #
    def prefill(self, embeds: torch.Tensor, slot: int, start: int = 0) -> torch.Tensor:
        """Run a prompt (``[L, H]``) for one row, returning hiddens ``[L, H]``.

        Loops the decode kernel over positions ``start .. start+L-1`` (batch=1),
        which is provably equal to the model's parallel prefill.
        """
        return self.prefill_batch([embeds], [slot])[0]

    def prefill_batch(self, embeds_list: List[torch.Tensor], slots: List[int]) -> List[torch.Tensor]:
        """Prefill several prompts together, batched per position.

        ``embeds_list[k]`` is ``[L_k, H]``; returns ``[L_k, H]`` hiddens per row.
        At position ``t`` every still-running prompt is processed in one batched
        ``decode_step`` (GEMMs batched across sequences), so this is bit-identical
        to looping :meth:`prefill` per row but with O(max L_k) steps instead of
        O(sum L_k). Each row sits at the same absolute position ``t``.
        """
        lengths = [e.shape[0] for e in embeds_list]
        for slot in slots:
            self.reset_slot(slot)
        outs: List[List[torch.Tensor]] = [[None] * L for L in lengths]
        t_max = max(lengths) if lengths else 0
        for t in range(t_max):
            act = [k for k in range(len(slots)) if t < lengths[k]]
            x = torch.stack([embeds_list[k][t] for k in act], dim=0)           # [a, H]
            slot_t = torch.tensor([slots[k] for k in act], device=self.device, dtype=torch.long)
            pos = torch.full((len(act),), t, device=self.device, dtype=torch.long)
            h = self.decode_step(x, slot_t, pos)
            for j, k in enumerate(act):
                outs[k][t] = h[j]
        return [torch.stack(outs[k], dim=0) for k in range(len(slots))]
