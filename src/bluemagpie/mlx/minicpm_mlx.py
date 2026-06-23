"""MiniCPM (VoxCPM2 acoustic LM) — MLX full-sequence forward.

One implementation covers all three places BlueMagpie uses a ``MiniCPMModel``:
- the RALM (residual acoustic LM): ``no_rope=True``, causal;
- the LocEnc encoder: rope (LongRoPE), non-causal;
- the LocDiT decoder: rope (LongRoPE), non-causal.

Faithful to ``minicpm4/model.py`` (full forward, no cache): LongRoPE, GQA, the
optional muP scale-depth residual, and SwiGLU MLP.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .barbet_mlx import _lin, _rms_norm, _rotate_half, _silu
from .convert import torch_params_to_mx


class MiniCPMMLX:
    def __init__(self, model) -> None:
        cfg = model.config
        self.cfg = cfg
        self.p = torch_params_to_mx(model)
        self.eps = cfg.rms_norm_eps
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.groups = self.nh // self.nkv
        self.hd = cfg.hidden_size // self.nh if cfg.kv_channels is None else cfg.kv_channels
        self.no_rope = cfg.no_rope
        self.use_mup = cfg.use_mup
        self.scale_depth = cfg.scale_depth
        self.num_layers = cfg.num_hidden_layers
        if not self.no_rope:
            rs = cfg.rope_scaling
            short = np.asarray(rs.short_factor, dtype=np.float64)
            long = np.asarray(rs.long_factor, dtype=np.float64)
            self._orig = rs.original_max_position_embeddings
            inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, self.hd, 2) / self.hd))
            scale = cfg.max_position_embeddings / self._orig
            self._scaling = math.sqrt(1 + math.log(scale) / math.log(self._orig))
            # Precompute (inv_freq / ext) as mx constants; rope is then pure-mx
            # (so the estimator can be mx.compile'd) and cached per seq_len.
            self._sel_short = mx.array((inv_freq / short).astype(np.float32))
            self._sel_long = mx.array((inv_freq / long).astype(np.float32))
            self._rope_cache: dict = {}

    def _rope(self, seq_len: int):
        cached = self._rope_cache.get(seq_len)
        if cached is not None:
            return cached
        sel = self._sel_long if seq_len > self._orig else self._sel_short  # [hd/2]
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * sel[None, :]                                  # [seq, hd/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)                      # [seq, hd]
        cos = mx.cos(emb) * self._scaling
        sin = mx.sin(emb) * self._scaling
        mx.eval(cos, sin)
        self._rope_cache[seq_len] = (cos, sin)
        return cos, sin

    def _attn(self, i: int, x: mx.array, is_causal: bool) -> mx.array:
        pre = f"layers.{i}.self_attn."
        b, s, _ = x.shape
        q = _lin(x, self.p[pre + "q_proj.weight"]).reshape(b, s, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = _lin(x, self.p[pre + "k_proj.weight"]).reshape(b, s, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = _lin(x, self.p[pre + "v_proj.weight"]).reshape(b, s, self.nkv, self.hd).transpose(0, 2, 1, 3)
        if not self.no_rope:
            cos, sin = self._rope(s)                                       # [s, hd]
            q = q * cos[None, None] + _rotate_half(q) * sin[None, None]
            k = k * cos[None, None] + _rotate_half(k) * sin[None, None]

        # Fused Metal SDPA (handles GQA by broadcasting the kv heads), matching
        # the reference's torch SDPA with is_causal + enable_gqa.
        scale = 1.0 / math.sqrt(self.hd)
        mask = "causal" if is_causal else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.nh * self.hd)
        return _lin(out, self.p[pre + "o_proj.weight"])

    def _mlp(self, i: int, x: mx.array) -> mx.array:
        pre = f"layers.{i}.mlp."
        return _lin(
            _silu(_lin(x, self.p[pre + "gate_proj.weight"])) * _lin(x, self.p[pre + "up_proj.weight"]),
            self.p[pre + "down_proj.weight"],
        )

    def _layer(self, i: int, h: mx.array, is_causal: bool) -> mx.array:
        pre = f"layers.{i}."
        scale = (self.scale_depth / math.sqrt(self.num_layers)) if self.use_mup else 1.0
        residual = h
        mixed = self._attn(i, _rms_norm(h, self.p[pre + "input_layernorm.weight"], self.eps), is_causal)
        h = residual + mixed * scale
        residual = h
        mlp = self._mlp(i, _rms_norm(h, self.p[pre + "post_attention_layernorm.weight"], self.eps))
        h = residual + mlp * scale
        return h

    def __call__(self, inputs_embeds: mx.array, is_causal: bool = True) -> mx.array:
        h = inputs_embeds
        for i in range(self.num_layers):
            h = self._layer(i, h, is_causal)
        return _rms_norm(h, self.p["norm.weight"], self.eps)

    # ------------------------------------------------------------------ #
    # Cached single-step decode (batch=1, growing KV) — used by the RALM,
    # which is no-rope + causal. Query at ``pos`` sees keys [0..pos] (all valid),
    # so no explicit mask is needed.
    # ------------------------------------------------------------------ #
    def init_cache(self) -> dict:
        return {"k": {}, "v": {}}

    def _attn_step(self, i: int, x: mx.array, pos: int, cache: dict) -> mx.array:
        pre = f"layers.{i}.self_attn."
        q = _lin(x, self.p[pre + "q_proj.weight"]).reshape(1, 1, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = _lin(x, self.p[pre + "k_proj.weight"]).reshape(1, 1, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = _lin(x, self.p[pre + "v_proj.weight"]).reshape(1, 1, self.nkv, self.hd).transpose(0, 2, 1, 3)
        if not self.no_rope:
            cos, sin = self._rope(pos + 1)
            cos = cos[pos][None, None, None]
            sin = sin[pos][None, None, None]
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        if i in cache["k"]:
            k = mx.concatenate([cache["k"][i], k], axis=2)
            v = mx.concatenate([cache["v"][i], v], axis=2)
        cache["k"][i] = k
        cache["v"][i] = v
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.hd), mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(1, self.nh * self.hd)
        return _lin(out, self.p[pre + "o_proj.weight"])

    def step(self, x: mx.array, pos: int, cache: dict) -> mx.array:
        """One decode step. ``x``: [1, H] at position ``pos`` -> [1, H]."""
        h = x
        scale = (self.scale_depth / math.sqrt(self.num_layers)) if self.use_mup else 1.0
        for i in range(self.num_layers):
            residual = h
            mixed = self._attn_step(i, _rms_norm(h, self.p[f"layers.{i}.input_layernorm.weight"], self.eps), pos, cache)
            h = residual + mixed * scale
            residual = h
            mlp = self._mlp(i, _rms_norm(h, self.p[f"layers.{i}.post_attention_layernorm.weight"], self.eps))
            h = residual + mlp * scale
        return _rms_norm(h, self.p["norm.weight"], self.eps)

    def prefill(self, embeds: mx.array, cache: dict) -> mx.array:
        outs = [self.step(embeds[:, t, :], t, cache) for t in range(embeds.shape[1])]
        return mx.stack(outs, axis=1)
