"""Barbet (hybrid Mamba2 + sliding/global attention) — MLX full-sequence forward.

A faithful MLX re-implementation of ``BarbetModel.forward`` (no cache), so it
can be checked for numerical parity against the PyTorch reference. Weights are
the trained torch weights, converted once to ``mx.array`` and looked up by their
``named_parameters()`` name.

Covers all three Barbet layer types — global attention, sliding-window
attention, and the Mamba2 mixer (depthwise causal conv + selective scan) — plus
qk-norm, optional qk-logit clipping, and the optional attention sink.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .convert import torch_params_to_mx

_F32_MIN = -3.4028234663852886e38


def _softplus(x: mx.array) -> mx.array:
    return nn.softplus(x)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _lin(x: mx.array, w: mx.array, b: mx.array = None) -> mx.array:
    y = x @ mx.transpose(w)
    return y + b if b is not None else y


def _rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    v = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    xn = x.astype(mx.float32) * mx.rsqrt(v + eps)
    return xn * w.astype(mx.float32)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, heads, S, d]; cos/sin: [B, S, d]
    return x * cos[:, None, :, :] + _rotate_half(x) * sin[:, None, :, :]


def _repeat_kv(x: mx.array, n: int) -> mx.array:
    if n == 1:
        return x
    b, nkv, s, hd = x.shape
    x = mx.broadcast_to(x[:, :, None, :, :], (b, nkv, n, s, hd))
    return x.reshape(b, nkv * n, s, hd)


class BarbetMLX:
    """MLX forward for a torch ``BarbetModel`` (``BarbetTSLM.backbone``)."""

    def __init__(self, backbone) -> None:
        cfg = backbone.config
        self.cfg = cfg
        self.p = torch_params_to_mx(backbone)
        self.eps = cfg.rms_norm_eps
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.groups = self.nh // self.nkv
        self.hd = cfg.head_dim
        self.layer_types = [cfg.layer_type(i) for i in range(cfg.num_hidden_layers)]

        # Mamba dims (shared across mamba layers).
        self.inner = cfg.hidden_size * cfg.mamba_expand
        self.d_state = max(cfg.mamba_d_state, 1)
        self.d_conv = cfg.mamba_d_conv
        self.m_heads = self.inner // self.hd
        self.m_groups = cfg.num_key_value_heads
        self.group_size = self.inner // self.m_groups
        self.group_for_head = mx.array(
            [h // (self.m_heads // self.m_groups) for h in range(self.m_heads)]
        )

        self.rope_scale = None
        if cfg.rope_scaling:
            stype = str(cfg.rope_scaling.get("type", "linear")).lower()
            factor = cfg.rope_scaling.get("factor")
            if stype == "linear" and factor:
                self.rope_scale = float(factor)

    # ------------------------------------------------------------------ #
    def _rope(self, seq_len: int, batch: int):
        positions = mx.arange(seq_len, dtype=mx.float32)
        if self.rope_scale and self.rope_scale > 1.0:
            positions = positions / self.rope_scale
        inv_freq = 1.0 / (self.cfg.rope_theta ** (mx.arange(0, self.hd, 2, dtype=mx.float32) / self.hd))
        freqs = positions[:, None] * inv_freq[None, :]          # [S, hd/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)           # [S, hd]
        cos = mx.broadcast_to(mx.cos(emb)[None], (batch, seq_len, self.hd))
        sin = mx.broadcast_to(mx.sin(emb)[None], (batch, seq_len, self.hd))
        return cos, sin

    def _attn(self, i: int, x: mx.array, cos, sin, window):
        pre = f"layers.{i}.mixer."
        b, s, _ = x.shape
        q = _lin(x, self.p[pre + "q_proj.weight"]).reshape(b, s, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = _lin(x, self.p[pre + "k_proj.weight"]).reshape(b, s, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = _lin(x, self.p[pre + "v_proj.weight"]).reshape(b, s, self.nkv, self.hd).transpose(0, 2, 1, 3)
        if self.cfg.qk_norm:
            q = _rms_norm(q, self.p[pre + "q_norm.weight"], self.eps)
            k = _rms_norm(k, self.p[pre + "k_norm.weight"], self.eps)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k = _repeat_kv(k, self.groups)
        v = _repeat_kv(v, self.groups)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.hd)   # [b, nh, s, s]
        if self.cfg.qk_logit_clip:
            thr = float(self.cfg.qk_clip_threshold)
            scores = thr * mx.tanh(scores / thr)

        qpos = mx.arange(s)[:, None]
        kpos = mx.arange(s)[None, :]
        allowed = kpos <= qpos
        if window is not None and window > 0:
            allowed = allowed & (kpos >= (qpos - window + 1))
        scores = mx.where(allowed[None, None], scores, _F32_MIN)

        if not self.cfg.attention_sink:
            probs = mx.softmax(scores.astype(mx.float32), axis=-1)
        else:
            sink = self.p[pre + "sink_logits"].reshape(1, self.nh, 1, 1).astype(mx.float32)
            sc = scores.astype(mx.float32)
            max_score = mx.maximum(mx.max(sc, axis=-1, keepdims=True), sink)
            real_exp = mx.exp(sc - max_score)
            sink_exp = mx.exp(sink - max_score)
            probs = real_exp / (mx.sum(real_exp, axis=-1, keepdims=True) + sink_exp)

        out = probs @ v                                              # [b, nh, s, hd]
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.nh * self.hd)
        return _lin(out, self.p[pre + "o_proj.weight"])

    def _causal_conv(self, values: mx.array, w: mx.array, bias: mx.array) -> mx.array:
        # values [b, s, c]; w [c, 1, K] (depthwise); returns silu(conv) [b, s, c]
        b, s, c = values.shape
        k = w.shape[-1]
        pad = mx.zeros((b, k - 1, c), dtype=values.dtype)
        padded = mx.concatenate([pad, values], axis=1)              # [b, s+k-1, c]
        wk = w[:, 0, :]                                             # [c, K]
        acc = mx.zeros((b, s, c), dtype=values.dtype)
        for t in range(k):
            acc = acc + padded[:, t : t + s, :] * wk[:, t][None, None, :]
        acc = acc + bias[None, None, :]
        return _silu(acc)

    def _mamba(self, i: int, x: mx.array) -> mx.array:
        pre = f"layers.{i}.mixer."
        b, s, _ = x.shape
        z = _lin(x, self.p[pre + "in_proj_z.weight"])
        xx = _lin(x, self.p[pre + "in_proj_x.weight"])
        bb = _lin(x, self.p[pre + "in_proj_b.weight"])
        cc = _lin(x, self.p[pre + "in_proj_c.weight"])
        dt = _lin(x, self.p[pre + "in_proj_dt.weight"])            # [b, s, m_heads]

        xx = self._causal_conv(xx, self.p[pre + "conv_x.weight"], self.p[pre + "conv_x.bias"])
        bb = self._causal_conv(bb, self.p[pre + "conv_b.weight"], self.p[pre + "conv_b.bias"])
        cc = self._causal_conv(cc, self.p[pre + "conv_c.weight"], self.p[pre + "conv_c.bias"])

        xx = xx.reshape(b, s, self.m_heads, self.hd)
        bb = bb.reshape(b, s, self.m_groups, self.d_state)
        cc = cc.reshape(b, s, self.m_groups, self.d_state)

        a = -mx.exp(self.p[pre + "A_log"].astype(mx.float32))      # [m_heads]
        d = self.p[pre + "D"]
        dt_bias = self.p[pre + "dt_bias"]

        state = mx.zeros((b, self.m_heads, self.hd, self.d_state))
        ys = []
        for pos in range(s):
            dt_pos = _softplus(dt[:, pos] + dt_bias)               # [b, m_heads]
            d_a = mx.exp(dt_pos * a)                               # [b, m_heads]
            b_pos = mx.take(bb[:, pos], self.group_for_head, axis=1)   # [b, m_heads, d_state]
            c_pos = mx.take(cc[:, pos], self.group_for_head, axis=1)
            x_pos = xx[:, pos]                                     # [b, m_heads, hd]
            state = state * d_a[:, :, None, None] + (
                dt_pos[:, :, None, None] * b_pos[:, :, None, :] * x_pos[:, :, :, None]
            )
            y = mx.sum(state * c_pos[:, :, None, :], axis=-1) + d[None, :, None] * x_pos
            ys.append(y.reshape(b, self.inner))
        y = mx.stack(ys, axis=1)                                   # [b, s, inner]

        # rms-norm gated (manual grouped path).
        hg = y * _silu(z)
        grouped = hg.reshape(b, s, self.m_groups, self.group_size)
        var = mx.mean(grouped.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        grouped = grouped.astype(mx.float32) * mx.rsqrt(var + 1e-5)
        nw = self.p[pre + "norm.weight"].reshape(1, 1, self.m_groups, self.group_size).astype(mx.float32)
        y = (grouped * nw).reshape(b, s, self.inner)
        return _lin(y, self.p[pre + "out_proj.weight"])

    def _mlp(self, i: int, x: mx.array) -> mx.array:
        pre = f"layers.{i}.mlp."
        return _lin(
            _silu(_lin(x, self.p[pre + "gate_proj.weight"])) * _lin(x, self.p[pre + "up_proj.weight"]),
            self.p[pre + "down_proj.weight"],
        )

    # ------------------------------------------------------------------ #
    # Cached single-step decode (batch=1, growing KV) — mirrors forward_step.
    # ------------------------------------------------------------------ #
    def init_cache(self) -> dict:
        cache = {"k": {}, "v": {}, "conv": {}, "ssm": {}}
        return cache

    def _rope_at(self, pos: int):
        p = pos / self.rope_scale if (self.rope_scale and self.rope_scale > 1.0) else pos
        inv_freq = 1.0 / (self.cfg.rope_theta ** (mx.arange(0, self.hd, 2, dtype=mx.float32) / self.hd))
        freqs = mx.array([float(p)])[:, None] * inv_freq[None, :]      # [1, hd/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)                 # [1, hd]
        return mx.cos(emb)[None], mx.sin(emb)[None]                   # [1, 1, hd]

    def _attn_step(self, i: int, x: mx.array, pos: int, cache: dict, window) -> mx.array:
        pre = f"layers.{i}.mixer."
        q = _lin(x, self.p[pre + "q_proj.weight"]).reshape(1, 1, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = _lin(x, self.p[pre + "k_proj.weight"]).reshape(1, 1, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = _lin(x, self.p[pre + "v_proj.weight"]).reshape(1, 1, self.nkv, self.hd).transpose(0, 2, 1, 3)
        if self.cfg.qk_norm:
            q = _rms_norm(q, self.p[pre + "q_norm.weight"], self.eps)
            k = _rms_norm(k, self.p[pre + "k_norm.weight"], self.eps)
        cos, sin = self._rope_at(pos)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if i in cache["k"]:
            k = mx.concatenate([cache["k"][i], k], axis=2)
            v = mx.concatenate([cache["v"][i], v], axis=2)
        cache["k"][i] = k
        cache["v"][i] = v
        kv_len = k.shape[2]
        kk = _repeat_kv(k, self.groups)
        vv = _repeat_kv(v, self.groups)

        scores = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(self.hd)   # [1, nh, 1, kv_len]
        if self.cfg.qk_logit_clip:
            thr = float(self.cfg.qk_clip_threshold)
            scores = thr * mx.tanh(scores / thr)
        kidx = mx.arange(kv_len)
        allowed = kidx <= pos
        if window is not None and window > 0:
            allowed = allowed & (kidx >= (pos - window + 1))
        scores = mx.where(allowed[None, None, None, :], scores, _F32_MIN)

        if not self.cfg.attention_sink:
            probs = mx.softmax(scores.astype(mx.float32), axis=-1)
        else:
            sink = self.p[pre + "sink_logits"].reshape(1, self.nh, 1, 1).astype(mx.float32)
            sc = scores.astype(mx.float32)
            max_score = mx.maximum(mx.max(sc, axis=-1, keepdims=True), sink)
            real_exp = mx.exp(sc - max_score)
            sink_exp = mx.exp(sink - max_score)
            probs = real_exp / (mx.sum(real_exp, axis=-1, keepdims=True) + sink_exp)

        out = (probs @ vv).transpose(0, 2, 1, 3).reshape(1, self.nh * self.hd)
        return _lin(out, self.p[pre + "o_proj.weight"])

    def _mamba_step(self, i: int, x: mx.array, cache: dict) -> mx.array:
        pre = f"layers.{i}.mixer."
        z = _lin(x, self.p[pre + "in_proj_z.weight"])
        xx = _lin(x, self.p[pre + "in_proj_x.weight"])
        bb = _lin(x, self.p[pre + "in_proj_b.weight"])
        cc = _lin(x, self.p[pre + "in_proj_c.weight"])
        dt = _lin(x, self.p[pre + "in_proj_dt.weight"])
        conv_inputs = mx.concatenate([xx, bb, cc], axis=-1)            # [1, channels]

        channels = conv_inputs.shape[-1]
        conv_state = cache["conv"].get(i)
        if conv_state is None:
            conv_state = mx.zeros((1, channels, self.d_conv))
        conv_state = mx.roll(conv_state, -1, axis=-1)
        conv_state = mx.concatenate([conv_state[:, :, :-1], conv_inputs[:, :, None]], axis=-1)
        w = mx.concatenate([
            self.p[pre + "conv_x.weight"], self.p[pre + "conv_b.weight"], self.p[pre + "conv_c.weight"]
        ], axis=0)                                                     # [channels, 1, K]
        bias = mx.concatenate([
            self.p[pre + "conv_x.bias"], self.p[pre + "conv_b.bias"], self.p[pre + "conv_c.bias"]
        ], axis=0)
        conv_out = mx.sum(conv_state * w[:, 0, :][None], axis=-1) + bias
        conv_out = _silu(conv_out)
        cache["conv"][i] = conv_state

        xx, bb, cc = mx.split(
            conv_out, [self.inner, self.inner + self.m_groups * self.d_state], axis=-1
        )
        xx = xx.reshape(1, self.m_heads, self.hd)
        bb = bb.reshape(1, self.m_groups, self.d_state)
        cc = cc.reshape(1, self.m_groups, self.d_state)
        z = z.reshape(1, self.m_heads, self.hd)

        a = -mx.exp(self.p[pre + "A_log"].astype(mx.float32))
        d = self.p[pre + "D"]
        dt_bias = self.p[pre + "dt_bias"]
        dt_pos = _softplus(dt + dt_bias)                              # [1, m_heads]
        d_a = mx.exp(dt_pos * a)
        b_pos = mx.take(bb, self.group_for_head, axis=1)             # [1, m_heads, d_state]
        c_pos = mx.take(cc, self.group_for_head, axis=1)

        state = cache["ssm"].get(i)
        if state is None:
            state = mx.zeros((1, self.m_heads, self.hd, self.d_state))
        state = state * d_a[:, :, None, None] + (dt_pos[:, :, None, None] * b_pos[:, :, None, :] * xx[:, :, :, None])
        y = mx.sum(state * c_pos[:, :, None, :], axis=-1) + d[None, :, None] * xx
        cache["ssm"][i] = state

        y = y.reshape(1, 1, self.inner)
        zg = z.reshape(1, 1, self.inner)
        hg = y * _silu(zg)
        grouped = hg.reshape(1, 1, self.m_groups, self.group_size)
        var = mx.mean(grouped.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        grouped = grouped.astype(mx.float32) * mx.rsqrt(var + 1e-5)
        nw = self.p[pre + "norm.weight"].reshape(1, 1, self.m_groups, self.group_size).astype(mx.float32)
        y = (grouped * nw).reshape(1, self.inner)
        return _lin(y, self.p[pre + "out_proj.weight"])

    def step(self, x: mx.array, pos: int, cache: dict) -> mx.array:
        """One decode step. ``x``: [1, H] at absolute position ``pos`` -> [1, H]."""
        h = x
        for i, lt in enumerate(self.layer_types):
            residual = h
            normed = _rms_norm(h, self.p[f"layers.{i}.input_layernorm.weight"], self.eps)
            if lt == "mamba":
                mixed = self._mamba_step(i, normed, cache)
            else:
                window = self.cfg.sliding_window_size if lt == "sliding_attention" else None
                mixed = self._attn_step(i, normed, pos, cache, window)
            h = residual + mixed
            h = h + self._mlp(i, _rms_norm(h, self.p[f"layers.{i}.post_attention_layernorm.weight"], self.eps))
        return _rms_norm(h, self.p["norm.weight"], self.eps)

    def prefill(self, embeds: mx.array, cache: dict) -> mx.array:
        """Loop the step kernel over a prompt ``[1, L, H]`` -> hiddens ``[1, L, H]``."""
        outs = [self.step(embeds[:, t, :], t, cache) for t in range(embeds.shape[1])]
        return mx.stack(outs, axis=1)

    # ------------------------------------------------------------------ #
    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        b, s, _ = inputs_embeds.shape
        cos, sin = self._rope(s, b)
        h = inputs_embeds
        for i, lt in enumerate(self.layer_types):
            residual = h
            normed = _rms_norm(h, self.p[f"layers.{i}.input_layernorm.weight"], self.eps)
            if lt == "mamba":
                mixed = self._mamba(i, normed)
            else:
                window = self.cfg.sliding_window_size if lt == "sliding_attention" else None
                mixed = self._attn(i, normed, cos, sin, window)
            h = residual + mixed
            h = h + self._mlp(i, _rms_norm(h, self.p[f"layers.{i}.post_attention_layernorm.weight"], self.eps))
        return _rms_norm(h, self.p["norm.weight"], self.eps)
