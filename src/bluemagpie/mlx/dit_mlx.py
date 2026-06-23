"""LocDiT (VoxCPMLocDiTV2) estimator + UnifiedCFM Euler solver — MLX.

The flow-matching DiT is the per-patch FLOP dominator. ``LocDiTMLX`` is the
estimator (timestep/delta-time embeddings + in/cond/out projections + the
non-causal LongRoPE MiniCPM decoder). ``solve_euler`` is the classifier-free
Euler sampler (mirrors ``UnifiedCFM.solve_euler`` with cfg-zero-star), and
``cfm_sample`` mirrors ``UnifiedCFM.forward`` (t_span build + the per-patch
noise draw, here an MLX RNG so it can be seeded/injected for parity).
"""

from __future__ import annotations

import math

import mlx.core as mx

from .barbet_mlx import _lin, _silu
from .convert import to_mx
from .minicpm_mlx import MiniCPMMLX


class LocDiTMLX:
    def __init__(self, dit) -> None:
        self.hidden = dit.config.hidden_size
        self.in_channels = dit.in_channels
        self.in_w = to_mx(dit.in_proj.weight); self.in_b = to_mx(dit.in_proj.bias)
        self.cond_w = to_mx(dit.cond_proj.weight); self.cond_b = to_mx(dit.cond_proj.bias)
        self.out_w = to_mx(dit.out_proj.weight); self.out_b = to_mx(dit.out_proj.bias)
        self.t1w = to_mx(dit.time_mlp.linear_1.weight); self.t1b = to_mx(dit.time_mlp.linear_1.bias)
        self.t2w = to_mx(dit.time_mlp.linear_2.weight); self.t2b = to_mx(dit.time_mlp.linear_2.bias)
        self.d1w = to_mx(dit.delta_time_mlp.linear_1.weight); self.d1b = to_mx(dit.delta_time_mlp.linear_1.bias)
        self.d2w = to_mx(dit.delta_time_mlp.linear_2.weight); self.d2b = to_mx(dit.delta_time_mlp.linear_2.bias)
        self.decoder = MiniCPMMLX(dit.decoder)

    def _sin_emb(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half = self.hidden // 2
        c = math.log(10000) / (half - 1)
        emb = mx.exp(mx.arange(half, dtype=mx.float32) * -c)        # [half]
        emb = scale * x[:, None] * emb[None, :]                     # [N, half]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)  # [N, hidden]

    def _temb(self, t: mx.array, w1, b1, w2, b2) -> mx.array:
        return _lin(_silu(_lin(self._sin_emb(t), w1, b1)), w2, b2)

    def __call__(self, x: mx.array, mu: mx.array, t: mx.array, cond: mx.array, dt: mx.array) -> mx.array:
        # x: [N, C, T]; mu: [N, C2]; t: [N]; cond: [N, C, T']; dt: [N]
        n = x.shape[0]
        xx = _lin(mx.transpose(x, (0, 2, 1)), self.in_w, self.in_b)        # [N, T, hidden]
        cc = _lin(mx.transpose(cond, (0, 2, 1)), self.cond_w, self.cond_b)  # [N, T', hidden]
        prefix = cc.shape[1]
        temb = self._temb(t, self.t1w, self.t1b, self.t2w, self.t2b) + self._temb(
            dt, self.d1w, self.d1b, self.d2w, self.d2b
        )                                                                  # [N, hidden]
        mu_r = mu.reshape(n, -1, self.hidden)                              # [N, mu_len, hidden]
        seq = mx.concatenate([mu_r, temb[:, None, :], cc, xx], axis=1)
        h = self.decoder(seq, is_causal=False)
        h = h[:, prefix + mu_r.shape[1] + 1 :, :]                          # the x positions
        h = _lin(h, self.out_w, self.out_b)                               # [N, T, C]
        return mx.transpose(h, (0, 2, 1))                                 # [N, C, T]


def _optimized_scale(pos_flat: mx.array, neg_flat: mx.array) -> mx.array:
    dot = mx.sum(pos_flat * neg_flat, axis=1, keepdims=True)
    sq = mx.sum(neg_flat ** 2, axis=1, keepdims=True) + 1e-8
    return dot / sq


def solve_euler(estimator: LocDiTMLX, x: mx.array, t_span: mx.array, mu: mx.array, cond: mx.array,
                cfg_value: float, use_cfg_zero_star: bool = True) -> mx.array:
    n_steps = t_span.shape[0]
    t = t_span[0]
    dt = t_span[0] - t_span[1]
    b = x.shape[0]
    zero_init_steps = max(1, int(n_steps * 0.04))
    for step in range(1, n_steps):
        if use_cfg_zero_star and step <= zero_init_steps:
            dphi = mx.zeros_like(x)
        else:
            x_in = mx.concatenate([x, x], axis=0)
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            t_in = mx.full((2 * b,), t, dtype=x.dtype)
            dt_in = mx.zeros((2 * b,), dtype=x.dtype)          # mean_mode=False
            cond_in = mx.concatenate([cond, cond], axis=0)
            out = estimator(x_in, mu_in, t_in, cond_in, dt_in)
            pos = out[:b]
            neg = out[b:]
            if use_cfg_zero_star:
                st = _optimized_scale(pos.reshape(b, -1), neg.reshape(b, -1)).reshape(b, 1, 1)
            else:
                st = 1.0
            dphi = neg * st + cfg_value * (pos - neg * st)
        x = x - dt * dphi
        t = t - dt
        if step < n_steps - 1:
            dt = t - t_span[step + 1]
    return x


def cfm_sample(estimator: LocDiTMLX, mu: mx.array, cond: mx.array, n_timesteps: int, patch_size: int,
               cfg_value: float, key=None, temperature: float = 1.0, sway_sampling_coef: float = 1.0) -> mx.array:
    b = mu.shape[0]
    in_ch = estimator.in_channels
    if key is None:
        z = mx.random.normal((b, in_ch, patch_size)) * temperature
    else:
        z = mx.random.normal((b, in_ch, patch_size), key=key) * temperature
    t_span = mx.linspace(1, 0, n_timesteps + 1)
    t_span = t_span + sway_sampling_coef * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
    return solve_euler(estimator, z, t_span, mu, cond, cfg_value)
