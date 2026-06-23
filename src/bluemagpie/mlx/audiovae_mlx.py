"""AudioVAE decoder — MLX.

Re-implements ``AudioVAE.decode`` (the causal-conv vocoder decoder) in MLX so the
whole inference path can run torch-free. Runs once at the end of generation, so
this is for all-MLX deployment, not for speed.

Conv layout (verified vs torch): mx works in NLC, torch conv weight ``[Cout,
Cin/g, K]`` -> mx ``[Cout, K, Cin/g]``; transpose-conv weight ``[Cin, Cout, K]``
-> mx ``[Cout, K, Cin]`` (no kernel flip). Causal padding/trim mirror
``CausalConv1d`` / ``CausalTransposeConv1d``. ``use_noise_block`` must be False
(the shipped default) — the decode is then deterministic.
"""

from __future__ import annotations

import mlx.core as mx

from .convert import to_mx


def _causal_conv(x, w, b, stride, dilation, groups, left):
    # x: NCL mx; w: torch-layout [Cout, Cin/g, K] mx
    xm = x.transpose(0, 2, 1)                                  # NLC
    if left > 0:
        xm = mx.pad(xm, [(0, 0), (left, 0), (0, 0)])
    o = mx.conv1d(xm, w.transpose(0, 2, 1), stride=stride, padding=0, dilation=dilation, groups=groups)
    if b is not None:
        o = o + b
    return o.transpose(0, 2, 1)                                # NCL


def _causal_tconv(x, w, b, stride, trim):
    # groups=1; w: torch-layout [Cin, Cout, K] mx -> mx [Cout, K, Cin]
    xm = x.transpose(0, 2, 1)
    o = mx.conv_transpose1d(xm, w.transpose(1, 2, 0), stride=stride, padding=0)
    if b is not None:
        o = o + b
    o = o.transpose(0, 2, 1)                                   # NCL
    return o[..., :-trim] if trim > 0 else o


def _snake(x, alpha):
    return x + (1.0 / (alpha + 1e-9)) * mx.sin(alpha * x) ** 2


class AudioVAEMLX:
    """MLX ``decode`` for a torch ``AudioVAE`` (use_noise_block=False)."""

    def __init__(self, vae) -> None:
        if getattr(vae, "use_noise_block", False):
            raise NotImplementedError("AudioVAEMLX: use_noise_block=True (stochastic decode) not supported")
        self.vae = vae
        dec = vae.decoder
        self.has_sr = dec.sr_bin_boundaries is not None
        self.sr_idx = None
        if self.has_sr:
            import torch

            sr = torch.tensor([vae.out_sample_rate], dtype=torch.int32)
            self.sr_idx = int(torch.bucketize(sr, dec.sr_bin_boundaries).item())
        # Cache converted weights by torch param id (decode runs once, but the
        # caller may decode several times).
        self._cache: dict = {}

    def _w(self, t):
        k = id(t)
        v = self._cache.get(k)
        if v is None:
            v = to_mx(t)
            self._cache[k] = v
        return v

    def _conv_weight(self, mod):
        # Recompute the weight-norm weight from g/v so it is correct even before
        # any torch forward has run (e.g. straight after loading a checkpoint).
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            k = id(mod.weight_v)
            v = self._cache.get(k)
            if v is None:
                import torch

                v = to_mx(torch._weight_norm(mod.weight_v, mod.weight_g, 0))
                self._cache[k] = v
            return v
        return self._w(mod.weight)

    # ------------------------------------------------------------------ #
    def _conv(self, mod, x):
        left = mod._CausalConv1d__padding * 2 - mod._CausalConv1d__output_padding
        b = self._w(mod.bias) if mod.bias is not None else None
        return _causal_conv(x, self._conv_weight(mod), b, mod.stride[0], mod.dilation[0], mod.groups, left)

    def _tconv(self, mod, x):
        trim = mod._CausalTransposeConv1d__padding * 2 - mod._CausalTransposeConv1d__output_padding
        b = self._w(mod.bias) if mod.bias is not None else None
        return _causal_tconv(x, self._conv_weight(mod), b, mod.stride[0], trim)

    def _snake1d(self, mod, x):
        return _snake(x, self._w(mod.alpha))

    def _residual(self, mod, x):
        # CausalResidualUnit: Sequential(Snake, conv, Snake, conv) + residual.
        y = self._seq(mod.block, x)
        return x + y

    def _sr_cond(self, mod, x):
        # scale_bias: x * scale_embed[idx] + bias_embed[idx]; out_layer Identity.
        scale = self._w(mod.scale_embed.weight)[self.sr_idx]      # [C]
        bias = self._w(mod.bias_embed.weight)[self.sr_idx]
        x = x * scale[None, :, None] + bias[None, :, None]
        return self._forward(mod.out_layer, x)

    def _seq(self, seq, x):
        for m in seq:
            x = self._forward(m, x)
        return x

    def _forward(self, mod, x):
        name = mod.__class__.__name__
        if name == "CausalConv1d":
            return self._conv(mod, x)
        if name == "CausalTransposeConv1d":
            return self._tconv(mod, x)
        if name == "Snake1d":
            return self._snake1d(mod, x)
        if name == "CausalResidualUnit":
            return self._residual(mod, x)
        if name in ("CausalDecoderBlock",):
            return self._seq(mod.block, x)
        if name == "SampleRateConditionLayer":
            return self._sr_cond(mod, x)
        if name == "Tanh":
            return mx.tanh(x)
        if name in ("Sequential", "ModuleList"):
            return self._seq(mod, x)
        if name == "Identity":
            return x
        if name == "NoiseBlock":
            raise NotImplementedError("NoiseBlock (stochastic) not supported in AudioVAEMLX")
        raise NotImplementedError(f"AudioVAEMLX: unhandled module {name}")

    # ------------------------------------------------------------------ #
    def decode(self, z: mx.array) -> mx.array:
        """``z``: [B, D, T] -> audio [B, 1, T*decode_chunk]."""
        dec = self.vae.decoder
        x = z
        if self.has_sr:
            for layer, cond in zip(dec.model, dec.sr_cond_model):
                if cond is not None:
                    x = self._sr_cond(cond, x)
                x = self._forward(layer, x)
            return x
        return self._seq(dec.model, x)
