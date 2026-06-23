"""BlueMagpieMLX — the full autoregressive TTS loop in MLX.

Mirrors ``BlueMagpieModel._inference``: prefill warms cached Barbet/RALM decode
states (looping the MLX step kernels), then each step runs the DiT/CFM sampler,
re-encodes the patch (LocEnc), checks the stop head, and advances Barbet + RALM
by one position. Produces latent patches ``[T, p, d]``; the AudioVAE decode stays
in torch (it runs once at the end).

The per-patch noise can be injected (``noises=...``) so the loop is bit-parity
testable against the torch reference, or drawn from MLX's RNG for real runs.
"""

from __future__ import annotations

import math
from typing import List, Optional

import mlx.core as mx

from .barbet_mlx import BarbetMLX, _lin, _rms_norm, _silu
from .convert import to_mx
from .dit_mlx import LocDiTMLX, solve_euler
from .layers_mlx import AdapterMLX, FSQMLX
from .locenc_mlx import LocEncMLX
from .minicpm_mlx import MiniCPMMLX


def _proj(x, wb):
    return _lin(x, wb[0], wb[1])


class BlueMagpieMLX:
    def __init__(self, model) -> None:
        self.barbet = BarbetMLX(model.base_lm.backbone)
        self.ralm = MiniCPMMLX(model.residual_lm)
        self.locenc = LocEncMLX(model.feat_encoder)
        self.dit = LocDiTMLX(model.feat_decoder.estimator)
        self.adapter = AdapterMLX(model.tslm_adapter)
        self.fsq = FSQMLX(model.fsq_layer)
        self.embed = to_mx(model.base_lm.embed_tokens.weight)       # [vocab, Hb]

        def wb(layer):
            return (to_mx(layer.weight), to_mx(layer.bias))

        self.enc_tslm = wb(model.enc_to_tslm_proj)
        self.enc_lm = wb(model.enc_to_lm_proj)
        self.lm_dit = wb(model.lm_to_dit_proj)
        self.res_dit = wb(model.res_to_dit_proj)
        self.fusion = wb(model.fusion_concat_proj)
        self.stop_proj = wb(model.stop_proj)
        self.stop_head_w = to_mx(model.stop_head.weight)            # no bias
        sp = model.speaker_projector
        self.spk = (to_mx(sp.norm.weight), to_mx(sp.proj.weight), to_mx(sp.proj.bias), sp.norm.eps)

        self.patch_size = model.patch_size
        self.feat_dim = model.config.feat_dim

    def _t_span(self, n_timesteps: int, sway: float = 1.0) -> mx.array:
        t = mx.linspace(1, 0, n_timesteps + 1)
        return t + sway * (mx.cos(math.pi / 2 * t) - 1 + t)

    def inference(self, text_token, audio_feat, text_mask, audio_mask, spk_mask=None, speaker_centroids=None,
                  min_len: int = 2, max_len: int = 2000, inference_timesteps: int = 10, cfg_value: float = 2.0,
                  noises: Optional[List[mx.array]] = None) -> mx.array:
        # ---- prefill (mirror model._inference) ----
        feat_locenc = self.locenc(audio_feat)                       # [1, L, h_enc]
        feat_embed_tslm = _proj(feat_locenc, self.enc_tslm)
        feat_embed_lm = _proj(feat_locenc, self.enc_lm)
        text_embed = mx.take(self.embed, text_token, axis=0)        # [1, L, Hb]
        combined = text_mask[..., None] * text_embed + audio_mask[..., None] * feat_embed_tslm
        if spk_mask is not None and speaker_centroids is not None:
            nw, pw, pb, eps = self.spk
            spk_vec = _lin(_rms_norm(speaker_centroids, nw, eps), pw, pb)   # [1, Hb]
            combined = combined + spk_mask[..., None] * spk_vec[:, None, :]

        bcache = self.barbet.init_cache()
        barbet_hidden = self.barbet.prefill(combined, bcache)       # [1, L, Hb]
        tslm_hidden = self.adapter(barbet_hidden)
        enc_outputs = self.fsq(tslm_hidden) * audio_mask[..., None] + tslm_hidden * text_mask[..., None]
        lm_hidden = enc_outputs[:, -1, :]                           # [1, Hv]

        residual_inputs = _proj(
            mx.concatenate([enc_outputs, audio_mask[..., None] * feat_embed_lm], axis=-1), self.fusion
        )
        rcache = self.ralm.init_cache()
        residual_seq = self.ralm.prefill(residual_inputs, rcache)
        residual_hidden = residual_seq[:, -1, :]                    # [1, Hv]
        prefix_feat_cond = audio_feat[:, -1, ...]                   # [1, p, d]

        pos = int(text_token.shape[1])
        t_span = self._t_span(inference_timesteps)
        patches = []
        for i in range(max_len):
            dit_hidden = mx.concatenate([_proj(lm_hidden, self.lm_dit), _proj(residual_hidden, self.res_dit)], axis=-1)
            cond = mx.transpose(prefix_feat_cond, (0, 2, 1))        # [1, d, p]
            z = noises[i] if noises is not None else mx.random.normal((1, self.feat_dim, self.patch_size))
            pred = solve_euler(self.dit, z, t_span, dit_hidden, cond, cfg_value)   # [1, d, p]
            pred_feat = mx.transpose(pred, (0, 2, 1))              # [1, p, d]

            curr_locenc = self.locenc(pred_feat[:, None])          # [1, 1, h_enc]
            curr_tslm = _proj(curr_locenc, self.enc_tslm)         # [1, 1, Hb]
            curr_lm = _proj(curr_locenc, self.enc_lm)             # [1, 1, Hv]
            patches.append(pred_feat)
            prefix_feat_cond = pred_feat

            stop_logits = _lin(_silu(_lin(lm_hidden, self.stop_proj[0], self.stop_proj[1])), self.stop_head_w)
            stop = int(mx.argmax(stop_logits, axis=-1)[0])
            if i > min_len and stop == 1:
                break

            barbet_step = self.barbet.step(curr_tslm[:, 0, :], pos, bcache)        # [1, Hb]
            lm_hidden = self.fsq(self.adapter(barbet_step[:, None, :]))[:, 0, :]   # [1, Hv]
            curr_residual = _proj(mx.concatenate([lm_hidden, curr_lm[:, 0, :]], axis=-1), self.fusion)
            residual_hidden = self.ralm.step(curr_residual, pos, rcache)           # [1, Hv]
            pos += 1

        return mx.concatenate(patches, axis=0)                     # [T, p, d]


def mlx_generate(model, mlx_model: "BlueMagpieMLX", target_text: str, *, prompt_text: str = "",
                 prompt_wav_path: str = "", reference_wav_path: str = "", speaker_centroid=None,
                 min_len: int = 2, max_len: int = 2000, inference_timesteps: int = 9, cfg_value: float = 2.8,
                 use_null_speaker: bool = True, seed: Optional[int] = None):
    """End-to-end MLX generate: torch input assembly + MLX AR loop + torch AudioVAE.

    ``model`` is the torch :class:`BlueMagpieModel` (used for tokenization, wav
    encoding, and the AudioVAE decode); ``mlx_model`` is :class:`BlueMagpieMLX`
    built from it. Returns a 48 kHz waveform ``torch.Tensor``.
    """
    import numpy as np
    import torch

    ref_feat = model._encode_wav(reference_wav_path, padding_mode="right") if reference_wav_path else None
    prompt_feat = model._encode_wav(prompt_wav_path, padding_mode="left") if prompt_wav_path else None
    text = (prompt_text + target_text) if prompt_feat is not None else target_text
    centroids = None if speaker_centroid is None else speaker_centroid.reshape(1, -1)
    slot = "centroid" if centroids is not None else ("null" if use_null_speaker else "none")
    text_token, audio_feat, text_mask, audio_mask, spk_mask = model._build_inputs(text, ref_feat, prompt_feat, slot)

    tt = mx.array(text_token.cpu().numpy())[None]
    af = to_mx(audio_feat.float())[None]
    txm = to_mx(text_mask.float())[None]
    aum = to_mx(audio_mask.float())[None]
    sm = to_mx(spk_mask.float())[None] if slot == "centroid" else None
    sc = to_mx(centroids.float()) if centroids is not None else None
    if seed is not None:
        mx.random.seed(seed)

    latents = mlx_model.inference(tt, af, txm, aum, spk_mask=sm, speaker_centroids=sc, min_len=min_len,
                                  max_len=max_len, inference_timesteps=inference_timesteps, cfg_value=cfg_value)
    mx.eval(latents)

    lt = torch.from_numpy(np.array(latents))                       # [T, p, d]
    feat_pred = lt.permute(2, 0, 1).reshape(model.config.feat_dim, -1)[None]  # [1, d, T*p]
    decode_audio = model.audio_vae.decode(feat_pred.to(torch.float32))
    return decode_audio.squeeze(1).squeeze(0).cpu()
