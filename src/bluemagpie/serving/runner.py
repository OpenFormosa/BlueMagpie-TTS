"""Batched model runner: orchestrates one TTS decode across many sequences.

Mirrors ``BlueMagpieModel._inference`` (the golden reference) but for a batch of
sequences at ragged positions, using the per-row batched LM kernels
(:class:`BatchedBarbet`, :class:`BatchedRALM`) and a per-request RNG for the DiT.

Stage order per engine step (matching ``_inference``'s loop body):
  DiT (LocDiT/UnifiedCFM, batched CFG) -> LocEnc re-encode -> stop head ->
  (for continuing rows) Barbet step -> FSQ -> RALM step.
The stop decision is read **before** advancing (break-before-step), so a row's
emitted patch count matches the reference exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .barbet_batch import BatchedBarbet
from .minicpm_batch import BatchedRALM


def dit_sample(feat_decoder, mu, cond, n_timesteps, patch_size, cfg_value, generators,
               temperature: float = 1.0, sway_sampling_coef: float = 1.0):
    """``UnifiedCFM.forward`` with a **per-row** noise draw.

    ``generators[i] is None`` -> draw from the global default RNG exactly like
    ``UnifiedCFM.forward`` (so batch=1 stays bit-identical to ``_inference``);
    otherwise draw from that row's ``torch.Generator`` (reproducible regardless
    of batch composition / admission order).
    """
    b = mu.shape[0]
    in_ch = feat_decoder.in_channels
    zs = []
    for i in range(b):
        g = generators[i] if generators is not None else None
        if g is None:
            zi = torch.randn((1, in_ch, patch_size), device=mu.device, dtype=mu.dtype)
        else:
            zi = torch.randn((1, in_ch, patch_size), generator=g, device=mu.device, dtype=mu.dtype)
        zs.append(zi * temperature)
    z = torch.cat(zs, dim=0)
    t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    return feat_decoder.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value, use_cfg_zero_star=True)


@dataclass
class _Prepared:
    """Per-sequence inputs assembled before the batched LM prefill."""

    slot: int
    combined_embed: torch.Tensor     # [L, h_b]
    feat_embed_lm: torch.Tensor      # [1, L, h_v]
    text_mask: torch.Tensor          # [1, L]
    audio_mask: torch.Tensor         # [1, L]
    prefix_feat_cond: torch.Tensor   # [1, p, d]
    length: int
    context_len: int
    context_patches: List[torch.Tensor]


@dataclass
class SeqState:
    """Per-sequence decode state inside the runner."""

    slot: int
    lm_hidden: torch.Tensor          # [1, h_v]
    residual_hidden: torch.Tensor    # [1, h_v]
    prefix_feat_cond: torch.Tensor   # [1, p, d]
    pos: int                         # next Barbet/RALM write position (= prompt_len + steps)
    min_len: int
    max_len: int
    n_timesteps: int
    cfg_value: float
    generator: Optional[torch.Generator] = None
    step: int = 0
    finished: bool = False
    context_len: int = 0             # leading context patches (continuation modes)
    patches: List[torch.Tensor] = field(default_factory=list)  # context + emitted [1, p, d]


class BlueMagpieRunner:
    """Holds the model + batched LM kernels and drives batched generation."""

    def __init__(self, model, max_seqs: int, max_len: int) -> None:
        self.model = model
        self.device = model._runtime_device()
        self.dtype = model._runtime_dtype()
        self.patch_size = model.patch_size
        self.barbet = BatchedBarbet(model.base_lm, max_seqs, max_len, self.device, self.dtype)
        self.ralm = BatchedRALM(model.residual_lm, max_seqs, max_len, self.device, self.dtype)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _prepare(self, slot, text_token, audio_feat, text_mask, audio_mask, spk_mask,
                 speaker_centroids, streaming_prefix_len) -> "_Prepared":
        """Per-sequence input assembly (LocEnc + embeds + speaker) — the part
        that is NOT batched. The expensive 28-layer Barbet / 8-layer RALM
        prefill happens later, batched across sequences."""
        m = self.model
        device, dtype = self.device, self.dtype
        text_token = text_token.to(device)
        audio_feat = audio_feat.to(device, dtype=dtype)[None]      # [1, L, p, d]
        text_mask = text_mask.to(device, dtype=dtype)[None]        # [1, L]
        audio_mask = audio_mask.to(device, dtype=dtype)[None]
        L = text_token.shape[0]

        feat_locenc = m.feat_encoder(audio_feat)                   # [1, L, h_enc]
        feat_embed_tslm = m.enc_to_tslm_proj(feat_locenc)
        feat_embed_lm = m.enc_to_lm_proj(feat_locenc)
        text_embed = m.base_lm.embed_tokens(text_token[None])      # [1, L, h_b]
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed_tslm
        if spk_mask is not None and speaker_centroids is not None:
            combined_embed = m._inject_speaker(combined_embed, speaker_centroids, spk_mask.to(device, dtype=dtype)[None])

        prefix_feat_cond = audio_feat[:, -1, ...]                  # [1, p, d]

        # Continuation modes (feat_mask ends with audio): seed the output with
        # the last (streaming_prefix_len-1) real prompt patches so the VAE
        # decode is smooth, exactly like ``_inference``.
        context_len = 0
        context_patches: List[torch.Tensor] = []
        if audio_mask[0, -1].item() == 1:
            audio_indices = audio_mask[0].nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            if context_len > 0:
                last = audio_indices[-context_len:]
                context_patches = list(audio_feat[0, last, :, :].split(1, dim=0))  # context_len x [1, p, d]

        return _Prepared(
            slot=slot,
            combined_embed=combined_embed[0],     # [L, h_b]
            feat_embed_lm=feat_embed_lm,          # [1, L, h_v]
            text_mask=text_mask,                  # [1, L]
            audio_mask=audio_mask,
            prefix_feat_cond=prefix_feat_cond,    # [1, p, d]
            length=L,
            context_len=context_len,
            context_patches=context_patches,
        )

    @torch.inference_mode()
    def prefill_batch(self, prepared: List["_Prepared"]) -> List[SeqState]:
        """Prefill a cohort of sequences together (batched Barbet + RALM)."""
        m = self.model
        slots = [p.slot for p in prepared]
        barbet_hidden = self.barbet.prefill_batch([p.combined_embed for p in prepared], slots)  # per-seq [L, h_b]

        residual_inputs = []
        enc_lasts = []
        for p, bh in zip(prepared, barbet_hidden):
            tslm_hidden_seq = m.tslm_adapter(bh[None])             # [1, L, h_v]
            enc_outputs = (
                m.fsq_layer(tslm_hidden_seq) * p.audio_mask.unsqueeze(-1)
                + tslm_hidden_seq * p.text_mask.unsqueeze(-1)
            )
            enc_lasts.append(enc_outputs[:, -1, :])               # [1, h_v]
            resid = m.fusion_concat_proj(
                torch.cat((enc_outputs, p.audio_mask.unsqueeze(-1) * p.feat_embed_lm), dim=-1)
            )
            residual_inputs.append(resid[0])                      # [L, h_v]

        residual_seqs = self.ralm.prefill_batch(residual_inputs, slots)

        states = []
        for p, lm_hidden, resid_seq in zip(prepared, enc_lasts, residual_seqs):
            states.append(
                SeqState(
                    slot=p.slot,
                    lm_hidden=lm_hidden,
                    residual_hidden=resid_seq[-1:].clone(),
                    prefix_feat_cond=p.prefix_feat_cond,
                    pos=p.length,
                    min_len=2,
                    max_len=2000,
                    n_timesteps=10,
                    cfg_value=2.0,
                    context_len=p.context_len,
                    patches=list(p.context_patches),
                )
            )
        return states

    def prefill(self, slot: int, text_token, audio_feat, text_mask, audio_mask, spk_mask=None,
                speaker_centroids=None, streaming_prefix_len: int = 4) -> SeqState:
        """Prefill a single sequence (thin wrapper over :meth:`prefill_batch`)."""
        prep = self._prepare(slot, text_token, audio_feat, text_mask, audio_mask, spk_mask,
                             speaker_centroids, streaming_prefix_len)
        return self.prefill_batch([prep])[0]

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def decode_step(self, active: List[SeqState]) -> Optional[torch.Tensor]:
        """Advance one engine step over the active sequences (in place).

        Groups must share ``(n_timesteps, cfg_value)`` for one batched DiT call;
        the engine/scheduler is responsible for grouping. Emits one patch per
        active row and marks rows finished per the break-before-step stop rule.
        Returns this step's per-row patch ``[n, p, d]`` (in ``active`` order) for
        streaming consumers, or ``None`` if ``active`` is empty.
        """
        m = self.model
        n = len(active)
        if n == 0:
            return None

        lm_hidden = torch.cat([s.lm_hidden for s in active], dim=0)            # [n, h_v]
        residual_hidden = torch.cat([s.residual_hidden for s in active], dim=0)
        prefix_feat_cond = torch.cat([s.prefix_feat_cond for s in active], dim=0)  # [n, p, d]
        generators = [s.generator for s in active]
        n_timesteps = active[0].n_timesteps
        cfg_value = active[0].cfg_value

        dit_hidden = torch.cat((m.lm_to_dit_proj(lm_hidden), m.res_to_dit_proj(residual_hidden)), dim=-1)
        pred_feat = dit_sample(
            m.feat_decoder, dit_hidden, prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps, self.patch_size, cfg_value, generators,
        ).transpose(1, 2)                                                      # [n, p, d]

        curr_locenc = m.feat_encoder(pred_feat.unsqueeze(1))                   # [n, 1, h_enc]
        curr_embed_tslm = m.enc_to_tslm_proj(curr_locenc)
        curr_embed_lm = m.enc_to_lm_proj(curr_locenc)

        # Emit this step's patch; update streaming condition.
        for i, s in enumerate(active):
            s.patches.append(pred_feat[i : i + 1])
            s.prefix_feat_cond = pred_feat[i : i + 1]

        # Stop decision from the CURRENT lm_hidden (break-before-step), per row.
        stop_flag = m.stop_head(m.stop_actn(m.stop_proj(lm_hidden))).argmax(dim=-1)  # [n]
        stop_cpu = stop_flag.to("cpu").tolist()
        cont_idx = []
        for i, s in enumerate(active):
            stop_now = (s.step > s.min_len and stop_cpu[i] == 1) or (s.step + 1 >= s.max_len)
            if stop_now:
                s.finished = True
            else:
                cont_idx.append(i)

        if not cont_idx:
            for s in active:
                s.step += 1
            return pred_feat

        # Advance only the continuing rows (one new position each).
        idx = torch.tensor(cont_idx, device=self.device, dtype=torch.long)
        slots = torch.tensor([active[i].slot for i in cont_idx], device=self.device, dtype=torch.long)
        positions = torch.tensor([active[i].pos for i in cont_idx], device=self.device, dtype=torch.long)

        barbet_in = curr_embed_tslm[idx, 0, :]                                 # [c, h_b]
        barbet_step_hidden = self.barbet.decode_step(barbet_in, slots, positions)
        new_lm_hidden = m.fsq_layer(m.tslm_adapter(barbet_step_hidden))        # [c, h_v]

        residual_in = m.fusion_concat_proj(torch.cat((new_lm_hidden, curr_embed_lm[idx, 0, :]), dim=-1))
        new_residual_hidden = self.ralm.decode_step(residual_in, slots, positions)

        for j, i in enumerate(cont_idx):
            s = active[i]
            s.lm_hidden = new_lm_hidden[j : j + 1]
            s.residual_hidden = new_residual_hidden[j : j + 1]
            s.pos += 1
            s.step += 1
        cont_set = set(cont_idx)
        for i, s in enumerate(active):
            if i not in cont_set:
                s.step += 1
        return pred_feat

    # ------------------------------------------------------------------ #
    def collect_latents(self, s: SeqState) -> torch.Tensor:
        """Concatenate a finished sequence's emitted patches -> [T, p, d]."""
        return torch.cat(s.patches, dim=0)
