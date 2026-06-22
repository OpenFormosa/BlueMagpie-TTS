"""BlueMagpie-TTS hybrid model.

Architecture = VoxCPM2 with its Text-Semantic LM (MiniCPM4) swapped for Barbet:

    text tokens ──Barbet.embed_tokens──────────────┐
                                                   ├─ interleave ─> Barbet backbone ─> H_b
    audio latents ─LocEnc─┬─ enc_to_tslm_proj(H_b)─┘                    │
                          └─ enc_to_lm_proj(H_v)──────────┐      tslm_adapter (H_b -> H_v)
                                                          │             │
                                                          │       FSQ (audio positions)
                                                          │             │ enc_outputs (H_v)
                                  fusion_concat_proj(cat) ┴─────────────┤
                                            │                           │
                                       RALM (MiniCPM4 8L)         lm_to_dit_proj
                                            │ res_to_dit_proj           │
                                            └───────── concat ──────────┘
                                                          │ mu (2 prefix tokens)
                                                     LocDiT (CFM)
                                                          │ latent patch
                                                      AudioVAE

Division of labour (matching VoxCPM2's design intent):
- Barbet (TSLM): what to say, prosody planning, pacing, emphasis, control text.
- RALM + LocDiT: fine-grained acoustic detail, kept verbatim from VoxCPM2 so
  pretrained weights load unchanged.
- ``tslm_adapter`` is the explicit bridge between the two incompatible hidden
  spaces; ``enc_to_tslm_proj`` is its input-side counterpart feeding LocEnc
  features into Barbet.

Everything outside the TSLM block mirrors
``voxcpm/model/voxcpm2.py`` (Apache-2.0, Copyright 2026 OpenBMB) so that
VoxCPM2 checkpoints remain loadable.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Generator, List, Optional, Tuple, Union

import librosa
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from bluemagpie._vendor.voxcpm.model.utils import get_dtype, next_and_close, pick_runtime_dtype, resolve_runtime_device
from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2
from bluemagpie._vendor.voxcpm.modules.layers import ScalarQuantizationLayer
from bluemagpie._vendor.voxcpm.modules.locdit import UnifiedCFM, VoxCPMLocDiTV2
from bluemagpie._vendor.voxcpm.modules.locenc import VoxCPMLocEnc
from bluemagpie._vendor.voxcpm.modules.minicpm4 import MiniCPMModel

from .adapter import ProjectionAdapter
from .conditioning import SpeakerProjector
from .config import BlueMagpieConfig
from .tslm import BarbetTSLM


class BlueMagpieModel(nn.Module):
    def __init__(
        self,
        config: BlueMagpieConfig,
        tokenizer=None,
        audio_vae: AudioVAEV2 = None,
        device: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = resolve_runtime_device(device, config.device)
        self.config.device = self.device
        resolved_dtype = pick_runtime_dtype(self.device, self.config.dtype)
        if resolved_dtype != self.config.dtype:
            print(
                f"[bluemagpie] adjusted dtype {self.config.dtype} -> {resolved_dtype} for device {self.device}",
                file=sys.stderr,
            )
            self.config.dtype = resolved_dtype

        vox_lm = config.vox_lm_config
        h_vox = vox_lm.hidden_size

        # ---------------- Text-Semantic LM: Barbet ---------------- #
        barbet_cfg, token_ids = config.resolve_barbet_config()
        self.barbet_config = barbet_cfg
        self.base_lm = BarbetTSLM(barbet_cfg)
        h_barbet = barbet_cfg.hidden_size

        self.text_tokenizer = tokenizer
        self.audio_start_token = token_ids["audio_start"]
        self.audio_end_token = token_ids["audio_end"]
        self.ref_audio_start_token = token_ids["ref_audio_start"]
        self.ref_audio_end_token = token_ids["ref_audio_end"]
        self.spk_token = token_ids["spk"]

        # ---------------- Residual Acoustic LM (VoxCPM2) ---------------- #
        residual_lm_config = vox_lm.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        residual_lm_config.no_rope = config.residual_lm_no_rope
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # ---------------- Local Encoder (VoxCPM2) ---------------- #
        encoder_config = vox_lm.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # ---------------- Local DiT (VoxCPM2) ---------------- #
        decoder_config = vox_lm.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiTV2(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_config.dit_mean_mode,
        )

        # ---------------- VoxCPM2 projections (semantic space H_v) ---------------- #
        self.fsq_layer = ScalarQuantizationLayer(
            h_vox, h_vox, config.scalar_quantization_latent_dim, config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, h_vox)
        self.lm_to_dit_proj = nn.Linear(h_vox, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(h_vox, config.dit_config.hidden_dim)
        self.fusion_concat_proj = nn.Linear(h_vox * 2, h_vox)

        # Stop Predictor (semantic space)
        self.stop_proj = nn.Linear(h_vox, h_vox)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(h_vox, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # ---------------- Barbet <-> VoxCPM2 bridges (new, trained from scratch) -------- #
        self.enc_to_tslm_proj = nn.Linear(config.encoder_config.hidden_dim, h_barbet)
        self.tslm_adapter = ProjectionAdapter(h_barbet, h_vox, config.adapter_config)

        # ---------------- Speaker conditioning (v1: centroid -> [spk] slot) ------------ #
        self.speaker_embed_dim = config.speaker_embed_dim
        self.speaker_projector = SpeakerProjector(config.speaker_embed_dim, h_barbet)

        # ---------------- Audio VAE ---------------- #
        self.audio_vae = audio_vae
        if audio_vae is not None:
            self.chunk_size = audio_vae.chunk_size
            self._decode_chunk_size = getattr(audio_vae, "decode_chunk_size", audio_vae.chunk_size)
            self._encode_sample_rate = audio_vae.sample_rate
            self.sample_rate = getattr(audio_vae, "out_sample_rate", audio_vae.sample_rate)

    def _dtype(self):
        return get_dtype(self.config.dtype)

    def _runtime_dtype(self) -> torch.dtype:
        """Return the dtype the non-AudioVAE network is currently using."""
        return next(self.parameters()).dtype

    def _runtime_device(self) -> torch.device:
        """Return the device the module is actually placed on.

        Distributed wrappers may move parameters after construction. Keep the
        cached device string in sync so Triton-backed kernels launch on the
        same CUDA device as their inputs.
        """
        device = next(self.parameters()).device
        if device.type == "cuda" and device.index is not None and torch.cuda.current_device() != device.index:
            torch.cuda.set_device(device)
        device_str = str(device)
        if device_str != str(self.device):
            self.device = device_str
            self.config.device = device_str
        return device

    def _inject_speaker(self, combined_embed, speaker_centroids, spk_mask):
        """Add the projected speaker centroid at the [spk] position.

        ``combined_embed``: [B, T, H_b]. ``speaker_centroids``: [B, D] (the
        per-speaker ECAPA centroid). ``spk_mask``: [B, T], 1 only at the [spk]
        slot of samples whose speaker is present (not dropped). Where
        ``spk_mask`` is 0 the slot keeps its [spk] token embedding, which acts
        as the learned null speaker.
        """
        if speaker_centroids is None or spk_mask is None:
            return combined_embed
        dtype = combined_embed.dtype
        spk_vec = self.speaker_projector(speaker_centroids.to(combined_embed.device, dtype=dtype))  # [B, H_b]
        spk_mask = spk_mask.to(combined_embed.device, dtype=dtype)
        return combined_embed + spk_mask.unsqueeze(-1) * spk_vec.unsqueeze(1)

    def _tokenize(self, text: str) -> List[int]:
        if self.text_tokenizer is None:
            raise ValueError("No tokenizer attached to BlueMagpieModel")
        if hasattr(self.text_tokenizer, "encode"):
            return self.text_tokenizer.encode(text, add_special_tokens=False)
        return self.text_tokenizer(text)

    # ------------------------------------------------------------------ #
    # Training forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        speaker_centroids: Optional[torch.Tensor] = None,
        spk_mask: Optional[torch.Tensor] = None,
        progress: float = 0.0,
        sample_generate: bool = False,
        sample_generate_timesteps: int = 10,
    ):
        del position_ids  # not used (parity with VoxCPM2)

        device = self._runtime_device()
        text_tokens = text_tokens.to(device, dtype=torch.long)
        text_mask = text_mask.to(device, dtype=self._dtype())
        audio_feats = audio_feats.to(device, dtype=self._dtype())
        audio_mask = audio_mask.to(device, dtype=self._dtype())
        loss_mask = loss_mask.to(device, dtype=self._dtype())
        labels = labels.to(device, dtype=torch.long)

        B, T, P, D = audio_feats.shape
        feat_locenc = self.feat_encoder(audio_feats)
        feat_embed_tslm = self.enc_to_tslm_proj(feat_locenc)  # Barbet input space
        feat_embed_lm = self.enc_to_lm_proj(feat_locenc)  # VoxCPM2 semantic space (RALM fusion)

        text_embed = self.base_lm.embed_tokens(text_tokens)
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed_tslm
        combined_embed = self._inject_speaker(combined_embed, speaker_centroids, spk_mask)

        barbet_hidden = self.base_lm(inputs_embeds=combined_embed)
        tslm_hidden = self.tslm_adapter(barbet_hidden).to(self._dtype())

        enc_outputs = self.fsq_layer(tslm_hidden) * audio_mask.unsqueeze(-1) + tslm_hidden * text_mask.unsqueeze(-1)
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        residual_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed_lm), dim=-1)
        )
        residual_outputs, _ = self.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = torch.cat((self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)), dim=-1)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        target_dtype = self._dtype()
        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
        )

        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=max(int(sample_generate_timesteps), 1),
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
            "stop_logits": stop_logits,
            # Pre-FSQ adapted hidden states, for hidden-space distillation
            # against the original VoxCPM2 TSLM (see README, stage 0).
            "tslm_hidden": tslm_hidden,
        }

    # ------------------------------------------------------------------ #
    # Input assembly (the four VoxCPM2 prompting modes)
    # ------------------------------------------------------------------ #
    def _encode_wav(
        self,
        wav_path: str,
        padding_mode: str = "right",
    ) -> torch.Tensor:
        audio, _ = librosa.load(wav_path, sr=self._encode_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = torch.nn.functional.pad(audio, pad)
        device = self._runtime_device()
        feat = self.audio_vae.encode(audio.to(device), self._encode_sample_rate).cpu()
        return feat.view(self.audio_vae.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _make_ref_prefix(self, ref_feat: torch.Tensor, device: torch.device):
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=device)
        tokens = torch.cat(
            [
                torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
            ]
        )
        feats = torch.cat([z1, ref_feat, z1], dim=0)
        t_mask = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(ref_len, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            ]
        ).to(device)
        a_mask = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.ones(ref_len, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ]
        ).to(device)
        return tokens, feats, t_mask, a_mask

    def _build_inputs(
        self,
        text: str,
        ref_feat: Optional[torch.Tensor] = None,
        prompt_feat: Optional[torch.Tensor] = None,
        speaker_slot: str = "none",
    ):
        """Assemble (text_token, audio_feat, text_mask, audio_mask, spk_mask).

        Layout: [spk?] [ref prefix?] [text + audio_start] [prompt audio?]

        ``speaker_slot`` is "none", "null", or "centroid". The null slot keeps
        the learned [spk] token embedding, matching speaker-dropout training.
        The centroid slot is filled by the projected speaker vector.
        """
        if speaker_slot not in {"none", "null", "centroid"}:
            raise ValueError(f"speaker_slot must be one of none/null/centroid, got {speaker_slot!r}")

        text_token = torch.LongTensor(self._tokenize(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.long, device=text_token.device),
            ],
            dim=-1,
        )
        text_length = text_token.shape[0]
        device = text_token.device

        text_pad_feat = torch.zeros(
            (text_length, self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=device,
        )

        tokens = [text_token]
        feats = [text_pad_feat]
        t_masks = [torch.ones(text_length, dtype=torch.int32, device=device)]
        a_masks = [torch.zeros(text_length, dtype=torch.int32, device=device)]
        s_masks = [torch.zeros(text_length, dtype=torch.int32, device=device)]

        if ref_feat is not None:
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, device)
            tokens.insert(0, ref_tokens.long())
            feats.insert(0, ref_feats)
            t_masks.insert(0, ref_t_mask)
            a_masks.insert(0, ref_a_mask)
            s_masks.insert(0, torch.zeros(ref_tokens.shape[0], dtype=torch.int32, device=device))

        if speaker_slot != "none":
            one_feat = torch.zeros((1, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=device)
            tokens.insert(0, torch.tensor([self.spk_token], dtype=torch.long, device=device))
            feats.insert(0, one_feat)
            t_masks.insert(
                0,
                torch.ones(1, dtype=torch.int32, device=device)
                if speaker_slot == "null"
                else torch.zeros(1, dtype=torch.int32, device=device),
            )
            a_masks.insert(0, torch.zeros(1, dtype=torch.int32, device=device))
            s_masks.insert(
                0,
                torch.ones(1, dtype=torch.int32, device=device)
                if speaker_slot == "centroid"
                else torch.zeros(1, dtype=torch.int32, device=device),
            )

        if prompt_feat is not None:
            prompt_len = prompt_feat.size(0)
            tokens.append(torch.zeros(prompt_len, dtype=torch.long, device=device))
            feats.append(prompt_feat)
            t_masks.append(torch.zeros(prompt_len, dtype=torch.int32, device=device))
            a_masks.append(torch.ones(prompt_len, dtype=torch.int32, device=device))
            s_masks.append(torch.zeros(prompt_len, dtype=torch.int32, device=device))

        return (
            torch.cat(tokens),
            torch.cat(feats, dim=0),
            torch.cat(t_masks),
            torch.cat(a_masks),
            torch.cat(s_masks),
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        speaker_centroid: Optional[torch.Tensor] = None,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        use_null_speaker: bool = True,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        device = self._runtime_device()
        ref_feat = self._encode_wav(reference_wav_path, padding_mode="right") if reference_wav_path else None
        prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left") if prompt_wav_path else None
        text = (prompt_text + target_text) if prompt_feat is not None else target_text

        speaker_centroids = None
        if speaker_centroid is not None:
            speaker_centroids = speaker_centroid.reshape(1, -1).to(device, dtype=self._runtime_dtype())

        speaker_slot = "centroid" if speaker_centroids is not None else ("null" if use_null_speaker else "none")
        text_token, audio_feat, text_mask, audio_mask, spk_mask = self._build_inputs(
            text, ref_feat, prompt_feat, speaker_slot=speaker_slot
        )

        text_token = text_token.unsqueeze(0).to(device)
        text_mask = text_mask.unsqueeze(0).to(device, dtype=self._runtime_dtype())
        audio_feat = audio_feat.unsqueeze(0).to(device, dtype=self._runtime_dtype())
        audio_mask = audio_mask.unsqueeze(0).to(device, dtype=self._runtime_dtype())
        spk_mask = spk_mask.unsqueeze(0).to(device, dtype=self._runtime_dtype())

        target_text_length = len(self._tokenize(target_text))

        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                speaker_centroids=speaker_centroids,
                spk_mask=spk_mask,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                with self.audio_vae.streaming_decode() as vae_dec:
                    for latent_pred, _, _ctx in inference_result:
                        decode_audio = vae_dec.decode_chunk(latent_pred.to(torch.float32))
                        yield decode_audio.squeeze(1).cpu()
                break
            else:
                latent_pred, pred_audio_feat, context_len = next_and_close(inference_result)
                if retry_badcase and pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                    print(
                        f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                        file=sys.stderr,
                    )
                    retry_badcase_times += 1
                    continue
                break

        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            decode_patch_len = self.patch_size * self._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len :].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield decode_audio

    # ------------------------------------------------------------------ #
    # Core AR loop
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        speaker_centroids: Optional[torch.Tensor] = None,
        spk_mask: Optional[torch.Tensor] = None,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]], int], None, None]:
        device = self._runtime_device()
        dtype = self._runtime_dtype()
        text = text.to(device)
        text_mask = text_mask.to(device, dtype=dtype)
        feat = feat.to(device, dtype=dtype)
        feat_mask = feat_mask.to(device, dtype=dtype)
        if speaker_centroids is not None:
            speaker_centroids = speaker_centroids.to(device, dtype=dtype)
        if spk_mask is not None:
            spk_mask = spk_mask.to(device, dtype=dtype)
        B, T, P, D = feat.shape

        feat_locenc = self.feat_encoder(feat)  # [b, t, h_enc]
        feat_embed_tslm = self.enc_to_tslm_proj(feat_locenc)
        feat_embed_lm = self.enc_to_lm_proj(feat_locenc)

        text_embed = self.base_lm.embed_tokens(text)
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed_tslm
        combined_embed = self._inject_speaker(combined_embed, speaker_centroids, spk_mask)

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        curr_embed = None

        # Streaming context patches (continuation modes only)
        has_continuation_audio = feat_mask[0, -1].item() == 1
        context_len = 0
        if has_continuation_audio:
            audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_audio_indices = audio_indices[-context_len:]
            pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
        else:
            pred_feat_seq = []

        # --- TSLM prefill (Barbet stepwise cache) --- #
        barbet_hidden_seq, tslm_state = self.base_lm.prefill(combined_embed)
        tslm_hidden_seq = self.tslm_adapter(barbet_hidden_seq)

        enc_outputs = (
            self.fsq_layer(tslm_hidden_seq) * feat_mask.unsqueeze(-1) + tslm_hidden_seq * text_mask.unsqueeze(-1)
        )
        lm_hidden = enc_outputs[:, -1, :]

        # --- RALM prefill (MiniCPM static KV cache) --- #
        residual_enc_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed_lm), dim=-1)
        )
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(1, 2)  # [b, p, d]

            curr_locenc = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, h_enc
            curr_embed_tslm = self.enc_to_tslm_proj(curr_locenc)
            curr_embed_lm = self.enc_to_lm_proj(curr_locenc)
            curr_embed = curr_embed_tslm  # naming parity with VoxCPM2

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                feat_pred = rearrange(pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size)
                yield feat_pred, pred_feat_seq, context_len
                if len(pred_feat_seq) > streaming_prefix_len:
                    pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break

            barbet_step_hidden = self.base_lm.forward_step(curr_embed[:, 0, :], tslm_state)
            lm_hidden = self.fsq_layer(self.tslm_adapter(barbet_step_hidden))

            curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed_lm[:, 0, :]), dim=-1))
            residual_hidden = self.residual_lm.forward_step(
                curr_residual_input,
                torch.tensor([self.residual_lm.kv_cache.step()], device=curr_residual_input.device),
            ).clone()

        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
            yield feat_pred, generated_feat, context_len

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_pred, generated_feat, _ = next_and_close(self._inference(*args, streaming=False, **kwargs))
        return feat_pred, generated_feat

    # ------------------------------------------------------------------ #
    # Checkpoint I/O (BlueMagpie's own format)
    # ------------------------------------------------------------------ #
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.config.model_dump_json(indent=2))
        state = {k: v for k, v in self.state_dict().items() if not k.startswith("audio_vae.")}
        torch.save(state, os.path.join(path, "pytorch_model.bin"))
        if self.audio_vae is not None:
            torch.save(self.audio_vae.state_dict(), os.path.join(path, "audiovae.pth"))
        if self.text_tokenizer is not None and hasattr(self.text_tokenizer, "save_pretrained"):
            self.text_tokenizer.save_pretrained(path)

    @classmethod
    def from_local(
        cls,
        path: str,
        tokenizer=None,
        training: bool = False,
        device: str | None = None,
    ) -> "BlueMagpieModel":
        from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2

        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = BlueMagpieConfig.model_validate_json(f.read())

        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception:
                tokenizer = None

        audio_vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
        vae_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(vae_path):
            vae_state = torch.load(vae_path, map_location="cpu", weights_only=True)
            audio_vae.load_state_dict(vae_state.get("state_dict", vae_state))

        model = cls(config, tokenizer, audio_vae, device=device)
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("audio_vae.")]
        if missing or unexpected:
            print(f"[bluemagpie] missing keys: {missing[:8]}... unexpected: {unexpected[:8]}...", file=sys.stderr)

        if not training:
            model = model.to(get_dtype(model.config.dtype))
        model.audio_vae = model.audio_vae.to(torch.float32)
        return model.to(model.device).eval() if not training else model.to(model.device)
