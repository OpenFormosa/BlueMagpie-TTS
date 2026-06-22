"""Configuration for BlueMagpie-TTS.

BlueMagpie = Barbet (TSLM) + VoxCPM2 acoustic stack (LocEnc / RALM / LocDiT / AudioVAE).

The config keeps two "hidden spaces" explicit:

- ``barbet_config``         -> Barbet's hidden space  (H_b, e.g. 1024 for 300M, 1536 for 1B)
- ``vox_lm_config``         -> VoxCPM2's semantic LM hidden space (H_v, 2048 for openbmb/VoxCPM2)

``vox_lm_config`` is the MiniCPM4 config of the *original* VoxCPM2 TSLM. The full
28-layer TSLM is never instantiated here — the config object is kept because it is
the template from which VoxCPM2 derives its RALM / LocEnc / LocDiT submodule
configs, and because ``hidden_size`` defines the space all pretrained projection
layers (fsq, fusion_concat_proj, lm_to_dit_proj, ...) live in.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from bluemagpie._vendor.voxcpm.model.voxcpm2 import VoxCPMDitConfig, VoxCPMEncoderConfig
from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEConfigV2
from bluemagpie._vendor.voxcpm.modules.minicpm4 import MiniCPM4Config


class AdapterConfig(BaseModel):
    """Projection adapter bridging Barbet hidden space into VoxCPM2 LM space."""

    num_residual_blocks: int = 1
    ffn_mult: float = 2.0
    rms_norm_eps: float = 1.0e-6


class BlueMagpieConfig(BaseModel):
    # Raw kwargs for barbet.BarbetConfig (kept as dict so the pydantic model
    # stays serializable independently of transformers).
    barbet_config: dict

    # Semantic-space template (the original VoxCPM2 TSLM config).
    vox_lm_config: MiniCPM4Config

    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfigV2] = None

    adapter_config: AdapterConfig = AdapterConfig()

    # Dimensionality of the speaker centroid (ECAPA-TDNN default = 192). The
    # SpeakerProjector maps this into the Barbet hidden space at the [spk] slot.
    speaker_embed_dim: int = 192

    # Special token ids in *Barbet* vocab space. -1 means "allocate the first
    # free id after the effective vocab" (resolved by resolve_barbet_config()).
    audio_start_token: int = -1
    audio_end_token: int = -1
    ref_audio_start_token: int = -1
    ref_audio_end_token: int = -1
    # Placeholder slot whose embedding row is overwritten by the projected
    # speaker centroid (and serves as the learned "null speaker" when dropped).
    spk_token: int = -1

    # Number of ids actually used by the tokenizer. Barbet R2 checkpoints pad
    # the embedding to a multiple of 128 for Megatron (vocab_size 114944 vs
    # effective 114822 for PangolinTokenizer), so auto-allocated special
    # tokens can live in the padding region without growing the embedding.
    # None -> auto-detect from the Pangolin contract, else fall back to
    # vocab_size (append-and-grow).
    barbet_effective_vocab_size: Optional[int] = None

    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"

    def _effective_vocab_size(self, barbet_cfg) -> int:
        """First id available for auto-allocated special tokens."""
        if self.barbet_effective_vocab_size is not None:
            return self.barbet_effective_vocab_size
        try:
            from barbet.configuration_barbet import EFFECTIVE_VOCAB_SIZE, MEGATRON_PADDED_VOCAB_SIZE

            # Pangolin contract: ids [EFFECTIVE, PADDED) are Megatron padding
            # rows the tokenizer never produces — free for our special tokens.
            if barbet_cfg.vocab_size == MEGATRON_PADDED_VOCAB_SIZE:
                return EFFECTIVE_VOCAB_SIZE
        except ImportError:  # pre-R2 barbet without the contract constants
            pass
        return barbet_cfg.vocab_size

    def resolve_barbet_config(self):
        """Build the BarbetConfig and resolve special-token ids.

        Returns:
            (barbet_config, token_ids) where token_ids is a dict with keys
            audio_start/audio_end/ref_audio_start/ref_audio_end/spk. Ids set to
            -1 are allocated from the first free id after the effective vocab —
            inside the Megatron padding region for R2 checkpoints (no
            embedding growth), or appended after the vocab otherwise. The
            vocab only grows when an id falls beyond the current size.
        """
        from barbet import BarbetConfig

        kwargs = dict(self.barbet_config)
        barbet_cfg = BarbetConfig(**kwargs)

        names = ["audio_start", "audio_end", "ref_audio_start", "ref_audio_end", "spk"]
        requested = [
            self.audio_start_token,
            self.audio_end_token,
            self.ref_audio_start_token,
            self.ref_audio_end_token,
            self.spk_token,
        ]
        token_ids = {}
        next_id = self._effective_vocab_size(barbet_cfg)
        for name, tok in zip(names, requested):
            if tok is None or tok < 0:
                token_ids[name] = next_id
                next_id += 1
            else:
                token_ids[name] = tok
        needed = max(token_ids.values()) + 1
        if needed > barbet_cfg.vocab_size:
            barbet_cfg.vocab_size = needed
        return barbet_cfg, token_ids


BlueMagpieConfig.model_rebuild()
