"""Shared tiny-config builders and stubs for CPU tests."""

import torch
from torch import nn

from bluemagpie._vendor.voxcpm.model.voxcpm2 import VoxCPMDitConfig, VoxCPMEncoderConfig
from bluemagpie._vendor.voxcpm.modules.locdit import CfmConfig
from bluemagpie._vendor.voxcpm.modules.minicpm4 import MiniCPM4Config
from bluemagpie._vendor.voxcpm.modules.minicpm4.config import RopeScalingConfig

from bluemagpie import BlueMagpieConfig

KV_CHANNELS = 8  # rope factor lists must have kv_channels / 2 entries


def tiny_vox_lm_config() -> MiniCPM4Config:
    return MiniCPM4Config(
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_scaling=RopeScalingConfig(
            type="longrope",
            long_factor=[1.0] * (KV_CHANNELS // 2),
            short_factor=[1.0] * (KV_CHANNELS // 2),
            original_max_position_embeddings=256,
        ),
        vocab_size=100,
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=32,
        scale_depth=1.0,
        rope_theta=10000.0,
        kv_channels=KV_CHANNELS,
    )


def tiny_config() -> BlueMagpieConfig:
    return BlueMagpieConfig(
        barbet_config=dict(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            sliding_window_size=6,
            global_attention_layers=(0,),
            mamba_layers=(2,),
            qk_norm=True,
            qk_logit_clip=True,
            attention_sink=True,
            mamba_d_conv=4,
            mamba_expand=2,
            mtp_enabled=False,
        ),
        vox_lm_config=tiny_vox_lm_config(),
        patch_size=2,
        feat_dim=8,
        residual_lm_num_layers=2,
        residual_lm_no_rope=True,
        scalar_quantization_latent_dim=16,
        scalar_quantization_scale=9,
        encoder_config=VoxCPMEncoderConfig(
            hidden_dim=16, ffn_dim=32, num_heads=2, num_layers=1, kv_channels=KV_CHANNELS
        ),
        dit_config=VoxCPMDitConfig(
            hidden_dim=16,
            ffn_dim=32,
            num_heads=2,
            num_layers=1,
            kv_channels=KV_CHANNELS,
            cfm_config=CfmConfig(sigma_min=1e-6, solver="euler", t_scheduler="log-norm", inference_cfg_rate=2.0),
        ),
        audio_vae_config=None,
        speaker_embed_dim=16,
        max_length=64,
        device="cpu",
        dtype="float32",
    )


class StubVAE(nn.Module):
    """Deterministic AudioVAE stand-in matching the packer/model interface.

    encode() pools each hop window to its mean and broadcasts it across the
    latent dim, so both student and teacher packs of the same waveform produce
    identical latents (what the distillation alignment relies on).
    """

    def __init__(self, latent_dim: int = 8, hop_length: int = 160, sample_rate: int = 16_000):
        super().__init__()
        self.latent_dim = latent_dim
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.chunk_size = hop_length

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sample_rate=None) -> torch.Tensor:
        if wav.dim() == 2:  # [B, T] -> [B, 1, T]
            wav = wav.unsqueeze(1)
        B, _, T = wav.shape
        n = T // self.hop_length
        frames = wav[:, 0, : n * self.hop_length].reshape(B, n, self.hop_length).mean(-1)
        return frames.unsqueeze(1).repeat(1, self.latent_dim, 1)  # [B, D, n]
