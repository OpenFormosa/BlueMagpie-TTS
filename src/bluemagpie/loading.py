"""Checkpoint assembly: pretrained VoxCPM2 acoustic stack + Barbet TSLM.

The hybrid is built by weight surgery:

- from the VoxCPM2 checkpoint we keep every module *except* the TSLM
  (``base_lm.*`` keys are dropped): LocEnc, RALM, LocDiT, FSQ, the projection
  layers, the stop head and the AudioVAE.
- Barbet weights are loaded into ``base_lm.backbone`` (with the 4 appended
  special-token embedding rows freshly initialized).
- the two bridge modules (``enc_to_tslm_proj``, ``tslm_adapter``) are always
  freshly initialized — they are what training stage 0/1 is for.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import torch

from bluemagpie._vendor.voxcpm.model.voxcpm2 import VoxCPMConfig
from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2

from .config import AdapterConfig, BlueMagpieConfig
from .model import BlueMagpieModel

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SAFETENSORS_AVAILABLE = False

# VoxCPM2 modules reused verbatim (state-dict key prefixes).
VOXCPM2_REUSED_PREFIXES = (
    "feat_encoder.",
    "residual_lm.",
    "feat_decoder.",
    "fsq_layer.",
    "enc_to_lm_proj.",
    "lm_to_dit_proj.",
    "res_to_dit_proj.",
    "fusion_concat_proj.",
    "stop_proj.",
    "stop_head.",
)


def _load_state_dict(path_dir: str, stem: str) -> dict:
    st_path = os.path.join(path_dir, f"{stem}.safetensors")
    pt_candidates = [os.path.join(path_dir, f"{stem}.pth"), os.path.join(path_dir, f"{stem}.bin")]
    if stem == "model":
        pt_candidates.insert(0, os.path.join(path_dir, "pytorch_model.bin"))
    if os.path.exists(st_path) and SAFETENSORS_AVAILABLE:
        return load_file(st_path, device="cpu")
    for p in pt_candidates:
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            return ckpt.get("state_dict", ckpt)
    raise FileNotFoundError(f"No checkpoint found for '{stem}' under {path_dir}")


def build_config_from_voxcpm2(
    voxcpm2_path: str,
    barbet_config_kwargs: dict,
    adapter_config: Optional[AdapterConfig] = None,
    **overrides,
) -> BlueMagpieConfig:
    """Derive a BlueMagpieConfig from a local VoxCPM2 checkpoint directory."""
    with open(os.path.join(voxcpm2_path, "config.json"), "r", encoding="utf-8") as f:
        vox_cfg = VoxCPMConfig.model_validate_json(f.read())

    cfg = BlueMagpieConfig(
        barbet_config=barbet_config_kwargs,
        vox_lm_config=vox_cfg.lm_config,
        patch_size=vox_cfg.patch_size,
        feat_dim=vox_cfg.feat_dim,
        residual_lm_num_layers=vox_cfg.residual_lm_num_layers,
        residual_lm_no_rope=vox_cfg.residual_lm_no_rope,
        scalar_quantization_latent_dim=vox_cfg.scalar_quantization_latent_dim,
        scalar_quantization_scale=vox_cfg.scalar_quantization_scale,
        encoder_config=vox_cfg.encoder_config,
        dit_config=vox_cfg.dit_config,
        audio_vae_config=vox_cfg.audio_vae_config,
        adapter_config=adapter_config or AdapterConfig(),
        max_length=vox_cfg.max_length,
        device=vox_cfg.device,
        dtype=vox_cfg.dtype,
        **overrides,
    )
    return cfg


def load_barbet_config_kwargs(barbet_path: str) -> dict:
    """Read a Barbet HF checkpoint's config.json into plain kwargs."""
    with open(os.path.join(barbet_path, "config.json"), "r", encoding="utf-8") as f:
        kwargs = json.load(f)
    for k in ("architectures", "auto_map", "model_type", "transformers_version", "torch_dtype"):
        kwargs.pop(k, None)
    return kwargs


def build_from_pretrained(
    voxcpm2_path: str,
    barbet_path: Optional[str] = None,
    barbet_config_kwargs: Optional[dict] = None,
    tokenizer=None,
    adapter_config: Optional[AdapterConfig] = None,
    device: str | None = None,
    training: bool = True,
    **config_overrides,
) -> BlueMagpieModel:
    """Assemble a BlueMagpieModel from pretrained VoxCPM2 + Barbet checkpoints.

    Args:
        voxcpm2_path: local directory of the VoxCPM2 checkpoint
            (e.g. ``huggingface_hub.snapshot_download("openbmb/VoxCPM2")``).
        barbet_path: local directory of a Barbet checkpoint. If None, the TSLM
            is randomly initialized from ``barbet_config_kwargs``.
        barbet_config_kwargs: Barbet config kwargs; defaults to the checkpoint's
            config.json (if ``barbet_path`` given) or BarbetConfig defaults.
        tokenizer: HF tokenizer matching Barbet's vocab. Defaults to
            ``AutoTokenizer.from_pretrained(barbet_path)`` when available.
        adapter_config: projection adapter hyperparameters.
        training: if True, returns a float32 trainable model (VAE frozen);
            if False, casts to the configured inference dtype and eval().
    """
    if barbet_config_kwargs is None:
        if barbet_path is not None:
            barbet_config_kwargs = load_barbet_config_kwargs(barbet_path)
        else:
            barbet_config_kwargs = {}

    config = build_config_from_voxcpm2(
        voxcpm2_path, barbet_config_kwargs, adapter_config=adapter_config, **config_overrides
    )

    if tokenizer is None and barbet_path is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(barbet_path)
        except Exception:
            print("[bluemagpie] no tokenizer found at barbet_path; attach one manually", file=sys.stderr)

    # Audio VAE from the VoxCPM2 checkpoint
    audio_vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
    vae_state = _load_state_dict(voxcpm2_path, "audiovae")
    audio_vae.load_state_dict(vae_state)

    model = BlueMagpieModel(config, tokenizer, audio_vae, device=device)

    # ---- VoxCPM2 weight surgery (drop base_lm.*) ---- #
    vox_state = _load_state_dict(voxcpm2_path, "model")
    kept = {k: v for k, v in vox_state.items() if k.startswith(VOXCPM2_REUSED_PREFIXES)}
    dropped = len(vox_state) - len(kept)
    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(
        f"[bluemagpie] VoxCPM2 surgery: kept {len(kept)} tensors, dropped {dropped} (base_lm.* etc.), "
        f"unexpected {len(unexpected)}",
        file=sys.stderr,
    )

    # ---- Barbet TSLM weights ---- #
    if barbet_path is not None:
        load_barbet_weights(model, barbet_path)

    model.audio_vae = model.audio_vae.to(torch.float32)
    for name, param in model.named_parameters():
        if name.startswith("audio_vae."):
            param.requires_grad = False

    if not training:
        from bluemagpie._vendor.voxcpm.model.utils import get_dtype

        non_vae = [m for n, m in model.named_children() if n != "audio_vae"]
        for m in non_vae:
            m.to(get_dtype(model.config.dtype))
        model = model.to(model.device).eval()
    else:
        model = model.to(model.device)
    return model


def load_barbet_weights(model: BlueMagpieModel, barbet_path: str) -> None:
    """Load Barbet checkpoint into model.base_lm.backbone.

    Accepts both BarbetModel and BarbetForCausalLM state dicts (the ``model.``
    prefix is stripped; ``lm_head``/``mtp`` keys are dropped — R2 checkpoints
    tie lm_head to the embeddings anyway). The embedding rows of the resolved
    TTS special tokens are always re-initialized from N(0, initializer_range):
    for R2 checkpoints those ids sit in the Megatron padding region, whose
    checkpoint rows are typically all-zero and would otherwise make
    audio_start/end/ref markers indistinguishable.
    """
    state = _load_state_dict(barbet_path, "model")
    cleaned = {}
    for k, v in state.items():
        if k.startswith("lm_head.") or k.startswith("mtp."):
            continue
        cleaned[k.removeprefix("model.")] = v

    backbone = model.base_lm.backbone
    target_embed = backbone.embed_tokens.weight
    src_embed = cleaned.get("embed_tokens.weight")
    if src_embed is not None and src_embed.shape[0] != target_embed.shape[0]:
        n_src = src_embed.shape[0]
        grown = target_embed.detach().clone()
        grown[:n_src] = src_embed
        cleaned["embed_tokens.weight"] = grown
        print(
            f"[bluemagpie] resized Barbet embeddings {n_src} -> {target_embed.shape[0]}",
            file=sys.stderr,
        )

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    print(
        f"[bluemagpie] Barbet load: missing {len(missing)}, unexpected {len(unexpected)}",
        file=sys.stderr,
    )

    special_ids = [
        model.audio_start_token,
        model.audio_end_token,
        model.ref_audio_start_token,
        model.ref_audio_end_token,
        model.spk_token,
    ]
    std = model.barbet_config.initializer_range
    with torch.no_grad():
        fresh = torch.empty(len(special_ids), target_embed.shape[1], dtype=target_embed.dtype)
        fresh.normal_(mean=0.0, std=std)
        backbone.embed_tokens.weight[special_ids] = fresh.to(target_embed.device)
    print(f"[bluemagpie] re-initialized special token rows {special_ids}", file=sys.stderr)


# ---------------------------------------------------------------------- #
# Stage-wise freezing
# ---------------------------------------------------------------------- #
STAGES = ("bridge", "tslm", "full")


def set_training_stage(model: BlueMagpieModel, stage: str) -> None:
    """Configure requires_grad for the staged training recipe.

    - ``bridge``: only the new bridge modules train (enc_to_tslm_proj +
      tslm_adapter). Use with hidden-space distillation against the original
      VoxCPM2 TSLM, or with the diffusion loss directly.
    - ``tslm``:   bridge + Barbet + speaker projector train; the VoxCPM2
      acoustic stack stays frozen (RALM / LocDiT keep handling acoustic detail
      unchanged). This is where v1 speaker conditioning is learned.
    - ``full``:   everything trains except the AudioVAE.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    bridge_prefixes = ("enc_to_tslm_proj.", "tslm_adapter.")
    tslm_prefixes = bridge_prefixes + ("base_lm.", "speaker_projector.")

    for name, param in model.named_parameters():
        if name.startswith("audio_vae."):
            param.requires_grad = False
        elif stage == "bridge":
            param.requires_grad = name.startswith(bridge_prefixes)
        elif stage == "tslm":
            param.requires_grad = name.startswith(tslm_prefixes)
        else:  # full
            param.requires_grad = True


def load_voxcpm2_teacher(voxcpm2_path: str, device: str | None = None):
    """Load the original VoxCPM2 model as a frozen distillation teacher.

    Its pre-FSQ TSLM hidden states (``enc_outputs`` before the fsq/text mix)
    are the target for aligning ``tslm_adapter`` outputs in stage ``bridge``.
    """
    from bluemagpie._vendor.voxcpm.model.voxcpm2 import VoxCPM2Model

    teacher = VoxCPM2Model.from_local(voxcpm2_path, optimize=False, training=False, device=device)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
