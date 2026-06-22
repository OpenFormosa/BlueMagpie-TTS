"""End-to-end smoke tests for the BlueMagpie hybrid on tiny CPU configs.

Exercises the full graph: Barbet TSLM -> projection adapter -> FSQ -> RALM ->
LocDiT diffusion loss (training), and the cached AR generation loop
(inference) — without any pretrained weights or audio files.
"""

import pytest
import torch
from tiny_models import tiny_config

from bluemagpie import BlueMagpieModel, set_training_stage


@pytest.fixture(scope="module")
def model() -> BlueMagpieModel:
    torch.manual_seed(0)
    return BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu")


def test_special_tokens_appended_after_barbet_vocab(model):
    # no padded region declared -> append-and-grow behaviour (5 special tokens)
    assert model.audio_start_token == 64
    assert model.ref_audio_end_token == 67
    assert model.spk_token == 68
    assert model.barbet_config.vocab_size == 69
    assert model.base_lm.embed_tokens.num_embeddings == 69


def test_special_tokens_use_padding_region_when_declared():
    cfg = tiny_config()
    cfg.barbet_config["vocab_size"] = 72  # simulated Megatron-padded embedding
    cfg.barbet_effective_vocab_size = 64  # ids the tokenizer actually uses
    barbet_cfg, ids = cfg.resolve_barbet_config()
    assert ids == {
        "audio_start": 64,
        "audio_end": 65,
        "ref_audio_start": 66,
        "ref_audio_end": 67,
        "spk": 68,
    }
    assert barbet_cfg.vocab_size == 72, "padding region must absorb the special tokens without growth"


def test_special_tokens_pangolin_contract_autodetect():
    from barbet.configuration_barbet import EFFECTIVE_VOCAB_SIZE, MEGATRON_PADDED_VOCAB_SIZE

    cfg = tiny_config()
    cfg.barbet_config["vocab_size"] = MEGATRON_PADDED_VOCAB_SIZE
    barbet_cfg, ids = cfg.resolve_barbet_config()
    assert ids["audio_start"] == EFFECTIVE_VOCAB_SIZE
    assert ids["ref_audio_end"] == EFFECTIVE_VOCAB_SIZE + 3
    assert ids["spk"] == EFFECTIVE_VOCAB_SIZE + 4
    assert barbet_cfg.vocab_size == MEGATRON_PADDED_VOCAB_SIZE


def test_training_forward_losses(model):
    torch.manual_seed(1)
    B, T = 2, 10
    P, D = model.patch_size, model.feat_dim
    n_text = 4

    text_tokens = torch.randint(0, 64, (B, T))
    text_mask = torch.zeros(B, T)
    text_mask[:, :n_text] = 1
    audio_mask = 1 - text_mask
    audio_feats = torch.randn(B, T, P, D)
    loss_mask = audio_mask.clone()
    labels = torch.zeros(B, T, dtype=torch.long)
    labels[:, -1] = 1

    out = model(
        text_tokens=text_tokens,
        text_mask=text_mask,
        audio_feats=audio_feats,
        audio_mask=audio_mask,
        loss_mask=loss_mask,
        position_ids=torch.arange(T).expand(B, -1),
        labels=labels,
    )

    assert torch.isfinite(out["loss/diff"]), "diffusion loss must be finite"
    assert torch.isfinite(out["loss/stop"]), "stop loss must be finite"
    assert out["tslm_hidden"].shape == (B, T, model.config.vox_lm_config.hidden_size)
    assert out["feat_gt"].shape == (B, D, T * P)

    total = out["loss/diff"] + out["loss/stop"]
    total.backward()
    grad = model.tslm_adapter.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    model.zero_grad(set_to_none=True)


def test_inference_loop_generates_latents(model):
    torch.manual_seed(2)
    model.eval()
    T_text = 6
    P, D = model.patch_size, model.feat_dim

    text = torch.randint(0, 64, (1, T_text))
    text[0, -1] = model.audio_start_token
    text_mask = torch.ones(1, T_text, dtype=torch.int32)
    audio_mask = torch.zeros(1, T_text, dtype=torch.int32)
    feat = torch.zeros(1, T_text, P, D)

    feat_pred, generated_feat = model.inference(
        text,
        text_mask,
        feat,
        audio_mask,
        min_len=2,
        max_len=5,
        inference_timesteps=2,
        cfg_value=1.5,
    )

    assert feat_pred.dim() == 3 and feat_pred.shape[0] == 1 and feat_pred.shape[1] == D
    assert feat_pred.shape[2] % P == 0 and 1 <= feat_pred.shape[2] // P <= 5
    assert generated_feat.shape[-2:] == (P, D)
    assert torch.isfinite(feat_pred).all()


def test_stage_freezing(model):
    set_training_stage(model, "bridge")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(("enc_to_tslm_proj.", "tslm_adapter.")) for n in trainable)
    assert any(n.startswith("tslm_adapter.") for n in trainable)

    set_training_stage(model, "tslm")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("base_lm.") for n in trainable)
    assert not any(n.startswith(("residual_lm.", "feat_decoder.", "feat_encoder.")) for n in trainable)

    set_training_stage(model, "full")
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert all(n.startswith("audio_vae.") for n in frozen)
