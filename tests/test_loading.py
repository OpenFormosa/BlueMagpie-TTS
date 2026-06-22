"""Weight-surgery tests against the Barbet R2 vocab contract."""

import torch
from tiny_models import tiny_config

from barbet import BarbetModel

from bluemagpie import BlueMagpieModel
from bluemagpie.loading import load_barbet_weights


def test_load_barbet_weights_reinits_padding_rows(tmp_path):
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.barbet_config["vocab_size"] = 72  # padded embedding, effective vocab 64
    cfg.barbet_effective_vocab_size = 64
    model = BlueMagpieModel(cfg, tokenizer=None, audio_vae=None, device="cpu")
    assert model.audio_start_token == 64
    assert model.barbet_config.vocab_size == 72

    # Source checkpoint: padding rows are all-zero, as Megatron leaves them.
    src = BarbetModel(model.barbet_config)
    with torch.no_grad():
        src.embed_tokens.weight[64:] = 0.0
    torch.save(src.state_dict(), tmp_path / "pytorch_model.bin")

    load_barbet_weights(model, str(tmp_path))

    tgt = model.base_lm.backbone.embed_tokens.weight
    # tokenizer-used rows copied verbatim
    torch.testing.assert_close(tgt[:64], src.embed_tokens.weight[:64])
    # special-token rows (audio_start/end, ref_start/end, spk) must not stay zero
    special = tgt[[64, 65, 66, 67, 68]]
    assert bool((special.abs().sum(dim=1) > 0).all()), "special rows must be re-initialized"
    # non-embedding weights load verbatim
    torch.testing.assert_close(
        model.base_lm.backbone.layers[0].mlp.gate_proj.weight,
        src.layers[0].mlp.gate_proj.weight,
    )


def test_load_barbet_weights_drops_tied_lm_head(tmp_path):
    """R2 checkpoints tie lm_head to embeddings; CausalLM dicts must load clean."""
    torch.manual_seed(1)
    cfg = tiny_config()
    model = BlueMagpieModel(cfg, tokenizer=None, audio_vae=None, device="cpu")

    src = BarbetModel(model.barbet_config)
    state = {f"model.{k}": v for k, v in src.state_dict().items()}
    state["lm_head.weight"] = src.embed_tokens.weight.detach().clone()  # tied copy
    torch.save(state, tmp_path / "pytorch_model.bin")

    load_barbet_weights(model, str(tmp_path))
    torch.testing.assert_close(
        model.base_lm.backbone.layers[0].mlp.gate_proj.weight,
        src.layers[0].mlp.gate_proj.weight,
    )
