"""forward_step must reproduce the full-sequence Barbet forward exactly.

Covers all three layer types (global attention, sliding-window attention,
mamba mixer) with qk-norm, qk-logit clipping and attention sink enabled, and
a sliding window small enough to actually trim the cache mid-sequence.
"""

import torch
from barbet import BarbetConfig

from bluemagpie.tslm import BarbetTSLM


def tiny_barbet_config(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        sliding_window_size=5,
        global_attention_layers=(0, 3),
        mamba_layers=(2, 5),
        qk_norm=True,
        qk_logit_clip=True,
        qk_clip_threshold=30.0,
        attention_sink=True,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mtp_enabled=False,
    )
    kwargs.update(overrides)
    return BarbetConfig(**kwargs)


@torch.no_grad()
def test_prefill_plus_steps_match_full_forward():
    torch.manual_seed(0)
    config = tiny_barbet_config()
    model = BarbetTSLM(config).eval().float()

    B, T_prompt, T_total = 2, 7, 19  # decode well past the sliding window
    embeds = torch.randn(B, T_total, config.hidden_size)

    full_hidden = model(inputs_embeds=embeds)

    prefill_hidden, state = model.prefill(embeds[:, :T_prompt, :])
    torch.testing.assert_close(prefill_hidden, full_hidden[:, :T_prompt, :], rtol=1e-4, atol=1e-5)

    for t in range(T_prompt, T_total):
        step_hidden = model.forward_step(embeds[:, t, :], state)
        torch.testing.assert_close(
            step_hidden,
            full_hidden[:, t, :],
            rtol=1e-4,
            atol=1e-5,
            msg=f"mismatch at position {t}",
        )
    assert state.pos == T_total


@torch.no_grad()
def test_step_equivalence_without_sink_and_clip():
    torch.manual_seed(1)
    config = tiny_barbet_config(attention_sink=False, qk_logit_clip=False)
    model = BarbetTSLM(config).eval().float()

    B, T_prompt, T_total = 1, 3, 12
    embeds = torch.randn(B, T_total, config.hidden_size)
    full_hidden = model(inputs_embeds=embeds)

    _, state = model.prefill(embeds[:, :T_prompt, :])
    for t in range(T_prompt, T_total):
        step_hidden = model.forward_step(embeds[:, t, :], state)
        torch.testing.assert_close(step_hidden, full_hidden[:, t, :], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_prompt_shorter_than_conv_kernel():
    torch.manual_seed(2)
    config = tiny_barbet_config()
    model = BarbetTSLM(config).eval().float()

    B, T_prompt, T_total = 1, 2, 8  # prompt shorter than mamba_d_conv - 1 padding window
    embeds = torch.randn(B, T_total, config.hidden_size)
    full_hidden = model(inputs_embeds=embeds)

    _, state = model.prefill(embeds[:, :T_prompt, :])
    for t in range(T_prompt, T_total):
        step_hidden = model.forward_step(embeds[:, t, :], state)
        torch.testing.assert_close(step_hidden, full_hidden[:, t, :], rtol=1e-4, atol=1e-5)
