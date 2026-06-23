"""Parity tests for the serving engine's batched kernels (CPU, tiny models).

The gating test: a single batched ``decode_step`` over rows at DISTINCT absolute
positions must equal looping the model's own single-sequence ``forward_step``,
one row at a time, each primed to its own position. This is the property the
whole continuous-batching engine rests on.
"""

import torch
from barbet import BarbetConfig

from bluemagpie import BlueMagpieModel
from bluemagpie._vendor.voxcpm.model.utils import next_and_close
from bluemagpie._vendor.voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from bluemagpie._vendor.voxcpm.modules.minicpm4.config import RopeScalingConfig
from bluemagpie.serving.barbet_batch import BatchedBarbet
from bluemagpie.serving.minicpm_batch import BatchedRALM
from bluemagpie.serving.runner import BlueMagpieRunner
from bluemagpie.tslm import BarbetTSLM
from tiny_models import tiny_config


def _tiny_barbet(**overrides) -> BarbetConfig:
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
        sliding_window_size=5,          # small enough that L>5 rows actually slide
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


def _reference_step(tslm: BarbetTSLM, prompt: torch.Tensor, step_x: torch.Tensor) -> torch.Tensor:
    """prefill(prompt) then one forward_step(step_x) -> hidden [1, H]."""
    _, state = tslm.prefill(prompt)            # prompt: [1, L, H]
    return tslm.forward_step(step_x, state)    # step_x: [1, H] -> [1, H]


@torch.no_grad()
def _run_ragged(config: BarbetConfig, lengths, slots):
    torch.manual_seed(0)
    tslm = BarbetTSLM(config).eval().float()
    H = config.hidden_size
    R = len(lengths)

    prompts = [torch.randn(1, L, H) for L in lengths]
    step_x = [torch.randn(1, H) for _ in range(R)]

    # Reference: independent single-sequence prefill + forward_step per row.
    ref = [_reference_step(tslm, prompts[i], step_x[i])[0] for i in range(R)]

    # Mine: prime each row's slot, then ONE batched decode step at ragged positions.
    bb = BatchedBarbet(tslm, max_seqs=max(slots) + 1, max_len=max(lengths) + 2, device=torch.device("cpu"), dtype=torch.float32)
    for i in range(R):
        bb.reset_slot(slots[i])
        bb.prefill(prompts[i][0], slot=slots[i])

    x_batch = torch.cat(step_x, dim=0)                                  # [R, H]
    slots_t = torch.tensor(slots, dtype=torch.long)
    positions_t = torch.tensor(lengths, dtype=torch.long)              # each row at its own length
    mine = bb.decode_step(x_batch, slots_t, positions_t)               # [R, H]

    for i in range(R):
        torch.testing.assert_close(mine[i], ref[i], rtol=1e-4, atol=1e-5, msg=f"row {i} (len={lengths[i]}, slot={slots[i]})")


@torch.no_grad()
def test_ragged_batched_step_matches_forward_step():
    # Distinct lengths (some > sliding window=5) at non-contiguous slots.
    _run_ragged(_tiny_barbet(), lengths=[3, 7, 5, 4], slots=[2, 0, 3, 1])


@torch.no_grad()
def test_ragged_without_sink_and_clip():
    _run_ragged(_tiny_barbet(attention_sink=False, qk_logit_clip=False), lengths=[2, 8, 6], slots=[0, 2, 1])


@torch.no_grad()
def test_prompt_shorter_than_conv_kernel_batched():
    # Rows whose prompt is shorter than mamba_d_conv-1, mixed with longer rows.
    _run_ragged(_tiny_barbet(), lengths=[2, 9, 3], slots=[1, 0, 2])


# --------------------------------------------------------------------------- #
# RALM (MiniCPM, no-RoPE) batched step parity
# --------------------------------------------------------------------------- #
KV_CHANNELS = 8


def _tiny_ralm_config(num_layers: int = 4) -> MiniCPM4Config:
    return MiniCPM4Config(
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=256,
        num_attention_heads=4,
        num_hidden_layers=num_layers,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        rope_scaling=RopeScalingConfig(
            type="longrope",
            long_factor=[1.0] * (KV_CHANNELS // 2),
            short_factor=[1.0] * (KV_CHANNELS // 2),
            original_max_position_embeddings=256,
        ),
        vocab_size=0,            # identity embedding (RALM consumes embeddings)
        no_rope=True,            # residual_lm_no_rope
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=32,
        scale_depth=1.0,
        rope_theta=10000.0,
        kv_channels=KV_CHANNELS,
    )


@torch.no_grad()
def test_ralm_ragged_batched_step_matches_forward_step():
    torch.manual_seed(0)
    cfg = _tiny_ralm_config()
    ralm = MiniCPMModel(cfg).eval().float()
    H = cfg.hidden_size
    lengths = [3, 7, 5, 4]
    slots = [2, 0, 3, 1]
    maxlen = max(lengths) + 2
    R = len(lengths)

    prompts = [torch.randn(1, L, H) for L in lengths]
    step_x = [torch.randn(1, H) for _ in range(R)]

    # Reference: per-row fresh StaticKVCache, full-forward prefill + forward_step.
    ref = []
    for i in range(R):
        ralm.setup_cache(1, maxlen, torch.device("cpu"), torch.float32)
        _, kv = ralm(inputs_embeds=prompts[i], is_causal=True)
        ralm.kv_cache.fill_caches(kv)
        pos = torch.tensor([ralm.kv_cache.step()])
        ref.append(ralm.forward_step(step_x[i], pos)[0])

    # Mine: prime each row, then ONE batched decode step at ragged positions.
    br = BatchedRALM(ralm, max_seqs=max(slots) + 1, max_len=maxlen, device=torch.device("cpu"), dtype=torch.float32)
    for i in range(R):
        br.prefill(prompts[i][0], slot=slots[i])
    mine = br.decode_step(
        torch.cat(step_x, dim=0),
        torch.tensor(slots, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )
    for i in range(R):
        torch.testing.assert_close(mine[i], ref[i], rtol=1e-4, atol=1e-5, msg=f"RALM row {i}")


# --------------------------------------------------------------------------- #
# End-to-end batch=1 parity: runner latents == BlueMagpieModel._inference
# --------------------------------------------------------------------------- #
def _zero_shot_inputs(model, n_text: int = 5):
    L = n_text + 1  # + audio_start
    text_token = torch.randint(0, 50, (L,), dtype=torch.long)
    text_token[-1] = model.audio_start_token
    audio_feat = torch.zeros(L, model.patch_size, model.config.feat_dim, dtype=torch.float32)
    text_mask = torch.ones(L, dtype=torch.float32)
    audio_mask = torch.zeros(L, dtype=torch.float32)
    return text_token, audio_feat, text_mask, audio_mask


@torch.no_grad()
def test_runner_batch1_matches_inference():
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    text_token, audio_feat, text_mask, audio_mask = _zero_shot_inputs(model)

    kw = dict(min_len=2, max_len=6, inference_timesteps=5, cfg_value=2.0)

    torch.manual_seed(123)
    ref_feat_pred, ref_generated, ref_ctx = next_and_close(
        model._inference(
            text_token[None], text_mask[None], audio_feat[None], audio_mask[None], streaming=False, **kw
        )
    )

    torch.manual_seed(123)
    runner = BlueMagpieRunner(model, max_seqs=4, max_len=64)
    st = runner.prefill(slot=0, text_token=text_token, audio_feat=audio_feat,
                        text_mask=text_mask, audio_mask=audio_mask)
    st.min_len, st.max_len, st.n_timesteps, st.cfg_value = 2, 6, 5, 2.0
    while not st.finished:
        runner.decode_step([st])
    mine = runner.collect_latents(st)  # [T, p, d]

    assert mine.shape[0] == ref_generated.shape[0], f"patch count {mine.shape[0]} != {ref_generated.shape[0]}"
    torch.testing.assert_close(mine, ref_generated, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_runner_batch2_identical_requests_match_batch1():
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    text_token, audio_feat, text_mask, audio_mask = _zero_shot_inputs(model)

    # Two identical requests, each with its OWN generator seeded the same -> the
    # batched DiT z-draw is per-row, so both rows match each other exactly.
    runner = BlueMagpieRunner(model, max_seqs=4, max_len=64)
    states = []
    for slot in (0, 1):
        st = runner.prefill(slot=slot, text_token=text_token, audio_feat=audio_feat,
                            text_mask=text_mask, audio_mask=audio_mask)
        st.min_len, st.max_len, st.n_timesteps, st.cfg_value = 2, 6, 5, 2.0
        g = torch.Generator(device="cpu"); g.manual_seed(999)
        st.generator = g
        states.append(st)

    while not all(s.finished for s in states):
        active = [s for s in states if not s.finished]
        runner.decode_step(active)

    a = runner.collect_latents(states[0])
    b = runner.collect_latents(states[1])
    assert a.shape == b.shape
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Engine: continuous batching produces the same per-request latents as singletons
# --------------------------------------------------------------------------- #
def _zero_shot_inputs_n(model, n_text, gen):
    L = n_text + 1
    text_token = torch.randint(0, 50, (L,), generator=gen, dtype=torch.long)
    text_token[-1] = model.audio_start_token
    audio_feat = torch.zeros(L, model.patch_size, model.config.feat_dim, dtype=torch.float32)
    text_mask = torch.ones(L, dtype=torch.float32)
    audio_mask = torch.zeros(L, dtype=torch.float32)
    return text_token, audio_feat, text_mask, audio_mask


@torch.no_grad()
def test_engine_continuous_batching_matches_singletons():
    from bluemagpie.serving import BlueMagpieEngine, EngineConfig

    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()

    g = torch.Generator().manual_seed(3)
    specs = [
        dict(inp=_zero_shot_inputs_n(model, 4, g), seed=11, max_len=4),
        dict(inp=_zero_shot_inputs_n(model, 6, g), seed=22, max_len=6),
        dict(inp=_zero_shot_inputs_n(model, 5, g), seed=33, max_len=5),
    ]
    common = dict(min_len=2, inference_timesteps=5, cfg_value=2.0)

    def run_single(spec):
        eng = BlueMagpieEngine(model, EngineConfig(max_num_seqs=1, max_model_len=64))
        eng.submit_prefill_inputs(*spec["inp"], max_len=spec["max_len"], seed=spec["seed"], **common)
        return eng.run()[0].latents

    singles = [run_single(s) for s in specs]

    # max_num_seqs=2 forces the 3rd request to wait then join mid-decode, and
    # forces slot reuse as the short request finishes first.
    eng = BlueMagpieEngine(model, EngineConfig(max_num_seqs=2, max_model_len=64))
    for s in specs:
        eng.submit_prefill_inputs(*s["inp"], max_len=s["max_len"], seed=s["seed"], **common)
    outs = eng.run()

    assert [o.request_id for o in outs] == [0, 1, 2]
    for i, o in enumerate(outs):
        assert o.latents.shape == singles[i].shape, f"req {i}: {o.latents.shape} vs {singles[i].shape}"
        torch.testing.assert_close(o.latents, singles[i], rtol=1e-5, atol=1e-6, msg=f"req {i}")


@torch.no_grad()
def test_runner_continuation_mode_matches_inference():
    """Prompt-audio continuation: feat_mask ends with audio, so a context prefix
    is seeded. The generated tail must match _inference's generated_feat."""
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    p, d = model.patch_size, model.config.feat_dim
    L_text, L_audio = 4, 3

    text_part = torch.randint(0, 50, (L_text,), dtype=torch.long)
    text_part[-1] = model.audio_start_token
    text_token = torch.cat([text_part, torch.zeros(L_audio, dtype=torch.long)])
    audio_feat = torch.cat([torch.zeros(L_text, p, d), torch.randn(L_audio, p, d)], dim=0)
    text_mask = torch.cat([torch.ones(L_text), torch.zeros(L_audio)])
    audio_mask = torch.cat([torch.zeros(L_text), torch.ones(L_audio)])

    kw = dict(min_len=2, max_len=6, inference_timesteps=5, cfg_value=2.0)

    torch.manual_seed(123)
    _, ref_generated, ref_ctx = next_and_close(
        model._inference(text_token[None], text_mask[None], audio_feat[None], audio_mask[None], streaming=False, **kw)
    )

    torch.manual_seed(123)
    runner = BlueMagpieRunner(model, max_seqs=2, max_len=64)
    st = runner.prefill(slot=0, text_token=text_token, audio_feat=audio_feat,
                        text_mask=text_mask, audio_mask=audio_mask)
    st.min_len, st.max_len, st.n_timesteps, st.cfg_value = 2, 6, 5, 2.0
    while not st.finished:
        runner.decode_step([st])
    full = runner.collect_latents(st)          # [context + T, p, d]
    mine_generated = full[st.context_len :]

    assert st.context_len == ref_ctx == min(3, L_audio)
    assert mine_generated.shape[0] == ref_generated.shape[0]
    torch.testing.assert_close(mine_generated, ref_generated, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_runner_speaker_centroid_matches_inference():
    """Speaker-centroid mode exercises _inject_speaker at the [spk] slot."""
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    p, d = model.patch_size, model.config.feat_dim
    L_text = 4

    text_part = torch.randint(0, 50, (L_text,), dtype=torch.long)
    text_part[-1] = model.audio_start_token
    text_token = torch.cat([torch.tensor([model.spk_token]), text_part])
    L = text_token.shape[0]
    audio_feat = torch.zeros(L, p, d)
    text_mask = torch.cat([torch.tensor([0.0]), torch.ones(L_text)])       # spk slot t_mask=0 (centroid)
    audio_mask = torch.zeros(L)
    spk_mask = torch.cat([torch.tensor([1.0]), torch.zeros(L_text)])
    centroid = torch.randn(model.speaker_embed_dim)
    centroids = centroid.reshape(1, -1)

    kw = dict(min_len=2, max_len=6, inference_timesteps=5, cfg_value=2.0)

    torch.manual_seed(123)
    _, ref_generated, _ = next_and_close(
        model._inference(text_token[None], text_mask[None], audio_feat[None], audio_mask[None],
                         speaker_centroids=centroids, spk_mask=spk_mask[None], streaming=False, **kw)
    )

    torch.manual_seed(123)
    runner = BlueMagpieRunner(model, max_seqs=2, max_len=64)
    st = runner.prefill(slot=0, text_token=text_token, audio_feat=audio_feat, text_mask=text_mask,
                        audio_mask=audio_mask, spk_mask=spk_mask, speaker_centroids=centroids)
    st.min_len, st.max_len, st.n_timesteps, st.cfg_value = 2, 6, 5, 2.0
    while not st.finished:
        runner.decode_step([st])
    mine = runner.collect_latents(st)
    assert mine.shape[0] == ref_generated.shape[0]
    torch.testing.assert_close(mine, ref_generated, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_prefill_batch_matches_single():
    """Pad-batched cohort prefill == looping single-sequence prefill."""
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    g = torch.Generator().manual_seed(5)
    inputs = [_zero_shot_inputs_n(model, n, g) for n in (4, 7, 5)]

    singles = []
    for inp in inputs:
        r = BlueMagpieRunner(model, max_seqs=1, max_len=64)
        singles.append(r.prefill(0, *inp))

    runner = BlueMagpieRunner(model, max_seqs=4, max_len=64)
    preps = [runner._prepare(slot, *inp, None, None, 4) for slot, inp in zip((2, 0, 1), inputs)]
    states = runner.prefill_batch(preps)
    for st, single in zip(states, singles):
        torch.testing.assert_close(st.lm_hidden, single.lm_hidden, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(st.residual_hidden, single.residual_hidden, rtol=1e-4, atol=1e-5)
        assert st.pos == single.pos


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_engine_stream_latents_match_run():
    """Per-request streamed latent chunks concatenate to the run() latents."""
    from collections import defaultdict
    from bluemagpie.serving import BlueMagpieEngine, EngineConfig

    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    g = torch.Generator().manual_seed(3)
    specs = [
        dict(inp=_zero_shot_inputs_n(model, 4, g), seed=11, max_len=4),
        dict(inp=_zero_shot_inputs_n(model, 6, g), seed=22, max_len=6),
        dict(inp=_zero_shot_inputs_n(model, 5, g), seed=33, max_len=5),
    ]
    common = dict(min_len=2, inference_timesteps=5, cfg_value=2.0)

    eng_run = BlueMagpieEngine(model, EngineConfig(max_num_seqs=2, max_model_len=64))
    for s in specs:
        eng_run.submit_prefill_inputs(*s["inp"], max_len=s["max_len"], seed=s["seed"], **common)
    run_out = {o.request_id: o.latents for o in eng_run.run()}

    eng_str = BlueMagpieEngine(model, EngineConfig(max_num_seqs=2, max_model_len=64))
    for s in specs:
        eng_str.submit_prefill_inputs(*s["inp"], max_len=s["max_len"], seed=s["seed"], **common)
    chunks = defaultdict(list)
    finished = set()
    for ch in eng_str.stream():
        assert ch.audio is None  # no VAE attached
        chunks[ch.request_id].append(ch.latents)
        if ch.finished:
            finished.add(ch.request_id)

    assert finished == {0, 1, 2}
    for rid, chs in chunks.items():
        streamed = torch.stack(chs, dim=0)            # [T, p, d]
        torch.testing.assert_close(streamed, run_out[rid], rtol=1e-5, atol=1e-6, msg=f"req {rid}")


def test_batched_streaming_vae_reconcile():
    from bluemagpie.serving.streaming import BatchedStreamingVAE

    class _FakeDec:
        def __init__(self): self._states = {}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def decode_chunk(self, z): return z

    class _FakeVAE:
        def streaming_decode(self): return _FakeDec()

    bvae = BatchedStreamingVAE(_FakeVAE())
    bvae._reconcile([10, 11, 12])                      # first step's order
    bvae.dec._states["m"] = torch.tensor([[[10.0]], [[11.0]], [[12.0]]])  # [3,1,1], tagged by key

    bvae._reconcile([10, 12, 13])                      # 11 leaves, 13 joins
    st = bvae.dec._states["m"]
    assert st.shape[0] == 3
    assert st[0].item() == 10.0                        # kept
    assert st[1].item() == 12.0                        # kept + reordered
    assert st[2].item() == 0.0                         # new row -> cold-start zeros


# --------------------------------------------------------------------------- #
# Acceleration gating (torch.compile / CUDA graphs are CUDA-only no-ops here)
# --------------------------------------------------------------------------- #
def test_compile_is_noop_off_cuda():
    import torch as _t
    from bluemagpie.serving.accel import optimize_for_inference
    from bluemagpie.serving import BlueMagpieEngine, EngineConfig

    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    enc_before, est_before = model.feat_encoder, model.feat_decoder.estimator

    assert optimize_for_inference(model) is False        # CPU -> no-op
    assert model.feat_encoder is enc_before
    assert model.feat_decoder.estimator is est_before

    # Engine honoring compile=True on CPU must still construct and leave modules eager.
    eng = BlueMagpieEngine(model, EngineConfig(max_num_seqs=2, compile=True, enforce_eager=False))
    assert model.feat_encoder is enc_before
    assert model.feat_decoder.estimator is est_before
    assert eng is not None
