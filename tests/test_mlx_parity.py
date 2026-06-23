"""Numerical-parity tests for the MLX port (Apple Silicon).

Each MLX module must reproduce its PyTorch reference on the tiny configs. These
run on the MLX GPU when available and skip cleanly when `mlx` is not installed
(so non-Apple-Silicon CI is unaffected). Tolerances are loose-ish because MLX
(Metal, fp32) and torch (CPU, fp32) accumulate in different orders.
"""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from barbet import BarbetConfig  # noqa: E402

from bluemagpie.mlx.barbet_mlx import BarbetMLX  # noqa: E402
from bluemagpie.mlx.convert import to_mx  # noqa: E402
from bluemagpie.tslm import BarbetTSLM  # noqa: E402


def _max_abs_diff(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.abs(ref - got).max())


def _tiny_barbet(**ov) -> BarbetConfig:
    kw = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
        rope_theta=10000.0, sliding_window_size=5, global_attention_layers=(0, 3),
        mamba_layers=(2, 5), qk_norm=True, qk_logit_clip=True, qk_clip_threshold=30.0,
        attention_sink=True, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mtp_enabled=False,
    )
    kw.update(ov)
    return BarbetConfig(**kw)


@torch.no_grad()
def _check_barbet(cfg: BarbetConfig, atol: float = 2e-3):
    torch.manual_seed(0)
    tslm = BarbetTSLM(cfg).eval().float()
    embeds = torch.randn(2, 9, cfg.hidden_size)
    ref = tslm(inputs_embeds=embeds).numpy()
    out = BarbetMLX(tslm.backbone)(to_mx(embeds))
    mx.eval(out)
    assert _max_abs_diff(ref, np.array(out)) < atol


def test_barbet_mlx_full_forward():
    # All three layer types, with sink + qk-clip enabled.
    _check_barbet(_tiny_barbet())


def test_barbet_mlx_without_sink_and_clip():
    _check_barbet(_tiny_barbet(attention_sink=False, qk_logit_clip=False))


# --------------------------------------------------------------------------- #
# MiniCPM (RALM / LocEnc encoder / LocDiT decoder)
# --------------------------------------------------------------------------- #
from bluemagpie._vendor.voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel  # noqa: E402
from bluemagpie._vendor.voxcpm.modules.minicpm4.config import RopeScalingConfig  # noqa: E402
from bluemagpie.mlx.minicpm_mlx import MiniCPMMLX  # noqa: E402

_KV = 8


def _tiny_minicpm(no_rope: bool, num_layers: int = 3) -> MiniCPM4Config:
    return MiniCPM4Config(
        bos_token_id=1, eos_token_id=2, hidden_size=32, intermediate_size=64,
        max_position_embeddings=256, num_attention_heads=4, num_hidden_layers=num_layers,
        num_key_value_heads=2, rms_norm_eps=1e-5,
        rope_scaling=RopeScalingConfig(type="longrope", long_factor=[1.0] * (_KV // 2),
                                       short_factor=[1.0] * (_KV // 2), original_max_position_embeddings=256),
        vocab_size=0, no_rope=no_rope, use_mup=False, scale_emb=1.0, dim_model_base=32,
        scale_depth=1.0, rope_theta=10000.0, kv_channels=_KV,
    )


@torch.no_grad()
def _check_minicpm(no_rope: bool, is_causal: bool, atol: float = 6e-3):
    torch.manual_seed(0)
    cfg = _tiny_minicpm(no_rope)
    model = MiniCPMModel(cfg).eval().float()
    x = torch.randn(2, 7, cfg.hidden_size)
    ref, _ = model(inputs_embeds=x, is_causal=is_causal)
    out = MiniCPMMLX(model)(to_mx(x), is_causal=is_causal)
    mx.eval(out)
    assert _max_abs_diff(ref.numpy(), np.array(out)) < atol


def test_minicpm_mlx_ralm_no_rope_causal():
    _check_minicpm(no_rope=True, is_causal=True)


def test_minicpm_mlx_rope_noncausal():
    # LocEnc encoder / LocDiT decoder shape: LongRoPE + non-causal attention.
    _check_minicpm(no_rope=False, is_causal=False)


# --------------------------------------------------------------------------- #
# LocEnc (VoxCPMLocEnc)
# --------------------------------------------------------------------------- #
from bluemagpie._vendor.voxcpm.modules.locenc import VoxCPMLocEnc  # noqa: E402
from bluemagpie.mlx.locenc_mlx import LocEncMLX  # noqa: E402


@torch.no_grad()
def test_locenc_mlx_parity():
    torch.manual_seed(0)
    feat_dim = 8
    cfg = _tiny_minicpm(no_rope=False, num_layers=1)   # non-causal encoder w/ LongRoPE
    enc = VoxCPMLocEnc(cfg, input_dim=feat_dim).eval().float()
    x = torch.randn(2, 4, 3, feat_dim)                  # [B, T, P, D]
    ref = enc(x).numpy()
    out = LocEncMLX(enc)(to_mx(x))
    mx.eval(out)
    assert _max_abs_diff(ref, np.array(out)) < 6e-3


# --------------------------------------------------------------------------- #
# Cached decode-step kernels (Barbet / RALM) — the fast path
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_barbet_mlx_cached_step():
    torch.manual_seed(0)
    cfg = _tiny_barbet()
    tslm = BarbetTSLM(cfg).eval().float()
    e = torch.randn(1, 11, cfg.hidden_size)
    m = BarbetMLX(tslm.backbone)
    cache = m.init_cache()
    step_pref = np.array(m.prefill(to_mx(e), cache))          # looped step
    # step-prefill must match the model's full forward.
    assert _max_abs_diff(tslm(inputs_embeds=e).numpy(), step_pref) < 2e-3
    # one more decode step == forward_step.
    e2 = torch.randn(1, cfg.hidden_size)
    _, st = tslm.prefill(e)
    ref = tslm.forward_step(e2, st)[0].numpy()
    got = np.array(m.step(to_mx(e2), 11, cache)[0])
    assert _max_abs_diff(ref, got) < 2e-3


@torch.no_grad()
def test_ralm_mlx_cached_step():
    torch.manual_seed(0)
    cfg = _tiny_minicpm(no_rope=True)
    model = MiniCPMModel(cfg).eval().float()
    e = torch.randn(1, 9, cfg.hidden_size)
    m = MiniCPMMLX(model)
    cache = m.init_cache()
    got = np.array(m.prefill(to_mx(e), cache))
    ref, _ = model(inputs_embeds=e, is_causal=True)
    assert _max_abs_diff(ref.numpy(), got) < 6e-3


# --------------------------------------------------------------------------- #
# LocDiT estimator + CFM solver
# --------------------------------------------------------------------------- #
from bluemagpie._vendor.voxcpm.modules.locdit import VoxCPMLocDiTV2, UnifiedCFM, CfmConfig  # noqa: E402
from bluemagpie.mlx.dit_mlx import LocDiTMLX, solve_euler  # noqa: E402


@torch.no_grad()
def test_locdit_and_cfm_mlx():
    torch.manual_seed(0)
    C, T, H, N = 8, 2, 16, 3
    cfg = _tiny_minicpm(no_rope=False, num_layers=1)
    cfg.hidden_size = 16  # ensure DiT hidden matches
    dit = VoxCPMLocDiTV2(cfg, in_channels=C).eval().float()
    m = LocDiTMLX(dit)
    x = torch.randn(N, C, T); mu = torch.randn(N, 2 * H); t = torch.rand(N); cond = torch.randn(N, C, T); dt = torch.zeros(N)
    ref = dit(x, mu, t, cond, dt).numpy()
    got = np.array(m(to_mx(x), to_mx(mu), to_mx(t), to_mx(cond), to_mx(dt)))
    assert _max_abs_diff(ref, got) < 3e-3

    cfm = UnifiedCFM(in_channels=C, cfm_params=CfmConfig(inference_cfg_rate=2.0), estimator=dit, mean_mode=False).eval()
    z = torch.randn(N, C, T)
    tspan = torch.linspace(1, 0, 6); tspan = tspan + 1.0 * (torch.cos(torch.pi / 2 * tspan) - 1 + tspan)
    ref2 = cfm.solve_euler(x=z, t_span=tspan, mu=mu, cond=cond, cfg_value=2.0, use_cfg_zero_star=True).numpy()
    got2 = np.array(solve_euler(m, to_mx(z), to_mx(tspan), to_mx(mu), to_mx(cond), 2.0))
    assert _max_abs_diff(ref2, got2) < 6e-3


# --------------------------------------------------------------------------- #
# Full AR loop == model._inference (with injected per-patch noise)
# --------------------------------------------------------------------------- #
from bluemagpie import BlueMagpieModel  # noqa: E402
from bluemagpie._vendor.voxcpm.model.utils import next_and_close  # noqa: E402
from bluemagpie.mlx.model_mlx import BlueMagpieMLX  # noqa: E402
from tiny_models import tiny_config  # noqa: E402


@torch.no_grad()
def test_full_ar_loop_matches_inference():
    torch.manual_seed(7)
    model = BlueMagpieModel(tiny_config(), tokenizer=None, audio_vae=None, device="cpu").eval()
    p, d = model.patch_size, model.config.feat_dim
    L = 6
    tt = torch.randint(0, 50, (L,), dtype=torch.long); tt[-1] = model.audio_start_token
    af = torch.zeros(L, p, d); txm = torch.ones(L); aum = torch.zeros(L)

    torch.manual_seed(123)
    noises = [torch.randn(1, d, p) for _ in range(8)]
    kw = dict(min_len=2, max_len=6, inference_timesteps=5, cfg_value=2.0)

    it = iter(noises)
    orig = torch.randn
    torch.randn = lambda *a, **k: next(it)
    try:
        _, ref_gen, _ = next_and_close(
            model._inference(tt[None], txm[None], af[None], aum[None], streaming=False, **kw)
        )
    finally:
        torch.randn = orig
    ref_gen = ref_gen.numpy()

    mm = BlueMagpieMLX(model)
    out = mm.inference(mx.array(tt.numpy())[None], to_mx(af)[None], to_mx(txm)[None], to_mx(aum)[None],
                       noises=[to_mx(z) for z in noises], **kw)
    mx.eval(out)
    out = np.array(out)
    assert out.shape == ref_gen.shape, f"{out.shape} vs {ref_gen.shape}"
    assert _max_abs_diff(ref_gen, out) < 4e-3
