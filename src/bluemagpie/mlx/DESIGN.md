# BlueMagpie-TTS — MLX port (Apple Silicon)

A native **MLX** (`mlx.core` / `mlx.nn`) re-implementation of the inference path
so BlueMagpie runs on the Apple-Silicon GPU (Metal, unified memory). It is an
**optional extra** — the core package stays torch-only (`pip install -e .[mlx]`).

## Approach

- Each module re-implements its PyTorch forward in MLX. Weights are the *trained
  torch weights*, converted once to `mx.array` and held in a flat
  `{named_parameters() name: mx.array}` dict (`convert.py`), looked up by name —
  no name remapping or per-module loaders. `nn.Linear` and (depthwise) `Conv1d`
  layouts are identical across frameworks, so weights transfer as-is.
- **Parity-first**: every module is checked against its PyTorch reference on the
  tiny configs (`tests/test_mlx_parity.py`), running on the dev's own
  Apple-Silicon GPU. Tolerance is ~`6e-3` (abs, unit-std outputs): MLX (Metal,
  fp32) and torch (CPU, fp32) accumulate in different orders, and the references
  mix fused SDPA with hand-rolled attention. A real bug shows up as `0.1+`/NaN.
- Attention uses the **fused Metal `mx.fast.scaled_dot_product_attention`** where
  the reference uses torch SDPA (MiniCPM) — it handles GQA by broadcasting kv
  heads and matches torch SDPA. Barbet's attention is hand-rolled to preserve its
  qk-logit-clip / attention-sink / fp32-softmax-upcast semantics (which SDPA
  can't express); its plain (no-sink, no-clip) path can move to fast SDPA later.

## Status — the full inference path is ported + parity-verified

All modules pass numerical parity against their PyTorch reference in
`tests/test_mlx_parity.py` (9 tests, on the Apple-Silicon GPU).

| Module | File | Covers |
|---|---|---|
| Barbet (hybrid TSLM) | `barbet_mlx.py` | full forward **and cached decode step**: global + sliding attention, **Mamba2** (depthwise causal conv + selective scan), RoPE, qk-norm, qk-clip, attention sink, SwiGLU |
| MiniCPM | `minicpm_mlx.py` | full forward + **cached decode step**; RALM (no-rope, causal), LocEnc/LocDiT backbone (LongRoPE, non-causal), GQA via fused Metal SDPA, muP |
| LocEnc | `locenc_mlx.py` | patch in-proj + special token + non-causal encoder |
| **LocDiT + CFM** | `dit_mlx.py` | DiT estimator (timestep/delta-time embeddings + in/cond/out proj + decoder) and the **Euler CFG sampler** (`solve_euler`, cfg-zero-star) — the per-patch FLOP dominator |
| FSQ + Adapter | `layers_mlx.py` | ScalarQuantizationLayer, ProjectionAdapter |
| **Full AR loop** | `model_mlx.py` | `BlueMagpieMLX.inference` mirrors `model._inference` (cached Barbet/RALM step + per-patch DiT/LocEnc/stop). Parity vs torch `_inference` at ~1e-3/patch with injected noise; `mlx_generate` adds torch input-assembly + AudioVAE decode |
| weight conversion | `convert.py` | torch params → `mx.array` flat dict |

**Decode is O(1)/step (cached), not O(T) re-run** — prefill loops the step kernel
to warm the cache (provably == full forward, like the serving engine), then each
step advances Barbet/RALM by one position with growing KV + Mamba state. The
cached Barbet step matches torch `forward_step` to ~`2e-7`.

**Speed** (medium random config, 30 patches): MLX-GPU **2.2×** faster than
torch-CPU end-to-end; a MiniCPM forward microbenchmark is **~8.8×** over torch-CPU
and **~2.5×** over torch-MPS. The real (larger) model is more compute-bound, where
the GPU gap widens.

## Notes

- `import bluemagpie.mlx` requires `mlx` (Apple Silicon only); the core package
  never imports it. The parity tests `pytest.importorskip("mlx.core")`, so they
  skip on non-Apple-Silicon CI.
- Parity tolerance is ~`6e-3` abs (unit-std outputs): MLX (Metal, fp32) and torch
  (CPU, fp32) accumulate differently, and references mix fused SDPA with
  hand-rolled attention. A real bug shows up as `0.1+`/NaN. The cached step path
  is much tighter (~`2e-7`) since it matches torch's own stepwise math.
- **Remaining perf options** (not correctness): the selective-scan prefill uses a
  per-position loop (an associative/chunked scan would speed long prefills);
  `mx.compile` over the decode step; AudioVAE could move to MLX (it runs once at
  the end, so the torch↔MLX boundary there is cheap).
