# BlueMagpie-TTS serving engine (vendored, device-agnostic, dependency-light)

A self-contained **nano-vLLM-style** continuous-batching inference engine for
BlueMagpie-TTS, living entirely inside this repo. It batches many TTS requests
together and streams 48 kHz waveform chunks per request.

## Goals / non-goals

**Goals**
- Continuous batching of multiple concurrent TTS requests (throughput).
- Per-request streaming output (yield waveform chunks as they decode).
- Runs on **CUDA, Apple-Silicon/MPS, and CPU** — same code path, graceful
  degradation. Optional CUDA-only accelerations (CUDA graphs, `mamba_ssm`
  selective-scan kernel, `torch.compile`) are auto-detected, never required.
- **Zero new hard dependencies** beyond what the package already uses (`torch`,
  `barbet`, `einops`). No `vllm` / `nano-vllm` / `flash-attn` / `triton` /
  `causal-conv1d`.
- Numerical parity with `bluemagpie.model.BlueMagpieModel._inference` at
  batch=1 (kept as the golden reference, never modified).

**Non-goals (deliberately)**
- True paged-block KV with prefix-cache dedup (flash-attn/triton-only to do
  efficiently). We use a **batched padded KV cache** instead — portable and
  dependency-free, at the cost of some wasted memory/compute on length skew.
- Tensor parallelism / multi-GPU sharding (single-device for v1).

## Why this shape (grounded in the reference accelerators)

`a710128/nanovllm-voxcpm` and `vllm-project/vllm-omni` both: (a) batch only the
pure-attention LMs, (b) run the LocDiT/CFM diffusion decoder **eagerly inside
the forward** (the 2× classifier-free-guidance batch is an internal tensor
batch, not continuous batching), and (c) run the AudioVAE decode eagerly
outside any graph. Neither has **any** Mamba/SSM support. BlueMagpie's TSLM is
the hybrid Mamba2 + sliding/global-attention **Barbet**, so we cannot reuse
their paged-attention path for it; we re-create the engine skeleton ourselves
with a hybrid, device-agnostic cache.

## The per-patch compute (what one decode step must do)

Mirrors `model.py::_inference`. Per generated latent patch, for each running
sequence:
1. `lm_to_dit_proj(lm_hidden)` ⊕ `res_to_dit_proj(residual_hidden)` → `dit_hidden`
2. **LocDiT / UnifiedCFM** `feat_decoder(mu=dit_hidden, cond=prefix_feat_cond,
   n_timesteps, cfg_value)` → `pred_feat` — *FLOP dominator*: `inference_timesteps`
   (default 9) Euler steps × 2 (CFG). Runs eagerly, batched over the active set.
3. **LocEnc** `feat_encoder(pred_feat)` → re-encode into both LM spaces.
4. stop head on `lm_hidden` → per-seq stop flag.
5. **Barbet TSLM** `forward_step(curr_embed_tslm)` → next `lm_hidden` (via
   `tslm_adapter` + `fsq_layer`).
6. **RALM** `forward_step(fusion(lm_hidden, curr_embed_lm))` → `residual_hidden`.

Prefill (once per request) runs the full prompt through Barbet + RALM to warm
the caches and produce the first `lm_hidden` / `residual_hidden`.

## Cache strategy — device-agnostic, hybrid

Instead of flash-attn varlen + triton paged blocks, attention uses **batched
padded KV + SDPA + additive masks** (`torch.nn.functional.scaled_dot_product_attention`
works on CUDA/MPS/CPU). Two cache kinds, allocated up to `max_num_seqs × max_model_len`:

- **`BatchedAttnCache`** (per attention layer): `K,V` of shape
  `[max_seqs, n_kv_heads, max_len, head_dim]`, plus per-row `seq_len`. Decode
  writes new K/V at each row's `seq_len` slot; SDPA runs against the valid
  prefix using a per-row additive mask (causal + optional **sliding window** +
  valid-length). Handles Barbet global (full causal) and sliding layers and the
  RALM (no-RoPE causal) uniformly via the mask.
- **`BatchedMambaState`** (per Barbet mamba layer): `conv_state`
  `[max_seqs, channels, d_conv]` and `ssm_state` `[max_seqs, n_heads, head_dim,
  d_state]`. The recurrence is per-row independent; a decode step advances every
  active row by exactly one position (decode is uniform), so the batched update
  is the single-step Barbet mamba path applied row-wise.

RoPE positions are per-row (`seq_len`), so sequences admitted at different times
and with different prompt lengths coexist in one batch.

**Slot model.** Each running sequence owns a row index (`slot`) into the cache
tensors, assigned on admission and freed on finish. No block table; a free-list
of slots. (Paged blocks are a future optimization, gated on flash-attn.)

## Prefill vs decode — one kernel

Decode does **not** reuse `BarbetModel.forward` / `forward_step` / RALM
`forward_step`: their caches (`BarbetCache.seen_tokens`,
`StaticKVCache.current_length`) are **scalars** shared across the batch, so they
cannot serve ragged continuous-batching positions. We reimplement a single
**batched decode-step kernel** with **per-row position vectors**, reusing the
model's own weights/submodules but our own cache + masks.

- **Prefill = loop the decode-step kernel** over the prompt embeddings, one
  position at a time (batch=1 per new sequence). `tests/test_step_equivalence.py`
  proves `prefill + forward_step == full forward`, so this is provably equal to
  the model's parallel prefill, needs only the one kernel, keeps prefill
  bit-exact, and writes **untrimmed** K/V into the absolute-position buffer.
  Cost is O(prompt_len) sequential micro-steps; **pad-batched prefill is a later
  throughput follow-up**. Crucially, because prefill is also stepwise, the
  **pure-PyTorch single-step mamba path is the ONLY mamba path** ever used — the
  CUDA `mamba_chunk_scan_combined` kernel (different `final_state`/conv-tail
  layout) is never on a parity-relevant path.
- **Decode is fully batched** across all running sequences: one engine step =
  one latent patch for every running sequence, through stages 1–6 above.

The `Scheduler` keeps `waiting`/`running` queues, admits prefills FIFO under
`max_num_seqs`, then runs a batched decode for the running set, mirroring
nano-vLLM's prefill/decode segregation. EOS is the 2-class **stop head** (not a
token); a sequence finishes on `stop_flag==1` (after `min_len`) or `max_len`.

## Module layout

```
src/bluemagpie/serving/
  __init__.py        # BlueMagpieEngine, EngineConfig, Request, RequestOutput
  _caps.py           # runtime capability detection (CUDA graphs / mamba_ssm / GQA / MPS)
  config.py          # EngineConfig (max_num_seqs, max_model_len, device, dtype,
                     #   enforce_eager, inference_timesteps, cfg_value, ...)
  cache.py           # BatchedKVCache, BatchedMambaState, SlotManager, build_allowed_mask
  barbet_batch.py    # BatchedBarbet: per-row batched Barbet decode_step + looped prefill
  minicpm_batch.py   # BatchedRALM (MiniCPM, no-RoPE): per-row decode_step + prefill
  runner.py          # BlueMagpieRunner + SeqState + dit_sample (per-request RNG):
                     #   prefill_batch (pad-batched) + batched decode (DiT->LocEnc->stop->Barbet->RALM)
  engine.py          # BlueMagpieEngine.add_request/step/run/stream, Request/RequestOutput,
                     #   continuous-batching scheduler + AudioVAE finalize
  streaming.py       # BatchedStreamingVAE (dynamic batch-dim conv state) + StreamChunk
  accel.py           # optional torch.compile(reduce-overhead) of DiT+LocEnc, CUDA-gated
  DESIGN.md
tests/
  test_serving_parity.py   # CPU: batched decode_step == per-seq forward_step (ragged);
                           #   batch=1 runner latents == model._inference; engine batching
```
(`graph.py` for CUDA-graph capture and a batched streaming `StreamingVAEDecoder`
are the named follow-ups below, not yet built.)

## Parity & testing (runs on macOS/CPU)

1. **Deterministic component parity** (no RNG): the batched `decode_step` for a
   batch of rows at *ragged positions* must equal looping the single-sequence
   `forward_step` per row. Extends `tests/test_step_equivalence.py`'s contract to
   the batched cache. Covers all three Barbet layer types + the RALM.
2. **Batch=1 end-to-end parity**: with a fixed seed, `BlueMagpieEngine` decoding
   one request produces `feat_pred`/`generated_feat` numerically equal
   (`torch.testing.assert_close`, tight tol) to `model._inference`. The DiT's
   per-patch `z=torch.randn` draw is seeded/threaded identically.
3. **Batch>1 sanity**: two identical requests in one batch produce identical
   latents to each other and to the batch=1 run (same seed per row), proving the
   batched caches/masks don't leak across rows.
4. **CUDA graph parity** (deferred, gated on `torch.cuda.is_available()`):
   `enforce_eager=False` replays the identical computation.

With `enforce_eager=True` the engine is numerically equal to `_inference` on
CPU — provable in CI on the dev's macOS machine. CUDA-only paths are validated
once on a borrowed CUDA box.

## Status (implemented & CPU-verified)

Done and covered by `tests/test_serving_parity.py` (runs on macOS/CPU, tiny models):

- **`cache.py`** — `BatchedKVCache` (absolute-position padded K/V, per-row scatter),
  `BatchedMambaState` (per-row conv tail + SSM state), `SlotManager` (free-list),
  `build_allowed_mask` (causal + sliding-floor + valid, per-row).
- **`barbet_batch.py`** — `BatchedBarbet`: the per-row batched Barbet decode kernel
  (manual matmul attention with qk-clip → mask → sink ordering and the non-CUDA
  fp32 softmax upcast; single-step Mamba via gather/step/scatter on the pure path)
  and `prefill` = looped decode steps. ✅ ragged-position parity vs `forward_step`
  across global/sliding/mamba layers, sink/clip on+off, short-prompt, non-contiguous slots.
- **`minicpm_batch.py`** — `BatchedRALM`: per-row batched no-RoPE MiniCPM step (SDPA +
  manual GQA). ✅ ragged-position parity vs `forward_step`.
- **`runner.py`** — `BlueMagpieRunner`: batched prefill + decode orchestration
  (DiT batched-CFG with **per-request `torch.Generator`** z-draw → `dit_sample`,
  LocEnc, FSQ, stop, Barbet/RALM advance, break-before-step stop). ✅ batch=1 latents
  == `model._inference` (bit-level); ✅ batch=2 identical requests identical; ✅
  continuation + speaker-centroid modes.
- **`engine.py`** — `BlueMagpieEngine`: continuous batching (slot free-list, waiting/
  running, admission, decode grouped by `(timesteps, cfg)`), `Request`/`RequestOutput`,
  **pad-batched prefill** (`prefill_batch`: admitted cohort prefilled in one batched
  pass, O(max L) steps not O(sum L)), AudioVAE decode on finalize, and `stream()`.
  ✅ 3 requests with `max_num_seqs=2` (forced wait + mid-decode join + slot reuse) ==
  per-request singletons; ✅ `prefill_batch` == looped single prefill.
- **`streaming.py`** — `BatchedStreamingVAE` + `StreamChunk`: one batched
  `StreamingVAEDecoder` whose causal-conv state batch dim is realigned to the active
  rows each step (slice on finish, zero-append on join, keyed by request id). Engine
  `stream()` yields per-request chunks. ✅ streamed latents == `run()` latents; ✅
  state reconciliation (drop/join/reorder).
- **`accel.py`** — optional `torch.compile(mode="reduce-overhead")` (captures CUDA
  graphs internally) of the DiT estimator + LocEnc, **hard-gated on CUDA**. ✅ no-op
  on CPU/MPS, engine still constructs and runs eager.
- **`config.py` / `_caps.py`** — `EngineConfig`, capability auto-detection. CUDA-only
  paths are gated; `enforce_eager` default keeps MPS/CPU on the golden path.

### Remaining follow-ups / caveats

- **CUDA-graph speedup needs a CUDA box to validate** — the `accel.py` wiring is gated
  and CPU-verified as a no-op, but the actual graph/latency win can only be measured on
  CUDA (this repo's primary dev box is macOS).
- **Streaming audio for prompt-audio continuation** streams latents only today (the
  shared causal-conv warmup for per-row context lengths is non-uniform); zero-shot /
  reference / speaker-centroid modes stream 48 kHz audio. Use `run()` for continuation audio.
- **Compiling the AR LM decode kernels** needs an on-device KV-length / fixed-shape
  refactor (today `int(positions.max())` forces a graph break); only the DiT + LocEnc
  are compiled.
- **Real paged blocks** — replace the padded per-row buffer once a flash-attn dep is
  acceptable; bigger `max_num_seqs` / longer horizons.

## Usage

```python
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel
from bluemagpie.serving import BlueMagpieEngine, EngineConfig, Request

tok = PreTrainedTokenizerFast(tokenizer_file=f"{model_dir}/tokenizer.json")
model = BlueMagpieModel.from_local(model_dir, tokenizer=tok, device="cuda")  # or "mps" / "cpu"

engine = BlueMagpieEngine(model, EngineConfig(max_num_seqs=16, max_model_len=2048))
engine.add_request(Request(target_text="今天天氣真好。", seed=0))
engine.add_request(Request(target_text="第二句話。", reference_wav_path="speaker.wav"))
for out in engine.run():                 # list[RequestOutput], request-id order
    # out.audio: 48 kHz waveform (when an AudioVAE is attached); out.latents: [T, p, d]
    ...
```

`EngineConfig(enforce_eager=True)` (default) is numerically equal to
`BlueMagpieModel.generate` at batch=1; per-request `seed` makes a request's output
independent of batch composition / admission order.
