# BlueMagpie-TTS — Usage

BlueMagpie-TTS is an open text-to-speech (TTS) model built for **Taiwanese
Mandarin** and **Mandarin–English code-switching**: sentences like
「這個 feature 明天上線」 are read naturally, as 48 kHz speech. On an internal
held-out evaluation it reaches **7.44% CER** vs. a reference baseline's 11.45%
on the same set — a **~35% relative error reduction** (see
[Evaluation](#evaluation)).

🔊 **Try it online:** [BlueMagpie-TTS Demo (Hugging Face Space)](https://huggingface.co/spaces/voidful/BlueMagpie-TTS-Demo)

## Supported environments

| Item | Requirement |
| --- | --- |
| Python | **3.10 – 3.12** (3.13 not yet supported) |
| OS | Linux, macOS (Windows untested) |
| Key dependencies | PyTorch ≥ 2.1, transformers ≥ 4.44 (5.x works), numpy ≥ 1.26 and < 2.4 |
| Model download | ~8 GB (fetched automatically on first run) |
| Hugging Face account | **Not required** (the model is public, not gated) |

| Hardware | Supported | How | Speed* |
| --- | :-: | --- | --- |
| NVIDIA GPU (CUDA) | ✅ | `device="cuda"` | fastest |
| Apple Silicon (MLX, optional) | ✅ | `pip install -e .[mlx]`, see [MLX](#apple-silicon-mlx-acceleration-optional) | RTF ≈ 0.77 (faster than real time) |
| Apple Silicon (MPS) | ✅ | `device="mps"` | RTF ≈ 1.1 |
| CPU | ✅ | `device="cpu"`, see [Running on CPU](#running-on-cpu) | RTF ≈ 2.5 |

<sub>* RTF (real-time factor) = compute seconds per second of synthesized audio;
below 1 is faster than real time. Measured with `scripts/bench_rtf.py` on Apple
M-series, fp32; your hardware will differ.</sub>

Feature support per interface:

| Feature | `generate` | `generate_streaming` | batch engine | MLX |
| --- | :-: | :-: | :-: | :-: |
| Plain synthesis | ✅ | ✅ | ✅ | ✅ |
| Voice cloning (reference clip) | ✅ | ✅ | ✅ | ✅ |
| Voice cloning / speaker selection (speaker vector) | ✅ | ✅ | ✅ | ✅ |
| Continuation (prompt audio) | ✅ | ✅ | ✅ (latents-only when streaming) | ✅ |
| Auto-retry (`retry_badcase`) | ✅ | — | — | — |

## Install

```bash
git clone https://github.com/OpenFormosa/BlueMagpie-TTS
cd BlueMagpie-TTS
pip install -e .
pip install soundfile        # for writing .wav files
```

The install pulls in the [`barbet`](https://github.com/OpenFormosa/Barbet)
package (the text-semantic language model) from GitHub. The acoustic modules are
vendored in `bluemagpie/_vendor/` (sourced from
[VoxCPM](https://github.com/OpenBMB/VoxCPM), Apache-2.0) and need no separate
install.

## Load the model

### From Hugging Face

```python
import os
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS")
# Load the tokenizer straight from tokenizer.json (works on transformers 5.x).
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

The model is public — no token or access request is needed to download it.

### From a local directory

```python
import os
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = "checkpoints/bluemagpie"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

- `device` may be `"cuda"`, `"mps"`, or `"cpu"` (auto-selected if omitted).
- Always use `training=False` for inference.

## Basic synthesis: text to speech

`generate` returns a speech waveform (`torch.Tensor`); pair it with `soundfile`
to write a `.wav`. The output sample rate is `model.sample_rate` (48 kHz).

```python
import soundfile as sf

audio = model.generate(target_text="今天天氣真好。", cfg_value=2.0)
sf.write("output.wav", audio.squeeze().cpu().numpy(), model.sample_rate)
```

## Voice cloning: mimic a reference speaker

Two ways — **neither needs a transcript** of the reference audio.

**A. Reference clip (`reference_wav_path`)** — simplest: pass a clean clip of at
least 3 seconds directly:

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="reference.wav",
    cfg_value=2.0,
)
```

> This path is officially trained starting from checkpoint `step_0006000`
> (released 2026-07; 8.99% CER on the internal eval). **Earlier checkpoints did
> not train this path and produce garbled content** — re-run
> `snapshot_download` to get the latest model.

**B. Speaker vector (`speaker_centroid`)** — most stable quality (7.44% CER):
extract a vector from the reference audio, then synthesize. More clips of the
same speaker average into a cleaner centroid:

```bash
pip install -e ".[clone]"   # extraction needs speechbrain (ECAPA-TDNN)
python scripts/extract_speaker_centroid.py --audio reference.wav --out my_voice.pt
# more clips of the same speaker -> cleaner centroid: --audio a.wav b.wav c.wav
```

```python
import torch

centroid = torch.load("my_voice.pt", weights_only=True)   # [192] speaker vector
audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=centroid,
    cfg_value=2.0,
)

# or extract it in-process:
from bluemagpie import extract_speaker_centroid
centroid = extract_speaker_centroid("reference.wav")      # [192]
```

**Which one?** Quick test with a single short clip → **A**. Production use with
the most stable timbre → **B** (averaged over several clips).

> ⚠️ Only clone voices you are authorized to use.

## Speaker selection: control timbre with a speaker vector

The model bundles a **multi-speaker table** at `checkpoints/speaker_centroids.pt`,
currently holding two speakers:

| speaker id | description | suggested `cfg_value` |
| --- | --- | --- |
| `hung_yi_lee` | Prof. Hung-yi Lee's speaker vector (used with his authorization) | 2.0–2.8 |
| `female_voice` | a generic female voice | 2.0–2.8 |

The table has the format `{"speaker_ids": [...], "centroids": tensor[N, 192], "dim": 192}`.
Load it with `torch.load`, **pick a speaker's `[192]` vector by id**, and pass it as
`speaker_centroid`:

```python
import os
import torch

table = torch.load(
    os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
    map_location="cpu",
    weights_only=True,
)
print(table["speaker_ids"])          # ['hung_yi_lee', 'female_voice']

# switch speaker by changing this line ("hung_yi_lee" or "female_voice")
speaker_id = "female_voice"
speaker_centroid = table["centroids"][table["speaker_ids"].index(speaker_id)]   # [192]

audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=speaker_centroid,   # or your own authorized speaker vector
    cfg_value=2.0,
)
```

If you only have the model id (haven't `snapshot_download`-ed the whole model yet),
grab just the table:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("OpenFormosa/BlueMagpie-TTS", "checkpoints/speaker_centroids.pt")
table = torch.load(path, map_location="cpu", weights_only=True)
```

> To add more speakers, extract your own (authorized) `[192]` vector with
> `extract_speaker_centroid` from the *Voice cloning* section above — it's passed the
> exact same way. The earlier single-speaker file
> `checkpoints/hung_yi_lee_speaker_centroids.pt` (same format) is still available.

## Streaming output

When you need to play while synthesizing, use `generate_streaming`. It is a
generator that yields audio chunks one at a time:

```python
chunks = []
for chunk in model.generate_streaming(target_text="今天天氣真好。"):
    chunks.append(chunk)
    # play or write each chunk in real time here
```

> Note: automatic retry (`retry_badcase`) is not supported in streaming mode.

## Four input modes

The model supports four input combinations through the same `generate` interface:

| Mode | Parameters | Use |
|---|---|---|
| Plain synthesis | `target_text` | Read the text aloud |
| Continuation | `target_text`, `prompt_text`, `prompt_wav_path` | Continue from an existing clip and its text |
| Reference clip | `target_text`, `reference_wav_path` | Mimic the reference speaker's timbre (no transcript needed) |
| Speaker vector | `target_text`, `speaker_centroid` | Clone a voice from a speaker vector |

## Common `generate` parameters

| Parameter | Default | Description |
|---|---|---|
| `target_text` | (required) | The text to synthesize |
| `prompt_text` | `""` | Prompt text, paired with `prompt_wav_path` for continuation |
| `prompt_wav_path` | `""` | Prompt audio path, for continuation |
| `reference_wav_path` | `""` | Reference audio path, for voice cloning |
| `speaker_centroid` | `None` | Speaker vector, to select a timbre |
| `cfg_value` | `2.0` | Guidance strength; higher follows the condition more closely but can sound less natural (suggested 2.0–2.8) |
| `inference_timesteps` | `10` | Sampling steps; more usually means better quality and slower speed |
| `min_len` / `max_len` | `2` / `2000` | Lower / upper bound on output length |
| `retry_badcase` | `False` | Auto-retry on detected bad output (unsupported in streaming; recommended for offline generation) |

## Running on CPU

No GPU required — set `device="cpu"` when loading; everything else is the same:

```python
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cpu")
```

- **Speed:** RTF ≈ 2.5 — about 2.5 compute seconds per second of audio
  (measured on Apple M-series; x86 varies with core count). A 10-and-some
  character sentence finishes in roughly fifteen seconds.
- For long texts, split into sentences and synthesize incrementally (see
  [Streaming output](#streaming-output)) so the first sentence plays sooner.
- On Apple Silicon, prefer the [MLX path](#apple-silicon-mlx-acceleration-optional)
  (~3.3× faster than CPU, faster than real time).

## Evaluation

Numbers from the model's internal held-out evaluation, published with the
checkpoint in
[`release_metadata.json`](https://huggingface.co/OpenFormosa/BlueMagpie-TTS/blob/main/release_metadata.json).
Protocol: synthesized speech is transcribed by an ASR system and compared to the
input text.

| System | Condition | CER ↓ | WER ↓ |
| --- | --- | ---: | ---: |
| **BlueMagpie-TTS** | speaker vector (centroid) | **7.44%** | **8.57%** |
| **BlueMagpie-TTS** | reference clip (no transcript) | 8.99% | 11.77% |
| Reference baseline | same eval set | 11.45% | 14.83% |

**How to read this:** CER (character error rate) is the fraction of characters
that come out wrong (substituted / dropped / inserted) when the synthesized
audio is transcribed back; WER is the word-level analogue. Lower is better. In
the speaker-vector condition, CER drops from the baseline's 11.45% to 7.44% — a
**35% relative reduction**, i.e. roughly 11 → 7 misread characters per 100.

Long-form diagnostics (same checkpoint): centroid CER/WER 7.51%, reference-clip
CER/WER 8.92%, speed ~4.0 chars/sec, cross-chunk speaker-similarity drop
0.109 in reference-clip mode.

> Internal model-selection eval, not a public benchmark; absolute numbers shift
> with a different ASR or eval set — read them as same-condition comparisons.

## Batch serving engine (multi-request acceleration)

To serve many synthesis requests at once for higher throughput, use the built-in
batch engine `BlueMagpieEngine`. It does **continuous batching**: requests are
decoded together as a batch, new requests can join mid-decode, and they do not
interfere with one another.

Highlights:

- **No extra dependencies** — torch only; no vLLM, flash-attn, etc.
- **Cross-device** — one code path on CUDA, Apple Silicon (MPS), and CPU.
  CUDA-only optimizations are auto-detected and enabled, and skipped elsewhere.
- **Numerically identical to single-call `generate`** at batch=1 (`model.generate`
  is always the reference).

### Basic usage

```python
import soundfile as sf
from bluemagpie.serving import BlueMagpieEngine, EngineConfig, Request

# load `model` and `tokenizer` as shown above (from_local)
engine = BlueMagpieEngine(model, EngineConfig(max_num_seqs=16))

engine.add_request(Request(target_text="今天天氣真好。", seed=0))
engine.add_request(Request(target_text="第二句話。", reference_wav_path="speaker.wav"))

for out in engine.run():            # returned in request-id (submission) order
    # out.audio: 48 kHz waveform (when an AudioVAE is attached); out.latents: [T, p, d]
    sf.write(f"output_{out.request_id}.wav", out.audio.numpy(), out.sample_rate)
```

`Request` supports the same four input modes as `generate` (plain, continuation,
reference clip, speaker vector) via the fields `target_text`, `prompt_text`,
`prompt_wav_path`, `reference_wav_path`, `speaker_centroid`, `cfg_value`,
`inference_timesteps`, etc. Each request may set a `seed`, which makes its output
independent of how many neighbours share the batch and of admission order.

### Streaming

`engine.stream()` is a generator that yields a chunk per request per step:

```python
for chunk in engine.stream():
    # chunk.request_id, chunk.latents, chunk.audio, chunk.finished
    play_or_write(chunk)
```

> Plain synthesis, reference-clip, and speaker-vector modes stream audio
> (`chunk.audio`); prompt-audio continuation streams `latents` only — use `run()`
> when you need its audio.

### Configuration

Common `EngineConfig` parameters:

| Parameter | Default | Description |
|---|---|---|
| `max_num_seqs` | `16` | Max concurrent requests batched together |
| `max_model_len` | `2048` | Max length per sequence (prompt + generated) |
| `inference_timesteps` | `10` | Sampling steps |
| `cfg_value` | `2.0` | Guidance strength |
| `enforce_eager` | `True` | Keep the path numerically identical to single-call `generate` |
| `compile` | `False` | Enable `torch.compile` (CUDA only; auto-skipped elsewhere) |

> See [`src/bluemagpie/serving/DESIGN.md`](src/bluemagpie/serving/DESIGN.md) for the
> engine's design, trade-offs, and known limitations.

### Why not just use vLLM?

People often expect "wrap it in vLLM and it gets fast", but for BlueMagpie that
does not work, for two reasons:

1. **The real compute bottleneck is the diffusion decoder, not the language
   model.** Per generated audio unit the DiT (LocDiT / CFM diffusion decoder) is
   called "sampling steps × the unconditional/conditional CFG pair" times —
   about 20 at the default 10 steps — while the language models (Barbet, RALM)
   run once each. vLLM is a
   *text language-model* inference framework — it does not touch the diffusion
   decoder at all, so even moving the LMs onto vLLM leaves the dominant compute
   running eagerly and barely moves end-to-end latency.
2. **vLLM does not support Barbet's hybrid architecture.** Barbet (the
   text-semantic LM) is a Mamba2 + attention hybrid, and vLLM (as well as
   nano-vllm and vllm-omni) has zero support for such a hybrid TSLM — you'd have
   to implement a first-class hybrid model yourself (large effort, CUDA-only).

So this engine **borrows vLLM's architectural techniques without depending on its
CUDA kernels**:

- **Continuous batching** of many requests (the main throughput win), sharing
  batched compute across requests.
- A **padded KV cache + SDPA + masks** instead of vLLM's PagedAttention /
  FlashAttention — trading peak speed and memory efficiency for cross-device,
  zero-dependency portability.
- Barbet's Mamba state handled with a **pure-PyTorch single-step recurrence**, no
  fused kernel required.
- Optional `compile=True` uses `torch.compile` (which captures CUDA graphs
  internally) to accelerate the **DiT and LocEnc** — the actual hot path, and
  exactly what wrapping in vLLM would *not* do for you.

> In short: we don't aim to beat vLLM on a single op; we use vLLM-class **batch
> scheduling** plus **DiT-bottleneck optimization** to raise overall throughput
> with no extra dependencies, across CUDA / MPS / CPU.

## Apple Silicon MLX acceleration (optional)

On Apple Silicon (M-series), a native **MLX** path runs inference directly on the
Apple GPU (Metal, unified memory) — typically faster than PyTorch's MPS backend.
It is an optional extra; the core package stays torch-only:

```bash
pip install -e .[mlx]
```

```python
import soundfile as sf
from bluemagpie import BlueMagpieModel
from bluemagpie.mlx import BlueMagpieMLX, mlx_generate

model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, device="cpu")
mlx_model = BlueMagpieMLX(model)          # converts the weights once

audio = mlx_generate(model, mlx_model, "今天天氣真好。", seed=0)   # 48 kHz waveform
sf.write("output.wav", audio.numpy(), model.sample_rate)
```

- The whole inference path (Barbet, RALM, LocEnc, LocDiT/CFM, the **AudioVAE
  decoder**, the AR loop) is re-implemented in MLX and numerically parity-checked,
  module by module — generation can run torch-free (only tokenization and
  reference-wav encoding stay in torch).
- Decode uses cached single-step kernels (it advances one position per step, not a
  full re-run).
- `mlx_generate` supports the same four input modes as `generate`.
- On the real 7.75 GB model: end-to-end **RTF 0.77** (faster than real time) —
  ~**1.45×** over torch-MPS and ~**3.27×** over torch-CPU (fp32,
  `scripts/bench_rtf.py`). See [`src/bluemagpie/mlx/DESIGN.md`](src/bluemagpie/mlx/DESIGN.md).

## Use cases & collaboration

BlueMagpie-TTS fits anywhere you need "a Taiwanese voice":

- **Audio content** — turn news, blogs, or newsletters into narrated audio
  (batch or streaming synthesis for long texts).
- **Customer service & voice UIs** — IVR prompts and dialogue replies; mixed
  Mandarin–English product names and acronyms are read as-is.
- **Accessibility** — natural Taiwanese-Mandarin read-aloud for web and documents.
- **Research & teaching** — fully open inference code (including the batch
  engine and the MLX port) as a base for TTS research.

The project is maintained by the [OpenFormosa](https://github.com/OpenFormosa)
community. We are looking for:

- **Speech data & speaker licensing** partners (more Taiwanese voices; Taiwanese
  Hokkien / Hakka extensions).
- **Deployment pilots** — teams with real scenarios; open an issue to discuss.
- **Contributions** — performance, hardware support, documentation.

Contact: [GitHub Issues](https://github.com/OpenFormosa/BlueMagpie-TTS/issues)
or the [Hugging Face discussion board](https://huggingface.co/OpenFormosa/BlueMagpie-TTS/discussions).

## Notes

- The examples load the tokenizer from `tokenizer.json` and pass it to
  `from_local`, which is stable on transformers 5.x. (`from_local`'s automatic
  tokenizer loading can fail on 5.x — see Troubleshooting.)
- The bundled `hung_yi_lee` speaker vector is authorized for example use. For any
  other speaker or voice cloning, use only reference audio or speaker vectors you
  are authorized to use.
- Keep speaker-vector tables and synthesized audio private; do not distribute
  them without authorization.
- Generated speech may be imperfect; review it before real-world use.

## Troubleshooting

**`Token is required (token=True), but no token found` when downloading.** The
model is public — no token is needed. If your snippet has
`snapshot_download(..., token=True)`, drop the `token=True`; or log in first
with `huggingface-cli login`.

**`pip install -e .` fails.** Check your Python version first: this package
supports **3.10–3.12**; Python 3.13+ is not yet supported. Dependencies require
`numpy>=1.26,<2.4` — if your environment pins a conflicting numpy, install into
a fresh virtual environment.

**Voice cloning (`reference_wav_path`) outputs garbled content.** Make sure you
are on checkpoint `step_0006000` (2026-07) or later — earlier checkpoints did
not train this path. Re-running `snapshot_download("OpenFormosa/BlueMagpie-TTS")`
updates to the latest revision. Also provide at least 3 seconds of clean speech.

**Tokenizer loading on newer transformers (5.x).** The examples load the
tokenizer explicitly from `tokenizer.json`, so they work on transformers 5.x with
no extra steps (the model only uses the tokenizer's `encode`).

If you instead rely on `from_local`'s automatic tokenizer loading (passing no
`tokenizer`), transformers 5.x may fail while parsing `tokenizer_config.json`
with `TypeError: ..._patch_mistral_regex() got multiple values for keyword
argument 'fix_mistral_regex'`, or appear to load but raise `ValueError: No
tokenizer attached to BlueMagpieModel` when you call `generate()`. Use the
explicit loading shown above instead.
