# BlueMagpie-TTS — Usage

BlueMagpie-TTS is a text-to-speech (TTS) model that synthesizes natural speech
from text. It supports three scenarios:

- **Plain synthesis** — read the text aloud.
- **Voice cloning** — mimic the timbre of a reference clip.
- **Speaker selection** — control the timbre with a prepared speaker vector.

It also supports **streaming output** for synthesize-while-you-play applications.

🔊 **Try it online:** [BlueMagpie-TTS Demo (Hugging Face Space)](https://huggingface.co/spaces/voidful/BlueMagpie-TTS-Demo)

## Install

```bash
git clone https://github.com/OpenFormosa/BlueMagpie-TTS
cd BlueMagpie-TTS
pip install -e .
```

The install pulls in the [`barbet`](https://github.com/OpenFormosa/Barbet)
package (the text-semantic language model) from GitHub. The acoustic modules are
vendored in `bluemagpie/_vendor/` (sourced from
[VoxCPM](https://github.com/OpenBMB/VoxCPM), Apache-2.0) and need no separate
install. To save synthesized audio, also install `soundfile`:

```bash
pip install soundfile
```

## Load the model

### From Hugging Face

```python
import os
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS", token=True)
# Load the tokenizer straight from tokenizer.json (works on transformers 5.x).
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

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

Two ways.

**A. Speaker vector (`speaker_centroid`)** — extract a vector from the reference
audio, then synthesize (no transcript needed):

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
    cfg_value=2.8,
)

# or extract it in-process:
from bluemagpie import extract_speaker_centroid
centroid = extract_speaker_centroid("reference.wav")      # [192]
```

**B. Reference clip (`reference_wav_path`)** — pass a reference clip directly:

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="reference.wav",
    cfg_value=2.8,
)
```

## Speaker selection: control timbre with a speaker vector

The model bundles Prof. Hung-yi Lee's speaker vector as an example (used with his
authorization), at `checkpoints/hung_yi_lee_speaker_centroids.pt`. Load the table
with `torch.load`, pick the vector by speaker id, and pass `speaker_centroid`:

```python
import os
import torch

centroids = torch.load(
    os.path.join(model_dir, "checkpoints", "hung_yi_lee_speaker_centroids.pt"),
    map_location="cpu",
    weights_only=True,
)
speaker_centroid = centroids["centroids"][centroids["speaker_ids"].index("hung_yi_lee")]

audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=speaker_centroid,   # or your own authorized speaker vector
    cfg_value=2.0,
)
```

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
| Reference clip | `target_text`, `reference_wav_path` | Mimic the reference speaker's timbre |
| Speaker vector | `target_text`, `speaker_centroid` | Clone a voice from a speaker vector |

## Common `generate` parameters

| Parameter | Default | Description |
|---|---|---|
| `target_text` | (required) | The text to synthesize |
| `prompt_text` | `""` | Prompt text, paired with `prompt_wav_path` for continuation |
| `prompt_wav_path` | `""` | Prompt audio path, for continuation |
| `reference_wav_path` | `""` | Reference audio path, for voice cloning |
| `speaker_centroid` | `None` | Speaker vector, to select a timbre |
| `cfg_value` | `2.0` | Guidance strength; higher follows the condition more closely but can sound less natural |
| `inference_timesteps` | `10` | Sampling steps; more usually means better quality and slower speed |
| `min_len` / `max_len` | `2` / `2000` | Lower / upper bound on output length |
| `retry_badcase` | `False` | Auto-retry on detected bad output (unsupported in streaming) |

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
| `inference_timesteps` | `9` | Sampling steps |
| `cfg_value` | `2.8` | Guidance strength |
| `enforce_eager` | `True` | Keep the path numerically identical to single-call `generate` |
| `compile` | `False` | Enable `torch.compile` (CUDA only; auto-skipped elsewhere) |

> See [`src/bluemagpie/serving/DESIGN.md`](src/bluemagpie/serving/DESIGN.md) for the
> engine's design, trade-offs, and known limitations.

### Why not just use vLLM?

People often expect "wrap it in vLLM and it gets fast", but for BlueMagpie that
does not work, for two reasons:

1. **The real compute bottleneck is the diffusion decoder, not the language
   model.** Per generated audio unit the DiT (LocDiT / CFM diffusion decoder) is
   called ~16–18 times (sampling steps × the unconditional/conditional CFG
   pair), while the language models (Barbet, RALM) run once each. vLLM is a
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

## Notes

- The examples load the tokenizer from `tokenizer.json` and pass it to
  `from_local`, which is stable on transformers 5.x. (`from_local`'s automatic
  tokenizer loading can fail on 5.x — see Troubleshooting.)
- A GPU is optional: set `device="cpu"` (slower, but short utterances take only
  tens of seconds). Output is 48 kHz mono.
- The bundled `hung_yi_lee` speaker vector is authorized for example use. For any
  other speaker or voice cloning, use only reference audio or speaker vectors you
  are authorized to use.
- Keep speaker-vector tables and synthesized audio private; do not distribute
  them without authorization.

## Troubleshooting

**Tokenizer loading on newer transformers (5.x).** The examples load the
tokenizer explicitly from `tokenizer.json`, so they work on transformers 5.x with
no extra steps (the model only uses the tokenizer's `encode`).

If you instead rely on `from_local`'s automatic tokenizer loading (passing no
`tokenizer`), transformers 5.x may fail while parsing `tokenizer_config.json`
with `TypeError: ..._patch_mistral_regex() got multiple values for keyword
argument 'fix_mistral_regex'`, or appear to load but raise `ValueError: No
tokenizer attached to BlueMagpieModel` when you call `generate()`. Use the
explicit loading shown above instead.
