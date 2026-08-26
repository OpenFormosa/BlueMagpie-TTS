"""BlueMagpie MLX "lite" inference: fp16 weights, cached checkpoint, no full
torch model in RAM.

The stock MLX path (``BlueMagpieMLX(model)``) holds the fp32 torch model
(~7.75 GB) *and* an fp32 MLX copy of every weight simultaneously — ~15+ GB
peak, which is uncomfortable on 16 GB Macs. This module fixes both problems:

- **fp16 weights**: every ``mx.array`` under the MLX model is converted to
  float16 (~4 GB instead of ~7.75 GB). Metal's fp16 GEMMs also run ~2x the
  memory bandwidth of fp32, so inference gets faster, not just smaller.
- **cached checkpoint**: the converted model is pickled once; every later
  launch loads the pickle directly. No torch model, no re-conversion, and —
  because only the torch AudioVAE *encoder* is needed for input assembly — the
  torch side is a lightweight proxy built from ``audiovae.pth`` alone, not the
  full 7.75 GB checkpoint.

Usage::

    # one-time conversion (~3-6 min, ~15 GB transient peak):
    convert_checkpoint(model_dir, "bluemagpie-lite.pkl")

    # every run (~5 GB steady-state):
    lite, mlx_model = load_model(model_dir, "bluemagpie-lite.pkl")
    audio = mlx_generate(lite, mlx_model, "今天天氣真好。")

``mlx_generate`` from ``model_mlx`` works unchanged: it only touches
``model._encode_wav`` / ``model._build_inputs`` / ``model.config`` /
``model.audio_vae``, all of which the lite proxy provides.
"""

from __future__ import annotations

import gc
import os
import pickle
from typing import Dict, Optional

import mlx.core as mx
import torch

from bluemagpie.config import BlueMagpieConfig
from bluemagpie.mlx.audiovae_mlx import AudioVAEMLX
from bluemagpie.mlx.model_mlx import BlueMagpieMLX, mlx_generate


# --------------------------------------------------------------------------- #
# fp16 conversion
# --------------------------------------------------------------------------- #
def _fp16(obj, visited: set) -> object:
    """Recursively convert every ``mx.array`` in the object graph to fp16.

    Handles the exact structures used by the MLX modules: mx.array, dict
    (``self.p`` flat weight dicts), tuple (``(weight, bias)`` pairs) and
    generic objects with ``__dict__`` (nested MLX modules). Torch tensors and
    scalars are left untouched.
    """
    if isinstance(obj, mx.array):
        # Only cast float32 weights; integer index arrays (e.g. Barbet's
        # group_for_head), fp64 rope factors and already-fp16 arrays must
        # keep their dtype (casting indices to fp16 breaks mx.take).
        if obj.dtype == mx.float32:
            return obj.astype(mx.float16)
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            nv = _fp16(v, visited)
            if nv is not v:
                obj[k] = nv
        return obj
    if isinstance(obj, tuple):
        return tuple(_fp16(v, visited) for v in obj)
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nv = _fp16(v, visited)
            if nv is not v:
                obj[i] = nv
        return obj
    if hasattr(obj, "__dict__"):
        if id(obj) in visited:
            return obj
        visited.add(id(obj))
        for k, v in list(vars(obj).items()):
            nv = _fp16(v, visited)
            if nv is not v:
                setattr(obj, k, nv)
        return obj
    return obj


def _fp32(obj, visited: set) -> object:
    """Reverse of :func:`_fp16`: cast float16 mx arrays back to float32."""
    if isinstance(obj, mx.array):
        if obj.dtype == mx.float16:
            return obj.astype(mx.float32)
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            nv = _fp32(v, visited)
            if nv is not v:
                obj[k] = nv
        return obj
    if isinstance(obj, tuple):
        return tuple(_fp32(v, visited) for v in obj)
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nv = _fp32(v, visited)
            if nv is not v:
                obj[i] = nv
        return obj
    if hasattr(obj, "__dict__"):
        if id(obj) in visited:
            return obj
        visited.add(id(obj))
        for k, v in list(vars(obj).items()):
            nv = _fp32(v, visited)
            if nv is not v:
                setattr(obj, k, nv)
        return obj
    return obj


# --------------------------------------------------------------------------- #
# int4 (MLX grouped-affine) quantization
# --------------------------------------------------------------------------- #
def _quantize_int4(mlx_model, group: int = 64, bits: int = 4,
                   keep_fp32_for_prefill: bool = False) -> dict:
    """Quantize all 2-D Linear weights in place to MLX grouped affine.

    The fused ``mx.quantized_matmul`` kernel is dispatched through the _lin
    hook in barbet_mlx (weights stay packed — no dequantized fp32 blobs).
    Embedding tables and anything not shaped [*, group-divisible] stays fp32.

    ``keep_fp32_for_prefill``: retain the fp32 weights so prefill calls take
    the plain GEMM path (_PREFILL_FP32 flag). Measured on a 16 GB M4: NOT
    worth it — the extra ~5.6 GB resident causes memory pressure that slows
    the whole pipeline more than the prefill saving. Only enable on ≥32 GB
    machines.
    """
    import mlx.core as mx
    from .barbet_mlx import register_quantized

    stats = {"quantized": 0, "bytes_saved": 0}
    store = getattr(mlx_model, "_quant_store", {})
    mlx_model._quant_store = store

    def walk(obj, visited, path):
        if isinstance(obj, mx.array):
            a = obj
            # dtype guard does the dedup: already-packed (uint32) or non-fp32
            # arrays are skipped, so a shared reference reaching a second path
            # just gets its own packed copy registered under its own id.
            if a.dtype != mx.float32 or a.ndim != 2 or a.shape[-1] % group != 0 or "embed" in path:
                return obj
            wq, sc, bi = mx.quantize(a, group_size=group, bits=bits)
            if sc.dtype != mx.float32 or (bi is not None and bi.dtype != mx.float32):
                raise RuntimeError(
                    f"quantize produced non-fp32 scales/biases at {path}: "
                    f"{sc.dtype}/{bi.dtype if bi is not None else None}"
                )
            register_quantized(wq, sc, bi, group, bits,
                               fp32_backup=a if keep_fp32_for_prefill else None)
            # side-car store survives pickling; reactivate_int4() re-registers
            # the (fresh) ids after unpickling.
            store[path] = {"wq": wq, "scales": sc, "biases": bi,
                           "group": group, "bits": bits}
            stats["quantized"] += 1
            stats["bytes_saved"] += 0 if keep_fp32_for_prefill else (
                a.size * 4 - (wq.size * 4 + sc.size * 4))
            return wq
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                nv = walk(v, visited, f"{path}.{k}" if path else k)
                if nv is not v:
                    obj[k] = nv
            return obj
        if isinstance(obj, tuple):
            return tuple(walk(v, visited, f"{path}[{i}]") for i, v in enumerate(obj))
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                nv = walk(v, visited, f"{path}[{i}]")
                if nv is not v:
                    obj[i] = nv
            return obj
        if hasattr(obj, "__dict__"):
            if id(obj) in visited:
                return obj
            visited.add(id(obj))
            for k, v in list(vars(obj).items()):
                if k.startswith("_quant"):
                    # never descend into the side-car store itself: its fp32
                    # scales are 2-D and group-divisible for large layers, so
                    # recursing into it would quantize the scales and corrupt
                    # the entries (this exact bug produced uint32 scales).
                    continue
                nv = walk(v, visited, f"{path}.{k}" if path else k)
                if nv is not v:
                    setattr(obj, k, nv)
            return obj
        return obj

    walk(mlx_model, set(), "")
    return stats


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert_checkpoint(model_dir: str, out_path: str, precision: str = "fp32", verbose: bool = True) -> str:
    """Load the torch checkpoint, build the MLX model, convert and pickle it
    to ``out_path``.

    ``precision``:
      - ``"fp32"`` (default): weights kept as-is (~7.75 GB pickle; on Apple
        Silicon MLX fp32 GEMMs are *faster* than fp16, so this is both the
        highest-quality and fastest option).
      - ``"fp16"``: weights cast to float16 (~4 GB pickle; ~40% less RAM at
        inference and slightly worse quality — measure with
        ``scripts/bench_compare.py`` before relying on it).

    The torch model is released before pickling, so the transient peak is
    roughly (fp32 torch 7.75 GB + MLX copy) ≈ 12-15 GB — heavy but feasible
    on 16 GB Macs (macOS swaps the tail). Run this once.
    """
    from bluemagpie import BlueMagpieModel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
    if verbose:
        print(f"[lite] loading torch checkpoint from {model_dir} ...", flush=True)
    model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, device="cpu")
    if verbose:
        print("[lite] converting weights to MLX ...", flush=True)
    mlx_model = BlueMagpieMLX(model)
    if precision == "fp16":
        if verbose:
            print("[lite] converting weights to float16 ...", flush=True)
        _fp16(mlx_model, set())
    elif precision not in ("fp32",):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")
    del model
    gc.collect()
    if verbose:
        print(f"[lite] pickling to {out_path} ...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(mlx_model, f, protocol=4)
    size_mb = os.path.getsize(out_path) / 1e6
    if verbose:
        print(f"[lite] done: {size_mb:.0f} MB checkpoint (precision={precision})", flush=True)
    return out_path


def _load_config(model_dir: str) -> BlueMagpieConfig:
    import json

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return BlueMagpieConfig.model_validate_json(f.read())


def _build_lite_torch(model_dir: str, config: BlueMagpieConfig, tokenizer):
    """A lightweight ``BlueMagpieModel`` proxy: only the AudioVAE encoder
    weights (``audiovae.pth``) + config/token metadata. Everything the input
    assembly of ``mlx_generate`` needs, nothing else.
    """
    import types

    from bluemagpie import BlueMagpieModel
    from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2

    vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
    vae_path = os.path.join(model_dir, "audiovae.pth")
    if os.path.exists(vae_path):
        st = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae.load_state_dict(st.get("state_dict", st))

    model = object.__new__(BlueMagpieModel)  # skip __init__ (no 7.75 GB init)

    def _set(k, v):
        # nn.Module.__setattr__ refuses assignments before Module.__init__;
        # we want plain attribute slots on the proxy regardless.
        object.__setattr__(model, k, v)

    _set("config", config)
    _set("text_tokenizer", tokenizer)
    _set("audio_vae", vae)
    _set("feat_dim", config.feat_dim)
    _set("patch_size", config.patch_size)
    _set("chunk_size", vae.chunk_size)
    _set("_encode_sample_rate", vae.sample_rate)
    _set("sample_rate", getattr(vae, "out_sample_rate", vae.sample_rate))

    _barbet_cfg, token_ids = config.resolve_barbet_config()
    _set("audio_start_token", token_ids["audio_start"])
    _set("audio_end_token", token_ids["audio_end"])
    _set("ref_audio_start_token", token_ids["ref_audio_start"])
    _set("ref_audio_end_token", token_ids["ref_audio_end"])
    _set("spk_token", token_ids["spk"])

    _set("device", "cpu")
    # _runtime_device() normally walks parameters(); the proxy has none.
    _set("_runtime_device", types.MethodType(lambda self: torch.device("cpu"), model))
    _set("_runtime_dtype", types.MethodType(lambda self: torch.float32, model))
    return model


def load_model(
    model_dir: str,
    cache_path: str,
    tokenizer=None,
    compute: str = "fp32",
) -> tuple:
    """Load the lite checkpoint + a lite torch proxy. Returns
    ``(lite_model, mlx_model)`` ready for ``mlx_generate``.

    ``compute``:
      - ``"fp32"`` (default): full precision (~7.75 GB weights, RTF ~2.0).
      - ``"fp16"``: half weight RAM, slower on M-series, slightly lower quality.
      - ``"int4"``: MLX grouped-affine 4-bit (group 64) on all Linear weights,
        fused ``mx.quantized_matmul`` kernels (~2 GB weights, RTF ~0.9 — the
        fastest option; embedding tables stay fp32). User-listener approved.
    """
    from transformers import PreTrainedTokenizerFast

    if tokenizer is None:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))

    with open(cache_path, "rb") as f:
        mlx_model = pickle.load(f)

    if compute == "fp32":
        _fp32(mlx_model, set())
        mlx_model._samplers = {}  # compiled traces were built for the stored dtype
    elif compute == "int4":
        _fp32(mlx_model, set())
        mlx_model._samplers = {}
        _quantize_int4(mlx_model)
    elif compute == "int4-cached":
        # Checkpoint already carries packed weights + the quant side-car;
        # just re-register the kernels (ids changed during unpickling).
        n = reactivate_int4(mlx_model)
        if n == 0:
            raise ValueError(
                f"{cache_path} has no quant side-car; convert it first with "
                "save_int4_checkpoint() on an int4-quantized model"
            )
        print(f"[lite] reactivated {n} int4 weights")
    elif compute != "fp16":
        raise ValueError(f"compute must be 'fp32', 'fp16', 'int4' or 'int4-cached', got {compute!r}")

    if compute in ("int4", "int4-cached") and os.environ.get("BM_EMBED_FP16", "1") == "1":
        # fp16 embedding: halves RAM (674->337MB) with ~1e-4 latent drift
        # (verified bit-safe vs int8/int4 embed variants; see scripts/ab_embed_*.py)
        import mlx.core as mx
        mlx_model.embed = mlx_model.embed.astype(mx.float16)
        print("[lite] embedding -> fp16 (saved ~337 MB)")

    config = _load_config(model_dir)

    # Attach a torch AudioVAE (structure for the MLX decoder + torch fallback).
    from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2

    vae = AudioVAEV2(config=config.audio_vae_config) if config.audio_vae_config else AudioVAEV2()
    vae_path = os.path.join(model_dir, "audiovae.pth")
    if os.path.exists(vae_path):
        st = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae.load_state_dict(st.get("state_dict", st))
    # AudioVAEMLX keeps only structure + lazy weight conversion, so rebuilding
    # it from the torch vae is cheap.
    mlx_model.vae = AudioVAEMLX(vae)

    lite = _build_lite_torch(model_dir, config, tokenizer)
    return lite, mlx_model


__all__ = ["convert_checkpoint", "load_model", "mlx_generate"]


def reactivate_int4(mlx_model) -> int:
    """Re-register quantized weights after unpickling (ids change on load).

    Returns the number of weights re-registered. Safe to call on a model that
    was never quantized (returns 0).
    """
    from .barbet_mlx import register_quantized

    store = getattr(mlx_model, "_quant_store", {})
    if not store:
        return 0

    # Walk with the same path naming as _quantize_int4 so entries match.
    import mlx.core as mx
    reactivated = 0

    def walk(obj, visited, path):
        nonlocal reactivated
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                walk(v, visited, f"{path}.{k}" if path else k)
        elif isinstance(obj, tuple):
            for i, v in enumerate(obj):
                walk(v, visited, f"{path}[{i}]")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, visited, f"{path}[{i}]")
        elif hasattr(obj, "__dict__"):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            for k, v in list(vars(obj).items()):
                walk(v, visited, f"{path}.{k}" if path else k)
        elif isinstance(obj, mx.array):
            ent = store.get(path)
            if ent is not None:
                register_quantized(ent["wq"], ent["scales"], ent["biases"],
                                   ent["group"], ent["bits"])
                reactivated += 1

    walk(mlx_model, set(), "")
    return reactivated


def save_int4_checkpoint(mlx_model, out_path: str) -> str:
    """Pickle the current (int4) mlx_model including the quant side-car.

    The model must already be int4-quantized (load_model(compute="int4") or a
    fresh _quantize_int4 call). Load it back with
    ``load_model(..., cache_path, compute="int4-cached")``.
    """
    if not getattr(mlx_model, "_quant_store", {}):
        raise ValueError("model is not int4-quantized; nothing to save")
    with open(out_path, "wb") as f:
        pickle.dump(mlx_model, f, protocol=4)
    print(f"[lite] saved int4 checkpoint: {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.0f} MB)")
    return out_path