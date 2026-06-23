"""BlueMagpieEngine: continuous-batching serving over the batched runner.

Continuous batching, nano-vLLM style: a free-list of cache slots, a waiting
queue, and a running set. Each ``step()`` either admits waiting requests
(prefill into free slots) or runs one batched decode over the running set,
grouped by ``(inference_timesteps, cfg_value)`` so one DiT call stays valid.
Requests added mid-flight join the running decode as slots free up.

Input assembly (text/ref/prompt audio/speaker) reuses the model's own
``_build_inputs`` / ``_encode_wav`` so the engine matches ``model.generate``'s
prompting modes exactly. Tests use :meth:`submit_prefill_inputs` to bypass the
tokenizer/VAE and drive the loop with synthetic latents.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import EngineConfig
from .cache import SlotManager
from .runner import BlueMagpieRunner, SeqState
from .streaming import BatchedStreamingVAE, StreamChunk


@dataclass
class Request:
    target_text: str
    prompt_text: str = ""
    prompt_wav_path: str = ""
    reference_wav_path: str = ""
    speaker_centroid: Optional[torch.Tensor] = None
    min_len: Optional[int] = None
    max_len: Optional[int] = None
    inference_timesteps: Optional[int] = None
    cfg_value: Optional[float] = None
    use_null_speaker: bool = True
    seed: Optional[int] = None
    retry_badcase_ratio_threshold: float = 6.0


@dataclass
class RequestOutput:
    request_id: int
    latents: torch.Tensor               # [T, p, d]
    audio: Optional[torch.Tensor] = None  # [samples] if an AudioVAE is attached
    sample_rate: Optional[int] = None


@dataclass
class _Running:
    request_id: int
    state: SeqState


class BlueMagpieEngine:
    def __init__(self, model, config: Optional[EngineConfig] = None) -> None:
        self.model = model
        self.config = config or EngineConfig()
        device = self.config.resolved_device(model._runtime_device())
        if str(device) != str(model._runtime_device()):
            raise ValueError(
                f"EngineConfig.device={device!r} != model device {model._runtime_device()!r}; "
                "move the model first."
            )
        if self.config.compile and not self.config.enforce_eager:
            from .accel import optimize_for_inference

            optimize_for_inference(model, mode=self.config.compile_mode)
        self.runner = BlueMagpieRunner(model, self.config.max_num_seqs, self.config.max_model_len)
        self.slots = SlotManager(self.config.max_num_seqs)
        self.waiting: deque = deque()
        self.running: List[_Running] = []
        self._next_id = 0
        self._outputs: Dict[int, RequestOutput] = {}

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #
    def add_request(self, req: Request) -> int:
        rid = self._next_id
        self._next_id += 1
        self.waiting.append((rid, req))
        return rid

    def submit_prefill_inputs(
        self,
        text_token,
        audio_feat,
        text_mask,
        audio_mask,
        spk_mask=None,
        speaker_centroids=None,
        *,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: Optional[int] = None,
        cfg_value: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> int:
        """Test/low-level entry: enqueue a request from pre-built tensors."""
        rid = self._next_id
        self._next_id += 1
        self.waiting.append(
            (
                rid,
                _PrebuiltInputs(
                    text_token=text_token,
                    audio_feat=audio_feat,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    spk_mask=spk_mask,
                    speaker_centroids=speaker_centroids,
                    min_len=min_len,
                    max_len=max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    seed=seed,
                ),
            )
        )
        return rid

    # ------------------------------------------------------------------ #
    # Prefill / admission
    # ------------------------------------------------------------------ #
    def _make_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        g = torch.Generator(device=self.runner.device)
        g.manual_seed(int(seed))
        return g

    def _apply_params(self, st: SeqState, *, min_len, max_len, inference_timesteps, cfg_value, seed) -> None:
        st.min_len = min_len
        st.max_len = max_len
        st.n_timesteps = inference_timesteps if inference_timesteps is not None else self.config.inference_timesteps
        st.cfg_value = cfg_value if cfg_value is not None else self.config.cfg_value
        st.generator = self._make_generator(seed)

    def _prepare_prebuilt(self, slot: int, pb: "_PrebuiltInputs"):
        prep = self.runner._prepare(
            slot, pb.text_token, pb.audio_feat, pb.text_mask, pb.audio_mask,
            pb.spk_mask, pb.speaker_centroids, self.config.streaming_prefix_len,
        )
        params = dict(min_len=pb.min_len, max_len=pb.max_len, inference_timesteps=pb.inference_timesteps,
                      cfg_value=pb.cfg_value, seed=pb.seed)
        return prep, params

    def _prepare_request(self, slot: int, req: Request):
        m = self.model
        device = self.runner.device
        dtype = self.runner.dtype
        ref_feat = m._encode_wav(req.reference_wav_path, padding_mode="right") if req.reference_wav_path else None
        prompt_feat = m._encode_wav(req.prompt_wav_path, padding_mode="left") if req.prompt_wav_path else None
        text = (req.prompt_text + req.target_text) if prompt_feat is not None else req.target_text

        speaker_centroids = None
        if req.speaker_centroid is not None:
            speaker_centroids = req.speaker_centroid.reshape(1, -1).to(device, dtype=dtype)
        speaker_slot = (
            "centroid" if speaker_centroids is not None else ("null" if req.use_null_speaker else "none")
        )
        text_token, audio_feat, text_mask, audio_mask, spk_mask = m._build_inputs(
            text, ref_feat, prompt_feat, speaker_slot=speaker_slot
        )
        prep = self.runner._prepare(
            slot, text_token, audio_feat, text_mask, audio_mask, spk_mask, speaker_centroids,
            self.config.streaming_prefix_len,
        )
        target_len = len(m._tokenize(req.target_text))
        max_len = min(int(target_len * req.retry_badcase_ratio_threshold + 10), req.max_len or self.config.max_len)
        params = dict(
            min_len=req.min_len if req.min_len is not None else self.config.min_len,
            max_len=max_len,
            inference_timesteps=req.inference_timesteps,
            cfg_value=req.cfg_value,
            seed=req.seed,
        )
        return prep, params

    # ------------------------------------------------------------------ #
    # Step loop
    # ------------------------------------------------------------------ #
    def _admit(self) -> bool:
        """Prefill all admittable waiting requests as ONE pad-batched cohort."""
        if not (self.waiting and self.slots.num_free > 0):
            return False
        batch = []  # (rid, prepared, params)
        while self.waiting and self.slots.num_free > 0:
            rid, payload = self.waiting.popleft()
            slot = self.slots.acquire()
            if isinstance(payload, _PrebuiltInputs):
                prep, params = self._prepare_prebuilt(slot, payload)
            else:
                prep, params = self._prepare_request(slot, payload)
            batch.append((rid, prep, params))
        states = self.runner.prefill_batch([b[1] for b in batch])
        for (rid, _prep, params), st in zip(batch, states):
            self._apply_params(st, **params)
            self.running.append(_Running(request_id=rid, state=st))
        return True

    def _decode_running(self):
        """One batched decode step; returns ``[(run, patch[1, p, d])]`` this step."""
        active = [r for r in self.running if not r.state.finished]
        groups: Dict[tuple, List[_Running]] = defaultdict(list)
        for r in active:
            groups[(r.state.n_timesteps, r.state.cfg_value)].append(r)
        results = []
        for runs in groups.values():
            pred = self.runner.decode_step([r.state for r in runs])
            if pred is not None:
                for j, r in enumerate(runs):
                    results.append((r, pred[j : j + 1]))
        return results

    @torch.inference_mode()
    def step(self) -> List[RequestOutput]:
        # Admission takes priority while slots are free (prefill / decode split).
        if self._admit():
            return []
        if not self.running:
            return []
        self._decode_running()

        outputs: List[RequestOutput] = []
        still: List[_Running] = []
        for r in self.running:
            if r.state.finished:
                outputs.append(self._finalize(r))
                self.slots.release(r.state.slot)
            else:
                still.append(r)
        self.running = still
        return outputs

    @torch.inference_mode()
    def stream(self):
        """Generator yielding :class:`StreamChunk` per active request per step.

        Audio is streamed via one batched :class:`BatchedStreamingVAE` for rows
        with no continuation context (zero-shot / reference / speaker-centroid);
        prompt-audio continuation rows stream latents only (decode audio with
        :meth:`run`). When no AudioVAE is attached, all rows stream latents.
        """
        vae = getattr(self.model, "audio_vae", None)
        bvae = BatchedStreamingVAE(vae) if vae is not None else None
        sr = getattr(self.model, "sample_rate", None)
        try:
            while self.waiting or self.running:
                if self._admit():
                    continue
                if not self.running:
                    break
                results = self._decode_running()

                audio_map: Dict[int, torch.Tensor] = {}
                if bvae is not None:
                    audio_rows = [(r, p) for (r, p) in results if r.state.context_len == 0]
                    if audio_rows:
                        order = [r.request_id for r, _ in audio_rows]            # request_id: never reused
                        z = torch.cat([p for _, p in audio_rows], dim=0).permute(0, 2, 1).to(torch.float32)
                        audio = bvae.decode(order, z)                            # [M, 1, chunk]
                        for i, (r, _) in enumerate(audio_rows):
                            audio_map[r.request_id] = audio[i].squeeze(0).cpu()

                for r, patch in results:
                    yield StreamChunk(
                        request_id=r.request_id,
                        latents=patch.squeeze(0),
                        audio=audio_map.get(r.request_id),
                        finished=r.state.finished,
                        sample_rate=sr,
                    )

                for r in [r for r in self.running if r.state.finished]:
                    self.slots.release(r.state.slot)
                    self.running.remove(r)
        finally:
            if bvae is not None:
                bvae.close()

    def _finalize(self, r: _Running) -> RequestOutput:
        pred_feat_seq = self.runner.collect_latents(r.state)          # [context+T, p, d]
        context_len = r.state.context_len
        latents = pred_feat_seq[context_len:]
        out = RequestOutput(request_id=r.request_id, latents=latents)
        vae = getattr(self.model, "audio_vae", None)
        if vae is not None:
            feat_pred = pred_feat_seq.permute(2, 0, 1).reshape(self.model.config.feat_dim, -1)[None]
            decode_audio = vae.decode(feat_pred.to(torch.float32))
            patch_len = self.model.patch_size * self.model._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., patch_len * context_len :]
            out.audio = decode_audio.squeeze(1).squeeze(0).cpu()
            out.sample_rate = self.model.sample_rate
        self._outputs[r.request_id] = out
        return out

    @torch.inference_mode()
    def run(self) -> List[RequestOutput]:
        """Drain all waiting + running requests; return outputs in request-id order."""
        while self.waiting or self.running:
            self.step()
        return [self._outputs[i] for i in sorted(self._outputs)]


@dataclass
class _PrebuiltInputs:
    text_token: torch.Tensor
    audio_feat: torch.Tensor
    text_mask: torch.Tensor
    audio_mask: torch.Tensor
    spk_mask: Optional[torch.Tensor]
    speaker_centroids: Optional[torch.Tensor]
    min_len: int
    max_len: int
    inference_timesteps: Optional[int]
    cfg_value: Optional[float]
    seed: Optional[int]
