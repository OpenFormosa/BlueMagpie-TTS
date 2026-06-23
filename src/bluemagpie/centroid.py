"""Extract a speaker centroid from reference audio, for voice cloning.

Computes an ECAPA-TDNN centroid in the same space as the model's training
centroids — the same mechanism the bundled ``hung_yi_lee`` example uses — ready
to pass as ``model.generate(..., speaker_centroid=centroid)``.

Requires ``speechbrain`` (an optional extra): ``pip install -e ".[clone]"``.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

DEFAULT_ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
_SR = 16000


def _load_encoder(model_id: str, device: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:  # pragma: no cover - heavy optional dep
        raise ImportError(
            f"speechbrain is required to extract speaker centroids ({e}). "
            'Install with: pip install -e ".[clone]"  (or: pip install speechbrain)'
        ) from e
    return EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})


@torch.no_grad()
def extract_speaker_centroid(
    audio: Union[str, Sequence[str]],
    *,
    ecapa_model: str = DEFAULT_ECAPA,
    device: str = "cpu",
    window_s: float = 6.0,
    encoder=None,
) -> torch.Tensor:
    """One or more reference clips (same speaker) -> a ``[192]`` L2-normalized centroid.

    Matches ``scripts/build_speaker_centroids.py``: each ~``window_s`` chunk is
    ECAPA-embedded and L2-normalized, all chunks are averaged, and the mean is
    L2-normalized — so the result lives in the model's training-centroid space.
    Pass a single path or a list of paths. Provide a loaded ``encoder`` to reuse it.
    """
    import librosa

    paths = [audio] if isinstance(audio, str) else list(audio)
    clf = encoder if encoder is not None else _load_encoder(ecapa_model, device)
    win = int(window_s * _SR)
    embs = []
    for path in paths:
        wav, _ = librosa.load(path, sr=_SR, mono=True)
        x = torch.from_numpy(np.ascontiguousarray(wav)).float()
        chunks = [x[i : i + win] for i in range(0, x.numel(), win)]
        chunks = [c for c in chunks if c.numel() >= _SR]
        if not chunks and x.numel() > 0:
            chunks = [x]  # shorter than 1 s: use as-is
        for c in chunks:
            e = clf.encode_batch(c.unsqueeze(0).to(device)).reshape(-1)  # [192]
            embs.append(torch.nn.functional.normalize(e, dim=0).cpu())
    if not embs:
        raise ValueError("no usable audio; provide ~3-10 s of clean single-speaker speech")
    mean = torch.stack(embs).mean(0)
    return torch.nn.functional.normalize(mean, dim=0)
