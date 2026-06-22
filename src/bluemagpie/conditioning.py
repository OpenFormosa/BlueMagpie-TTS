"""Speaker conditioning for BlueMagpie-TTS (v1).

v1 conditions on a *speaker centroid* — a per-speaker ECAPA-TDNN embedding
averaged over several clips, computed offline. The centroid carries the constant part
of a voice (identity, timbre, baseline pitch); using a denoised per-speaker
average rather than a per-utterance embedding prevents the model from leaking
that clip's prosody/content as a shortcut.

``SpeakerProjector`` maps the centroid into the Barbet hidden space; the
resulting vector is placed at the ``[spk]`` token slot in the input sequence
(see ``BlueMagpieModel``), so Barbet's causal attention propagates the speaker
identity into every downstream position.
"""

from __future__ import annotations

import torch
from torch import nn

from .adapter import RMSNorm


class SpeakerProjector(nn.Module):
    """Project an L2-normalized speaker centroid into the Barbet hidden space."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(in_dim, eps)
        self.proj = nn.Linear(in_dim, out_dim)
        # Small init so a cold-started speaker vector does not destabilize the
        # warm-started (pretrained) Barbet backbone early in training.
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, centroid: torch.Tensor) -> torch.Tensor:
        """``centroid``: [B, in_dim] -> [B, out_dim]."""
        return self.proj(self.norm(centroid))
