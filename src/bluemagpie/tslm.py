"""Barbet as the Text-Semantic LM (TSLM), with incremental decoding.

VoxCPM2's generation loop decodes one latent patch at a time, so the TSLM needs
a stateful single-step path. As of the open_formosa R2 architecture, Barbet
ships exactly that: a real **Mamba2** mixer and a native ``BarbetCache`` holding

- attention K/V states (rolling window for sliding-window layers),
- the trailing ``d_conv - 1`` causal-conv inputs per Mamba layer, and
- the Mamba2 selective-scan SSM state per Mamba layer.

We delegate stepwise decoding to it rather than re-deriving a cache by hand:
the selective-scan state cannot be reconstructed from a conv ring buffer alone,
so the official cache is the only correct option (and it stays in lockstep with
upstream). ``tests/test_step_equivalence.py`` asserts prefill + forward_step
reproduces the full-sequence forward across all layer types.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from barbet import BarbetConfig, BarbetModel
from barbet.modeling_barbet import BarbetCache


@dataclass
class BarbetStepState:
    """Decoding state: Barbet's native hybrid cache (tracks position itself)."""

    cache: BarbetCache

    @property
    def pos(self) -> int:
        return self.cache.seen_tokens


class BarbetTSLM(nn.Module):
    """Barbet backbone with embeddings, full forward, and cached stepwise decode."""

    def __init__(self, config: BarbetConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = BarbetModel(config)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-sequence forward (training / teacher-forcing), no cache."""
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    # ------------------------------------------------------------------ #
    # Stepwise decoding (delegated to BarbetCache)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prefill(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, BarbetStepState]:
        """Run the prompt and return (hidden_states, state) with a warm cache."""
        cache = BarbetCache(self.config)
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return out.last_hidden_state, BarbetStepState(cache=out.past_key_values or cache)

    @torch.no_grad()
    def forward_step(self, x: torch.Tensor, state: BarbetStepState) -> torch.Tensor:
        """Decode one position. ``x``: [B, H] input embedding -> [B, H] hidden.

        The cache tracks the running position (``seen_tokens``), so RoPE offsets
        and the sliding-window / conv / SSM states advance automatically.
        """
        out = self.backbone(
            inputs_embeds=x.unsqueeze(1),
            past_key_values=state.cache,
            use_cache=True,
            return_dict=True,
        )
        return out.last_hidden_state[:, 0, :]
