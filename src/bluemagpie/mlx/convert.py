"""PyTorch <-> MLX weight conversion helpers.

The MLX modules re-implement the forward pass but consume the *same* trained
weights as the PyTorch model. We keep weights as a flat ``{name: mx.array}``
dict keyed exactly by the torch ``named_parameters()`` name, so each MLX module
just looks up the tensors it needs — no name remapping or per-module loaders.

Layout notes:
- ``nn.Linear`` weight is ``[out, in]`` in both frameworks (both compute
  ``x @ W.T``), so it transfers as-is; the MLX modules do the matmul manually.
- ``nn.Conv1d`` weight is ``[out, in/groups, kernel]`` in torch; the MLX modules
  here implement the (depthwise, causal) conv manually from that exact layout,
  so the weight also transfers as-is.
"""

from __future__ import annotations

from typing import Dict

import mlx.core as mx


def to_mx(t) -> mx.array:
    """torch.Tensor -> mx.array (float32, on the default MLX device)."""
    return mx.array(t.detach().to("cpu", dtype=__import__("torch").float32).numpy())


def torch_params_to_mx(module) -> Dict[str, mx.array]:
    """Flat ``{param_name: mx.array}`` from a torch module's ``named_parameters``."""
    return {name: to_mx(p) for name, p in module.named_parameters()}
