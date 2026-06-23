"""Apple-Silicon MLX acceleration for BlueMagpie-TTS (optional).

The full inference path is re-implemented with MLX (`mlx.core` / `mlx.nn`) so it
runs natively on the Apple-Silicon GPU (Metal, unified memory). This is an
**optional extra** — the core package stays torch-only; install with
`pip install -e .[mlx]`.

Usage::

    from bluemagpie import BlueMagpieModel
    from bluemagpie.mlx import BlueMagpieMLX, mlx_generate

    model = BlueMagpieModel.from_local(model_dir, tokenizer=tok, device="cpu")
    mlx_model = BlueMagpieMLX(model)                 # converts weights once
    audio = mlx_generate(model, mlx_model, "今天天氣真好。", seed=0)  # 48 kHz waveform

Every MLX module is checked for numerical parity against its PyTorch reference on
the tiny test configs (`tests/test_mlx_parity.py`), so the port is verifiable on
the dev's own Mac. See `DESIGN.md`.
"""

from .convert import to_mx, torch_params_to_mx
from .model_mlx import BlueMagpieMLX, mlx_generate

__all__ = ["to_mx", "torch_params_to_mx", "BlueMagpieMLX", "mlx_generate"]
