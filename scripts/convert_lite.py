#!/usr/bin/env python3
"""One-time conversion of the BlueMagpie checkpoint to the fp16 MLX lite format.

Usage:
    python scripts/convert_lite.py [MODEL_DIR] [OUT_PATH]

Defaults to the HF-cache model dir and bluemagpie-lite.pkl in the current dir.
Peak RAM ~12 GB (fp32 torch model + fp16 MLX copy); run with nothing heavy
open. The lite checkpoint (~4 GB fp16) replaces the 7.75 GB torch model for
every later launch — see bluemagpie/mlx/lite.py.
"""
import os
import sys
import time

MODEL_DIR = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS"),
    "snapshots",
    max(
        os.listdir(os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots")),
        key=lambda d: os.path.getmtime(
            os.path.join(
                os.path.expanduser("~/.cache/huggingface/hub/models--OpenFormosa--BlueMagpie-TTS/snapshots"), d
            )
        ),
    ),
)


def main() -> int:
    args = sys.argv[1:]
    out_path = "bluemagpie-lite.pkl"
    precision = "fp32"
    if args and args[0] == "--out":
        out_path = args[1]
        args = args[2:]
    if args and args[0] == "--precision":
        precision = args[1]
        args = args[2:]
    model_dir = args[0] if args else MODEL_DIR
    if not os.path.isdir(model_dir):
        print(f"model dir not found: {model_dir}", file=sys.stderr)
        return 1
    from bluemagpie.mlx.lite import convert_checkpoint

    t0 = time.time()
    convert_checkpoint(model_dir, out_path, precision=precision)
    print(f"total: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())