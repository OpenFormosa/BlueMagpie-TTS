#!/usr/bin/env python3
"""Print mamba state shape facts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bluemagpie.mlx.lite import load_model

MD = os.path.join(
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


def main():
    lite, m = load_model(MD, "bluemagpie-int4.pkl", compute="int4-cached")
    bar = m.barbet
    print("hd:", bar.hd, "d_state:", bar.d_state, "m_heads:", bar.m_heads,
          "inner:", bar.inner, "group_size:", bar.group_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
