#!/usr/bin/env python3
"""Debug fused mamba shapes."""
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
    pre = "layers.0.mixer."
    keys = [k for k in m.barbet.p if k.startswith("layers.0.mixer.")]
    for k in sorted(keys):
        print(k, m.barbet.p[k].shape)
    print("hidden:", m.barbet.cfg.hidden_size, "inner:", m.barbet.inner,
          "m_groups*d_state:", m.barbet.m_groups * m.barbet.d_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
