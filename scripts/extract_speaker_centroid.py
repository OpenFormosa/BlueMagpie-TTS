#!/usr/bin/env python3
"""Extract a speaker centroid from reference audio, for voice cloning.

    python scripts/extract_speaker_centroid.py --audio reference.wav --out my_voice.pt

Then synthesize in that voice:

    import torch
    from bluemagpie import BlueMagpieModel
    centroid = torch.load("my_voice.pt", weights_only=True)
    audio = model.generate(target_text="今天天氣真好。", speaker_centroid=centroid, cfg_value=2.0)

This uses the ``speaker_centroid`` voice-clone path (the same mechanism as the
bundled hung_yi_lee example).

Requires speechbrain:  pip install -e ".[clone]"
Tip: pass several clips of the same speaker (``--audio a.wav b.wav c.wav``) for a
cleaner, more robust centroid.
"""

import argparse

import torch

from bluemagpie.centroid import DEFAULT_ECAPA, extract_speaker_centroid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", nargs="+", required=True, help="one or more reference wav files (same speaker)")
    p.add_argument("--out", required=True, help="output .pt file (a [192] tensor)")
    p.add_argument("--ecapa-model", default=DEFAULT_ECAPA)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--window-s", type=float, default=6.0, help="chunk length in seconds for averaging")
    args = p.parse_args()

    centroid = extract_speaker_centroid(
        args.audio, ecapa_model=args.ecapa_model, device=args.device, window_s=args.window_s
    )
    torch.save(centroid, args.out)
    print(f"saved speaker_centroid (dim={centroid.numel()}, norm={centroid.norm():.3f}) to {args.out}")
    print(f'use: model.generate(target_text=..., speaker_centroid=torch.load("{args.out}", weights_only=True), cfg_value=2.0)')


if __name__ == "__main__":
    main()
