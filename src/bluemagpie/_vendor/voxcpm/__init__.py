"""Vendored subset of VoxCPM (https://github.com/OpenBMB/VoxCPM), Apache-2.0.

Only the modules BlueMagpie-TTS needs at inference time are bundled here:
VoxCPM2's acoustic stack (AudioVAE / LocEnc / RALM / LocDiT) and its config /
runtime helpers. The original top-level ``voxcpm`` package eagerly imported the
full ``VoxCPM`` pipeline (``core``), which pulls in dependencies BlueMagpie-TTS
does not use, so this ``__init__`` is intentionally empty.

Modified from upstream: the original ``voxcpm/__init__.py`` imported
``from .core import VoxCPM``; that line was removed and replaced with this stub.

Import the concrete submodules directly, e.g.::

    from bluemagpie._vendor.voxcpm.modules.audiovae import AudioVAEV2

See ``LICENSE`` and ``PROVENANCE.md`` in this directory for details.
"""
