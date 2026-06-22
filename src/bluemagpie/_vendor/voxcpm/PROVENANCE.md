# Vendored VoxCPM (inference subset)

This directory contains a **partial copy** of [VoxCPM](https://github.com/OpenBMB/VoxCPM),
bundled into BlueMagpie-TTS so the package has no external git dependency on
`voxcpm`.

- **Source:** https://github.com/OpenBMB/VoxCPM.git
- **Commit:** `856d2fc2a853656e324e491706d1e8a6bfac361c`
- **License:** Apache-2.0 (see `LICENSE` in this directory), Copyright OpenBMB.

## What was copied

Only the modules BlueMagpie-TTS uses at inference time:

- `model/utils.py` — runtime helpers (`get_dtype`, `next_and_close`,
  `pick_runtime_dtype`, `resolve_runtime_device`).
- `model/voxcpm2.py` — VoxCPM2 config classes (`VoxCPMConfig`,
  `VoxCPMDitConfig`, `VoxCPMEncoderConfig`) and the `VoxCPM2Model` reference
  implementation.
- `modules/audiovae/`, `modules/layers/`, `modules/locdit/`, `modules/locenc/`,
  `modules/minicpm4/` — the AudioVAE / LocEnc / RALM / LocDiT acoustic stack.

## What was changed

- The package `__init__.py` files (`voxcpm/__init__.py`, `voxcpm/model/__init__.py`)
  were replaced with empty stubs so importing this subset does **not** pull in
  the upstream `core` pipeline or the v1 `VoxCPMModel`.
- Nothing else was modified; all module source is upstream-verbatim, and all
  internal imports are relative, so the subtree is self-contained.

## Updating

To refresh against a newer VoxCPM release, re-copy the files listed above from
the upstream commit, keep the two stub `__init__.py` files, and re-run the test
suite.
