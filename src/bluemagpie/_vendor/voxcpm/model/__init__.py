"""VoxCPM2 model package (vendored subset).

The upstream ``voxcpm.model`` package also exposed the v1 ``VoxCPMModel``;
BlueMagpie-TTS only uses VoxCPM2, so this ``__init__`` is empty and callers
import the concrete submodules directly.

Modified from upstream: the original ``model/__init__.py`` imported
``from .voxcpm import VoxCPMModel`` and ``from .voxcpm2 import VoxCPM2Model``;
those lines were removed and replaced with this stub (the v1 ``voxcpm.py`` is
not vendored). Example::

    from bluemagpie._vendor.voxcpm.model.voxcpm2 import VoxCPMConfig
    from bluemagpie._vendor.voxcpm.model.utils import get_dtype
"""
