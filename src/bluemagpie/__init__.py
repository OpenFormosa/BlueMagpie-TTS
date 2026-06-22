"""BlueMagpie-TTS: Barbet TSLM on top of VoxCPM2's acoustic stack."""

from .adapter import ProjectionAdapter
from .conditioning import SpeakerProjector
from .config import AdapterConfig, BlueMagpieConfig
from .loading import (
    build_config_from_voxcpm2,
    build_from_pretrained,
    load_voxcpm2_teacher,
    set_training_stage,
)
from .model import BlueMagpieModel
from .tslm import BarbetStepState, BarbetTSLM

__all__ = [
    "AdapterConfig",
    "BlueMagpieConfig",
    "BlueMagpieModel",
    "BarbetTSLM",
    "BarbetStepState",
    "ProjectionAdapter",
    "SpeakerProjector",
    "build_config_from_voxcpm2",
    "build_from_pretrained",
    "load_voxcpm2_teacher",
    "set_training_stage",
]
