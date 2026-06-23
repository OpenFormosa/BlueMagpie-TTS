"""BlueMagpie-TTS: Barbet TSLM on top of VoxCPM2's acoustic stack."""

from .adapter import ProjectionAdapter
from .centroid import extract_speaker_centroid
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
    "extract_speaker_centroid",
    "build_config_from_voxcpm2",
    "build_from_pretrained",
    "load_voxcpm2_teacher",
    "set_training_stage",
]
