"""BlueMagpie-TTS serving engine — a vendored, device-agnostic, dependency-light
nano-vLLM-style continuous-batching engine. See DESIGN.md.

Importing this package must not require CUDA, so it loads on macOS/CPU and the
parity tests run locally.
"""

from .config import EngineConfig
from .engine import BlueMagpieEngine, Request, RequestOutput
from .streaming import StreamChunk

__all__ = ["EngineConfig", "BlueMagpieEngine", "Request", "RequestOutput", "StreamChunk"]
