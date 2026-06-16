from .base import BaseTemporalMemory
from .no_memory import NoMemory
from .one_hz_encoder import OneHzHistoryEncoder

TEMPORAL_MEMORY_REGISTRY = {
    "no_memory": NoMemory,
    "one_hz": OneHzHistoryEncoder,
}

def build_temporal_memory(memory_mode: str, **kwargs):
    if memory_mode not in TEMPORAL_MEMORY_REGISTRY:
        raise ValueError(
            f"Unknown temporal memory mode {memory_mode!r}. "
            f"Available: {sorted(TEMPORAL_MEMORY_REGISTRY)}."
        )
    return TEMPORAL_MEMORY_REGISTRY[memory_mode](**kwargs)

__all__ = [
    "BaseTemporalMemory",
    "NoMemory",
    "OneHzHistoryEncoder",
    "TEMPORAL_MEMORY_REGISTRY",
    "build_temporal_memory"
]
