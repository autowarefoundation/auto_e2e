"""Teacher backends for reasoning-band pseudo-label generation (issue #98).

Teachers are TRAIN-ONLY offline autolabellers — they are never part of the
inference loop.  The default real backend is :class:`Qwen2VLTeacher`; for CI
and unit tests use :class:`DeterministicTeacher` (no GPU, no network).

Extension point: to add a new teacher backend (e.g. an Alpamayo CoC
autolabeller for v2), subclass :class:`VLMTeacher` from ``base.py`` and
register the class in ``_TEACHER_REGISTRY`` below.

    from model_components.reasoning.teachers.base import VLMTeacher

    class MyTeacher(VLMTeacher):
        ...

    # Optional: register so consumers can look it up by name.
    _TEACHER_REGISTRY["my_teacher"] = MyTeacher
"""

from .base import VLMTeacher
from .deterministic import DeterministicTeacher
from .multi_teacher import MultiTeacher

__all__ = [
    "VLMTeacher",
    "DeterministicTeacher",
    "MultiTeacher",
]

# Registry: maps a string key to a teacher class.  Populated lazily so that
# importing this package never requires heavy dependencies.
_TEACHER_REGISTRY: dict[str, type[VLMTeacher]] = {
    "deterministic": DeterministicTeacher,
    "multi": MultiTeacher,
}


def _register_lazy_backends() -> None:
    """Register the heavy (lazily-imported) backends by name."""
    from .openai_endpoint import OpenAIEndpointTeacher
    from .qwen2vl import Qwen2VLTeacher
    from .videollama3 import VideoLlama3Teacher

    _TEACHER_REGISTRY.setdefault("qwen2vl", Qwen2VLTeacher)
    _TEACHER_REGISTRY.setdefault("videollama3", VideoLlama3Teacher)
    _TEACHER_REGISTRY.setdefault("openai_endpoint", OpenAIEndpointTeacher)


_register_lazy_backends()
