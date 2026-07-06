"""Teacher backends for reasoning-band pseudo-label generation (issue #98).

Teachers are TRAIN-ONLY offline autolabellers — they are never part of the
inference loop.  For CI and unit tests use :class:`DeterministicTeacher`
(no GPU, no network); real VLM backends (Qwen2-VL, VideoLLaMA3) plug in as
follow-up work through the same registry.

Extension point: to add a new teacher backend (e.g. an Alpamayo CoC
autolabeller), subclass :class:`VLMTeacher` from ``base.py`` and register the
class in ``_TEACHER_REGISTRY`` below.

    from model_components.reasoning.teachers.base import VLMTeacher

    class MyTeacher(VLMTeacher):
        ...

    # Optional: register so consumers can look it up by name.
    _TEACHER_REGISTRY["my_teacher"] = MyTeacher
"""

from .base import VLMTeacher
from .deterministic import DeterministicTeacher

__all__ = ["VLMTeacher", "DeterministicTeacher"]

# Registry: maps a string key to a teacher class.  Populated lazily so that
# importing this package never requires heavy dependencies.
_TEACHER_REGISTRY: dict[str, type[VLMTeacher]] = {
    "deterministic": DeterministicTeacher,
}
