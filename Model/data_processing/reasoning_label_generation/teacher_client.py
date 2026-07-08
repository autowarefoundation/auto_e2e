"""Abstract, model-agnostic teacher client (issue #98, R1/R2).

A teacher client turns a driving sample (clip frames + optional ego/route/map
context) into a :class:`ReasoningLabelRecord` — five horizons of action-relevant
labels plus provenance. Every backend (mock / cached / OpenAI-compatible /
rule-based) shares this one interface, so the offline pipeline swaps backends
with no code change and AutoE2E stays backend-agnostic (depends only on
``provider / base_url / model / prompt_version / schema_version / request_mode``).

TRAIN-ONLY / OFFLINE: teachers run during preprocessing (a Flyte task, a local
script, a batch job) and are NEVER instantiated in the model forward pass or the
vehicle. They live under ``data_processing`` precisely so ``model_components``
never imports them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from model_components.reasoning.reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    ReasoningTaxonomy,
)

from .schema import ReasoningLabelRecord


class TeacherRequest:
    """One sample's inputs for the teacher (offline).

    Attributes:
        sample_id: unique id (e.g. ``"scene_000123_t0"``).
        dataset_name: source dataset (e.g. ``"l2d"``).
        frames: ordered clip frames — one per horizon (current + 4 future) for
            ``clip_horizons``, or a single current frame for ``per_frame``.
            Each is a ``[3, H, W]`` tensor. Backends that do not consume pixels
            (mock / rule_based) may ignore them.
        timestamp: sample timestamp (seconds).
        extra_context: optional ego/route/map text folded into the prompt.
        dataset_version: optional dataset version for provenance.
    """

    def __init__(
        self,
        sample_id: str,
        dataset_name: str,
        frames: Sequence[Any] = (),
        timestamp: float = 0.0,
        extra_context: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> None:
        self.sample_id = sample_id
        self.dataset_name = dataset_name
        self.frames = list(frames)
        self.timestamp = timestamp
        self.extra_context = extra_context
        self.dataset_version = dataset_version


class TeacherClient(ABC):
    """Base class for offline reasoning-label teachers.

    Args:
        provider: backend label recorded in provenance (e.g. ``"mock"``,
            ``"openai_compatible"``).
        model: model name recorded in provenance (e.g. ``"cosmos3-nano"``).
        prompt_version: prompt version for provenance / reproducibility.
        request_mode: ``"per_frame"`` or ``"clip_horizons"``.
        taxonomy: label registry (defaults to :data:`DEFAULT_TAXONOMY`).
        strict: if True (default) a backend raises on endpoint/parse/schema
            failure; if False it returns an abstained record (R9).
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        prompt_version: str = "action_relevant_reasoning_v2",
        request_mode: str = "clip_horizons",
        taxonomy: Optional[ReasoningTaxonomy] = None,
        strict: bool = True,
    ) -> None:
        if request_mode not in ("per_frame", "clip_horizons"):
            raise ValueError(
                f"request_mode must be 'per_frame' or 'clip_horizons', got {request_mode!r}."
            )
        self.provider = provider
        self.model = model
        self.prompt_version = prompt_version
        self.request_mode = request_mode
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        self.strict = strict

    @abstractmethod
    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        """Produce one sample's :class:`ReasoningLabelRecord` (5 horizons)."""
        raise NotImplementedError

    def label_batch(
        self, requests: Sequence[TeacherRequest]
    ) -> List[ReasoningLabelRecord]:
        """Label a batch of samples (default: one call per request)."""
        return [self.label(r) for r in requests]

    def _abstain(self, request: TeacherRequest, error: str) -> ReasoningLabelRecord:
        """Build an abstained record for *request* (R9)."""
        return ReasoningLabelRecord.abstain(
            sample_id=request.sample_id,
            dataset_name=request.dataset_name,
            teacher_provider=self.provider,
            teacher_model=self.model,
            prompt_version=self.prompt_version,
            request_mode=self.request_mode,
            teacher_error=error,
            timestamp=request.timestamp,
        )


# ``build_teacher`` registry — string key -> factory. Populated by the backends
# so a config string selects a backend without importing every module eagerly.
_TEACHER_FACTORIES: Dict[str, Any] = {}


def register_teacher(name: str, factory: Any) -> None:
    """Register a teacher backend factory under *name*."""
    _TEACHER_FACTORIES[name] = factory


def build_teacher(provider: str, **kwargs: Any) -> TeacherClient:
    """Construct a teacher backend by provider name.

    Backends register lazily on import; this imports the known modules so the
    registry is populated before lookup.
    """
    from . import cached_teacher, mock_teacher, openai_compatible  # noqa: F401

    if provider not in _TEACHER_FACTORIES:
        raise ValueError(
            f"Unknown teacher provider {provider!r}. "
            f"Available: {sorted(_TEACHER_FACTORIES)}."
        )
    return _TEACHER_FACTORIES[provider](**kwargs)
