"""Deterministic mock teacher (issue #98, R3).

Produces valid, reproducible :class:`ReasoningLabelRecord`s WITHOUT any model,
network, or GPU — so CI and contributors without a cluster can exercise the
whole offline pipeline (label → validate → write → train) locally.

Labels are a deterministic function of the sample_id (hashed), so the same
sample always gets the same labels across runs — useful for tests and for a
cheap smoke of the training path before a real teacher is wired in.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from model_components.reasoning.reasoning_taxonomy import LabelMode, ReasoningTaxonomy

from .schema import (
    HORIZON_SECONDS,
    SCHEMA_VERSION,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from .teacher_client import TeacherClient, TeacherRequest, register_teacher

_CORE_GROUPS = (
    "relation_to_ego", "hazard_event", "cause",
    "longitudinal_response", "lateral_response", "tactical_response", "rule_response",
)


def _seed(sample_id: str, horizon_idx: int) -> int:
    """Stable integer seed from (sample_id, horizon) via a hash (no RNG state)."""
    h = hashlib.sha256(f"{sample_id}:{horizon_idx}".encode()).hexdigest()
    return int(h[:8], 16)


class MockTeacher(TeacherClient):
    """Deterministic offline teacher (no model / network / GPU)."""

    def __init__(
        self,
        *,
        model: str = "mock",
        prompt_version: str = "action_relevant_reasoning_v2",
        request_mode: str = "clip_horizons",
        taxonomy: Optional[ReasoningTaxonomy] = None,
        strict: bool = True,
    ) -> None:
        super().__init__(
            provider="mock", model=model, prompt_version=prompt_version,
            request_mode=request_mode, taxonomy=taxonomy, strict=strict,
        )

    def label(self, request: TeacherRequest) -> ReasoningLabelRecord:
        horizons = []
        for i, sec in enumerate(HORIZON_SECONDS):
            seed = _seed(request.sample_id, i)
            kwargs: dict[str, Any] = {"horizon_sec": sec, "provenance": "teacher_gt"}
            for gi, group in enumerate(_CORE_GROUPS):
                labels = self.taxonomy.labels(group)
                pick = labels[(seed + gi) % len(labels)]
                if self.taxonomy[group].mode is LabelMode.MULTI:
                    kwargs[group] = [pick]
                else:
                    kwargs[group] = pick
            # Deterministic confidence in [0.5, 1.0].
            kwargs["confidence"] = 0.5 + (seed % 500) / 1000.0
            kwargs["evidence"] = f"mock label for {request.sample_id} h{i}"
            horizons.append(ReasoningHorizonLabel(**kwargs))

        return ReasoningLabelRecord(
            schema_version=SCHEMA_VERSION,
            sample_id=request.sample_id,
            timestamp=request.timestamp,
            dataset_name=request.dataset_name,
            teacher_provider=self.provider,
            teacher_model=self.model,
            prompt_version=self.prompt_version,
            request_mode=self.request_mode,
            horizons=horizons,
            dataset_version=request.dataset_version,
            provenance="teacher_gt",
        )


register_teacher("mock", MockTeacher)
