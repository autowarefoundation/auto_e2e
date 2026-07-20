"""Validation for reasoning label artifacts (issue #98, R4/R5/R9).

A record is only usable for training if it is structurally sound: exactly five
horizons in order, labels drawn only from the taxonomy, single-label groups
holding at most one label. Abstained records are exempt from the horizon/label
checks (they carry no horizons by construction) but must be marked as such.

These checks run offline before an artifact is written, so a malformed teacher
response is caught at generation time rather than silently poisoning training.
"""

from __future__ import annotations

from typing import List

from model_components.reasoning.reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    LabelMode,
    ReasoningTaxonomy,
)

from .schema import HORIZON_SECONDS, NUM_HORIZONS, ReasoningLabelRecord

# The (group_name, attribute_name, is_optional) triples the validator checks.
# Required action-relevant groups map to non-None attributes; optional v2
# context groups are only checked when present.
_REQUIRED_MULTI = ("hazard_event", "cause")
_REQUIRED_SINGLE = (
    "relation_to_ego",
    "longitudinal_response",
    "lateral_response",
    "tactical_response",
    "rule_response",
)


class LabelValidationError(ValueError):
    """Raised when a label record violates the schema/taxonomy contract."""


def validate_record(
    record: ReasoningLabelRecord, taxonomy: ReasoningTaxonomy = DEFAULT_TAXONOMY
) -> None:
    """Validate one record against the schema and taxonomy.

    Abstained records only need the abstain marking; successful records must
    have exactly five ordered horizons with in-taxonomy labels and at most one
    label per single-label group.

    Raises:
        LabelValidationError: on any violation.
    """
    if record.abstained:
        if record.teacher_error is None:
            raise LabelValidationError(
                f"{record.sample_id}: abstained record must carry teacher_error."
            )
        return  # abstained records carry no horizons by construction

    horizons = record.horizons
    if len(horizons) != NUM_HORIZONS:
        raise LabelValidationError(
            f"{record.sample_id}: expected {NUM_HORIZONS} horizons, got {len(horizons)}."
        )
    got_secs = [h.horizon_sec for h in horizons]
    if got_secs != list(HORIZON_SECONDS):
        raise LabelValidationError(
            f"{record.sample_id}: horizons must be {list(HORIZON_SECONDS)} in order, "
            f"got {got_secs} (missing / duplicated / unordered)."
        )

    for h_idx, h in enumerate(horizons):
        for group in _REQUIRED_MULTI:
            _check_multi(record, h_idx, group, getattr(h, group), taxonomy)
        for group in _REQUIRED_SINGLE:
            _check_single(record, h_idx, group, getattr(h, group), taxonomy)

        # Optional v2 multi-label context groups: check only when labelled.
        for group in taxonomy.group_names():
            if group in _REQUIRED_MULTI or group in _REQUIRED_SINGLE:
                continue
            value = getattr(h, group, None)
            if value is None:
                continue
            if taxonomy.mode(group) is LabelMode.MULTI:
                _check_multi(record, h_idx, group, value, taxonomy)
            else:
                _check_single(record, h_idx, group, value, taxonomy)

        if not (0.0 <= h.confidence <= 1.0):
            raise LabelValidationError(
                f"{record.sample_id} h{h_idx}: confidence {h.confidence} not in [0, 1]."
            )


def _check_multi(
    record: ReasoningLabelRecord, h_idx: int, group: str,
    values: List[str], taxonomy: ReasoningTaxonomy,
) -> None:
    allowed = set(taxonomy.labels(group))
    for v in values:
        if v not in allowed:
            raise LabelValidationError(
                f"{record.sample_id} h{h_idx}: unknown label '{v}' for multi-label "
                f"group '{group}'."
            )


def _check_single(
    record: ReasoningLabelRecord, h_idx: int, group: str,
    value, taxonomy: ReasoningTaxonomy,
) -> None:
    if value is None:
        return  # single-label abstain (masked out with ignore_index at collate)
    if not isinstance(value, str):
        raise LabelValidationError(
            f"{record.sample_id} h{h_idx}: single-label group '{group}' expects one "
            f"string or None, got {type(value).__name__}."
        )
    if value not in set(taxonomy.labels(group)):
        raise LabelValidationError(
            f"{record.sample_id} h{h_idx}: unknown label '{value}' for single-label "
            f"group '{group}'."
        )


def validate_records(
    records: List[ReasoningLabelRecord], taxonomy: ReasoningTaxonomy = DEFAULT_TAXONOMY
) -> None:
    """Validate a batch of records; raises on the first violation."""
    for record in records:
        validate_record(record, taxonomy)
