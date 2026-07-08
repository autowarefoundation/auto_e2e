"""Tests for the reasoning label schema + validators (issue #98, R4/R5/R9).

Pure-Python, no torch/GPU/network. Covers:
    * a well-formed 5-horizon record validates;
    * missing / duplicated / unordered horizons fail;
    * unknown labels (multi and single) fail;
    * a single-label group given a list fails;
    * abstained records validate (and require teacher_error), carrying no horizons.
"""

from __future__ import annotations

import pytest

from data_processing.reasoning_label_generation.schema import (
    HORIZON_SECONDS,
    SCHEMA_VERSION,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)
from data_processing.reasoning_label_generation.validators import (
    LabelValidationError,
    validate_record,
)


def _horizon(sec: float, **overrides) -> ReasoningHorizonLabel:
    base = dict(
        horizon_sec=sec,
        relation_to_ego="same_lane_ahead",
        hazard_event=["no_hazard"],
        cause=["lead_vehicle"],
        longitudinal_response="follow_lead_vehicle",
        lateral_response="keep_lane",
        tactical_response="proceed",
        rule_response="none",
        confidence=0.8,
        provenance="teacher_gt",
    )
    base.update(overrides)
    return ReasoningHorizonLabel(**base)


def _record(horizons) -> ReasoningLabelRecord:
    return ReasoningLabelRecord(
        schema_version=SCHEMA_VERSION,
        sample_id="scene_000123_t0",
        timestamp=0.0,
        dataset_name="l2d",
        teacher_provider="mock",
        teacher_model="mock",
        prompt_version="action_relevant_reasoning_v2",
        request_mode="clip_horizons",
        horizons=horizons,
    )


def test_wellformed_record_validates():
    rec = _record([_horizon(s) for s in HORIZON_SECONDS])
    validate_record(rec)  # no raise


def test_missing_horizon_fails():
    rec = _record([_horizon(s) for s in HORIZON_SECONDS[:-1]])
    with pytest.raises(LabelValidationError, match="expected 5 horizons"):
        validate_record(rec)


def test_unordered_horizons_fail():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[1], horizons[2] = horizons[2], horizons[1]
    with pytest.raises(LabelValidationError, match="in order"):
        validate_record(_record(horizons))


def test_duplicated_horizon_fails():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[2] = _horizon(1.0)  # duplicate the 1s horizon
    with pytest.raises(LabelValidationError, match="in order"):
        validate_record(_record(horizons))


def test_unknown_multilabel_fails():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[0] = _horizon(0.0, cause=["not_a_real_cause"])
    with pytest.raises(LabelValidationError, match="unknown label"):
        validate_record(_record(horizons))


def test_unknown_singlelabel_fails():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[0] = _horizon(0.0, longitudinal_response="teleport")
    with pytest.raises(LabelValidationError, match="unknown label"):
        validate_record(_record(horizons))


def test_singlelabel_given_list_fails():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[0] = _horizon(0.0, relation_to_ego=["same_lane_ahead"])
    with pytest.raises(LabelValidationError, match="expects one string"):
        validate_record(_record(horizons))


def test_singlelabel_none_is_allowed_abstain():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[0] = _horizon(0.0, rule_response=None)
    validate_record(_record(horizons))  # None = masked-out single-label, no raise


def test_confidence_out_of_range_fails():
    horizons = [_horizon(s) for s in HORIZON_SECONDS]
    horizons[0] = _horizon(0.0, confidence=1.5)
    with pytest.raises(LabelValidationError, match="confidence"):
        validate_record(_record(horizons))


def test_abstained_record_validates():
    rec = ReasoningLabelRecord.abstain(
        sample_id="scene_x_t0",
        dataset_name="l2d",
        teacher_provider="openai_compatible",
        teacher_model="cosmos3-nano",
        prompt_version="action_relevant_reasoning_v2",
        request_mode="clip_horizons",
        teacher_error="HTTP 503 from endpoint",
    )
    assert rec.abstained is True
    assert rec.horizons == []
    validate_record(rec)  # no raise


def test_abstained_without_error_fails():
    rec = ReasoningLabelRecord.abstain(
        sample_id="x", dataset_name="l2d", teacher_provider="mock",
        teacher_model="mock", prompt_version="v2", request_mode="clip_horizons",
        teacher_error="tmp",
    )
    rec.teacher_error = None
    with pytest.raises(LabelValidationError, match="teacher_error"):
        validate_record(rec)
