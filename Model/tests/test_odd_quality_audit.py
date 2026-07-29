from __future__ import annotations

import copy
import json

import pytest

from data_processing.odd_labeling.quality_audit import (
    ANNOTATION_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    evaluate_human_audit_annotations,
    validate_human_audit_annotations,
)


def _ontology() -> dict:
    return {
        "sources": [
            "map_route",
            "gnss_ins",
            "vlm",
            "image_qc",
            "fusion",
            "can_optional",
        ],
        "labels": [
            {
                "key": "odd.road.context",
                "namespace": "odd",
                "cardinality": "single",
                "values": [
                    {"value": "urban"},
                    {"value": "rural"},
                ],
            },
            {
                "key": "event.ego.maneuver",
                "namespace": "event",
                "cardinality": "single",
                "values": [
                    {"value": "turn_left"},
                    {"value": "turn_right"},
                ],
            },
        ],
    }


def _label_unit(
    index: int,
    *,
    predicted_status: str,
    predicted_values: list[str],
    reference_status: str,
    reference_values: list[str],
    confidence: float,
    weight: float,
    agreement: str = "unanimous",
) -> dict:
    start = index * 1_000_000_000
    return {
        "unit_uid": f"label-unit-{index}",
        "scene_uid": "scene-1",
        "label_key": "odd.road.context",
        "source": "map_route",
        "start_timestamp_ns": start,
        "end_timestamp_ns": start + 1_000_000_000,
        "sampling_weight": weight,
        "agreement": agreement,
        "prediction": {
            "status": predicted_status,
            "values": predicted_values,
            "confidence": confidence,
            "evidence_uid": (
                f"evidence-{index}"
                if predicted_status == "valid"
                else None
            ),
        },
        "reference": {
            "status": reference_status,
            "values": reference_values,
        },
    }


def _event_unit(
    index: int,
    *,
    match_status: str,
    predicted_interval: tuple[int, int] | None,
    reference_interval: tuple[int, int] | None,
    actor_continuity: str,
    agreement: str = "unanimous",
    weight: float = 1.0,
) -> dict:
    return {
        "unit_uid": f"event-unit-{index}",
        "scene_uid": "scene-1",
        "primary_event_key": "event.ego.maneuver",
        "source": "gnss_ins",
        "sampling_weight": weight,
        "agreement": agreement,
        "match_status": match_status,
        "predicted_event_uid": (
            f"predicted-event-{index}"
            if predicted_interval is not None
            else None
        ),
        "reference_event_uid": (
            f"reference-event-{index}"
            if reference_interval is not None
            else None
        ),
        "predicted_start_timestamp_ns": (
            predicted_interval[0] if predicted_interval is not None else None
        ),
        "predicted_end_timestamp_ns": (
            predicted_interval[1] if predicted_interval is not None else None
        ),
        "reference_start_timestamp_ns": (
            reference_interval[0] if reference_interval is not None else None
        ),
        "reference_end_timestamp_ns": (
            reference_interval[1] if reference_interval is not None else None
        ),
        "actor_continuity": actor_continuity,
    }


def _annotations() -> dict:
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "labelset_id": "oddls-test",
        "annotation_set_id": "a" * 64,
        "audit_manifest_sha256": "b" * 64,
        "adjudication_status": "adjudicated",
        "reviewer_count": 2,
        "label_units": [
            _label_unit(
                1,
                predicted_status="valid",
                predicted_values=["urban"],
                reference_status="valid",
                reference_values=["urban"],
                confidence=0.9,
                weight=1.0,
            ),
            _label_unit(
                2,
                predicted_status="valid",
                predicted_values=["rural"],
                reference_status="valid",
                reference_values=["urban"],
                confidence=0.8,
                weight=2.0,
                agreement="adjudicated_disagreement",
            ),
            _label_unit(
                3,
                predicted_status="unavailable",
                predicted_values=[],
                reference_status="valid",
                reference_values=["rural"],
                confidence=0.2,
                weight=1.0,
            ),
            _label_unit(
                4,
                predicted_status="valid",
                predicted_values=["urban"],
                reference_status="not_observable",
                reference_values=[],
                confidence=0.7,
                weight=1.0,
                agreement="majority",
            ),
        ],
        "event_units": [
            _event_unit(
                1,
                match_status="matched",
                predicted_interval=(100_000_000, 300_000_000),
                reference_interval=(120_000_000, 320_000_000),
                actor_continuity="correct",
            ),
            _event_unit(
                2,
                match_status="spurious",
                predicted_interval=(400_000_000, 500_000_000),
                reference_interval=None,
                actor_continuity="not_applicable",
                weight=2.0,
            ),
            _event_unit(
                3,
                match_status="missed",
                predicted_interval=None,
                reference_interval=(600_000_000, 700_000_000),
                actor_continuity="not_applicable",
            ),
            _event_unit(
                4,
                match_status="matched",
                predicted_interval=(800_000_000, 1_000_000_000),
                reference_interval=(780_000_000, 1_020_000_000),
                actor_continuity="switched",
                agreement="adjudicated_disagreement",
            ),
        ],
        "ontology_questions": [
            {
                "question_uid": "question-1",
                "label_key": "odd.road.context",
                "question": "How should a mixed industrial suburb be labeled?",
                "status": "open",
                "resolution": None,
            }
        ],
    }


def test_adjudicated_audit_computes_label_calibration_and_event_metrics() -> None:
    annotations = _annotations()

    result = evaluate_human_audit_annotations(
        annotations,
        _ontology(),
    )

    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["status"] == "measured"
    assert result["certified"] is False
    assert len(result["annotation_document_sha256"]) == 64
    assert result["ontology_questions"] == {
        "total": 1,
        "open": 1,
        "resolved": 0,
    }
    assert result["reviewer_agreement"]["counts"] == {
        "adjudicated_disagreement": 2,
        "majority": 1,
        "unanimous": 5,
    }

    values = {
        row["value"]: row for row in result["label_value_metrics"]
    }
    urban = values["urban"]
    assert urban["true_positive"] == 1
    assert urban["false_positive"] == 0
    assert urban["false_negative"] == 1
    assert urban["precision"] == 1.0
    assert urban["recall"] == 0.5
    assert urban["sample_sufficiency"]["positive_sufficient"] is False
    rural = values["rural"]
    assert rural["true_positive"] == 0
    assert rural["false_positive"] == 1
    assert rural["false_negative"] == 1
    assert rural["precision"] == 0.0
    assert rural["recall"] == 0.0
    assert urban["precision_ci95"][0] < urban["precision_ci95"][1]

    calibration = result["calibration_metrics"][0]
    assert calibration["audited_predictions"] == 2
    assert calibration["expected_calibration_error"] == pytest.approx(0.5)
    assert calibration["bands"][0]["empirical_accuracy"] == pytest.approx(
        1.0 / 3.0
    )

    event = result["event_metrics"][0]
    assert event["matched"] == 2
    assert event["spurious"] == 1
    assert event["missed"] == 1
    assert event["precision"] == pytest.approx(2.0 / 3.0)
    assert event["recall"] == pytest.approx(2.0 / 3.0)
    assert event["weighted_precision"] == 0.5
    assert event["weighted_recall"] == pytest.approx(2.0 / 3.0)
    assert event["boundary_error_ms"]["onset_mean"] == 20.0
    assert event["boundary_error_ms"]["offset_mean"] == 20.0
    assert event["actor_continuity"]["error_rate"] == 0.5
    assert event["actor_continuity"]["switched_count"] == 1
    json.dumps(result, allow_nan=False)


def test_draft_or_single_reviewer_cannot_produce_measured_results() -> None:
    draft = _annotations()
    draft["adjudication_status"] = "draft"
    draft["reviewer_count"] = 1

    validate_human_audit_annotations(draft, _ontology())
    with pytest.raises(ValueError, match="require adjudicated"):
        evaluate_human_audit_annotations(draft, _ontology())

    invalid = _annotations()
    invalid["reviewer_count"] = 1
    with pytest.raises(ValueError, match="at least two reviewers"):
        validate_human_audit_annotations(invalid, _ontology())


def test_duplicate_evidence_and_event_annotations_are_rejected() -> None:
    duplicate_evidence = _annotations()
    duplicate_evidence["label_units"][1]["prediction"]["evidence_uid"] = (
        duplicate_evidence["label_units"][0]["prediction"]["evidence_uid"]
    )
    with pytest.raises(ValueError, match="evidence_uid"):
        validate_human_audit_annotations(
            duplicate_evidence,
            _ontology(),
        )

    duplicate_event = _annotations()
    duplicate_event["event_units"][1]["predicted_event_uid"] = (
        duplicate_event["event_units"][0]["predicted_event_uid"]
    )
    with pytest.raises(ValueError, match="predicted event"):
        validate_human_audit_annotations(duplicate_event, _ontology())


def test_annotation_contract_rejects_invalid_values_and_event_sides() -> None:
    invalid_value = _annotations()
    invalid_value["label_units"][0]["reference"]["values"] = ["motorway"]
    with pytest.raises(ValueError, match="outside the ontology"):
        validate_human_audit_annotations(invalid_value, _ontology())

    invalid_event = _annotations()
    invalid_event["event_units"][1]["actor_continuity"] = "correct"
    with pytest.raises(ValueError, match="must be not_applicable"):
        validate_human_audit_annotations(invalid_event, _ontology())

    invalid_question = copy.deepcopy(_annotations())
    invalid_question["ontology_questions"][0]["status"] = "resolved"
    with pytest.raises(ValueError, match="needs a resolution"):
        validate_human_audit_annotations(
            invalid_question,
            _ontology(),
        )
