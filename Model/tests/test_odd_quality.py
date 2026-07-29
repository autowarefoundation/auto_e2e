from __future__ import annotations

import copy

import pytest

from data_processing.odd_labeling.quality import (
    build_quality_documents,
    validate_labelset_records,
)


def _ontology() -> dict:
    return {
        "statuses": [
            "valid",
            "unavailable",
            "not_observable",
            "ambiguous",
        ],
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
                "values": [{"value": "urban"}, {"value": "rural"}],
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


def _record(scene_uid: str = "scene-a") -> dict:
    return {
        "scene_uid": scene_uid,
        "start_timestamp_ns": 100,
        "end_timestamp_ns": 400,
        "distance_m": 12.5,
        "evidence": [
            {
                "evidence_uid": f"evidence-road-{scene_uid}",
                "label_key": "odd.road.context",
                "status": "valid",
                "values": ["urban"],
                "source": "map_route",
                "confidence": 0.95,
                "candidate_values": [],
                "scope": {
                    "scene_uid": scene_uid,
                    "start_timestamp_ns": 100,
                    "end_timestamp_ns": 400,
                },
            },
            {
                "evidence_uid": f"evidence-event-{scene_uid}",
                "label_key": "event.ego.maneuver",
                "status": "valid",
                "values": ["turn_left"],
                "source": "gnss_ins",
                "confidence": 0.9,
                "candidate_values": [],
                "scope": {
                    "scene_uid": scene_uid,
                    "start_timestamp_ns": 150,
                    "end_timestamp_ns": 350,
                },
            },
        ],
        "observations": [
            {
                "observation_uid": f"observation-road-{scene_uid}",
                "scene_uid": scene_uid,
                "key": "odd.road.context",
                "status": "valid",
                "values": ["urban"],
                "source": "map_route",
                "confidence": 0.95,
                "start_timestamp_ns": 100,
                "end_timestamp_ns": 400,
                "evidence_uids": [f"evidence-road-{scene_uid}"],
                "conflicting_evidence_uids": [],
            },
            {
                "observation_uid": f"observation-event-{scene_uid}",
                "scene_uid": scene_uid,
                "key": "event.ego.maneuver",
                "status": "valid",
                "values": ["turn_left"],
                "source": "gnss_ins",
                "confidence": 0.9,
                "start_timestamp_ns": 150,
                "end_timestamp_ns": 350,
                "evidence_uids": [f"evidence-event-{scene_uid}"],
                "conflicting_evidence_uids": [],
            },
        ],
        "events": [
            {
                "event_uid": f"event-{scene_uid}",
                "scene_uid": scene_uid,
                "start_timestamp_ns": 150,
                "end_timestamp_ns": 350,
                "primary_event_key": "event.ego.maneuver",
                "status": "valid",
                "confidence": 0.9,
                "observation_uids": [
                    f"observation-event-{scene_uid}"
                ],
                "supporting_evidence_uids": [
                    f"evidence-event-{scene_uid}"
                ],
                "phases": [
                    {
                        "phase": "onset",
                        "start_timestamp_ns": 150,
                        "end_timestamp_ns": 200,
                    },
                    {
                        "phase": "active",
                        "start_timestamp_ns": 200,
                        "end_timestamp_ns": 300,
                    },
                    {
                        "phase": "resolution",
                        "start_timestamp_ns": 300,
                        "end_timestamp_ns": 350,
                    },
                ],
            }
        ],
    }


def _statistics() -> dict:
    common = {
        "namespace": "odd",
        "eligible_duration_ns": 300,
        "valid_duration_ns": 300,
        "eligible_distance_m": 12.5,
        "valid_distance_m": 12.5,
        "attempted_count": 1,
        "successful_count": 1,
        "conflict_count": 0,
        "quality_tier": "experimental",
        "status_scene_counts": {
            "valid": 1,
            "unavailable": 0,
            "not_observable": 0,
            "ambiguous": 0,
        },
        "status_duration_ns": {
            "valid": 300,
            "unavailable": 0,
            "not_observable": 0,
            "ambiguous": 0,
        },
    }
    event = dict(common)
    event["namespace"] = "event"
    return {
        "labelset_id": "oddls-test",
        "scene_count": 1,
        "keys": [
            {"key": "odd.road.context", **common},
            {"key": "event.ego.maneuver", **event},
        ],
    }


def test_quality_documents_validate_and_keep_certification_pending() -> None:
    record = _record()

    documents = build_quality_documents(
        [record],
        _statistics(),
        _ontology(),
        labelset_id="oddls-test",
    )

    assert set(documents) == {
        "coverage",
        "audit_manifest",
        "calibration",
    }
    coverage = documents["coverage"]
    assert coverage["structural_validation"]["status"] == "passed"
    assert coverage["structural_validation"]["evidence_count"] == 2
    assert all(
        row["support_state"] == "supported_experimental"
        for row in coverage["keys"]
    )
    audit = documents["audit_manifest"]
    assert audit["status"] == "pending_human_audit"
    assert audit["strata"]
    assert all(
        len(row["selected"]) <= row["target_count"]
        for row in audit["strata"]
    )
    calibration = documents["calibration"]
    assert calibration["rows"]
    assert all(
        row["calibration_status"] == "pending_human_audit"
        and row["certified"] is False
        for row in calibration["rows"]
    )


def test_audit_selection_is_independent_of_scene_order() -> None:
    records = [_record("scene-b"), _record("scene-a")]

    first = build_quality_documents(
        records,
        _statistics(),
        _ontology(),
        labelset_id="oddls-test",
    )
    second = build_quality_documents(
        list(reversed(records)),
        _statistics(),
        _ontology(),
        labelset_id="oddls-test",
    )

    assert first == second


def test_coverage_uses_dataset_support_contract_states() -> None:
    statistics = _statistics()
    statistics["keys"].extend(
        [
            {
                "key": "odd.unsupported",
                "namespace": "odd",
                "eligible_duration_ns": 300,
                "valid_duration_ns": 0,
                "eligible_distance_m": 12.5,
                "valid_distance_m": 0,
                "attempted_count": 1,
                "successful_count": 0,
                "conflict_count": 0,
                "quality_tier": "experimental",
                "status_scene_counts": {
                    "valid": 0,
                    "unavailable": 1,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
                "status_duration_ns": {
                    "valid": 0,
                    "unavailable": 300,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
            },
            {
                "key": "odd.disabled",
                "namespace": "odd",
                "eligible_duration_ns": 300,
                "valid_duration_ns": 0,
                "eligible_distance_m": 12.5,
                "valid_distance_m": 0,
                "attempted_count": 0,
                "successful_count": 0,
                "conflict_count": 0,
                "quality_tier": "experimental",
                "status_scene_counts": {
                    "valid": 0,
                    "unavailable": 0,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
                "status_duration_ns": {
                    "valid": 0,
                    "unavailable": 0,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
            },
        ]
    )
    ontology = _ontology()
    ontology["labels"].extend(
        [
            {
                "key": "odd.unsupported",
                "namespace": "odd",
                "cardinality": "single",
                "values": [{"value": "value"}],
            },
            {
                "key": "odd.disabled",
                "namespace": "odd",
                "cardinality": "single",
                "values": [{"value": "value"}],
            },
        ]
    )

    coverage = build_quality_documents(
        [_record()],
        statistics,
        ontology,
        labelset_id="oddls-test",
    )["coverage"]
    support = {
        row["key"]: row["support_state"] for row in coverage["keys"]
    }

    assert support["odd.road.context"] == "supported_experimental"
    assert support["odd.unsupported"] == "unsupported_missing_source"
    assert support["odd.disabled"] == "disabled_pending_audit"


def test_structural_validation_rejects_invalid_label_and_reference() -> None:
    invalid_status = _record()
    invalid_status["observations"][0]["status"] = "not_observable"

    with pytest.raises(ValueError, match="non-valid status"):
        validate_labelset_records([invalid_status], _ontology())

    invalid_reference = _record()
    invalid_reference["observations"][0]["evidence_uids"] = ["unknown"]

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_labelset_records([invalid_reference], _ontology())

    duplicate = _record()
    duplicate["events"].append(copy.deepcopy(duplicate["events"][0]))

    with pytest.raises(ValueError, match="duplicate event_uid"):
        validate_labelset_records([duplicate], _ontology())
