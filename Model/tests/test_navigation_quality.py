"""Route-quality policy and corpus audit tests."""

from __future__ import annotations

import pytest

from navigation.quality import (
    DEFAULT_NAVIGATION_QUALITY_POLICY,
    NAVIGATION_QUALITY_AUDIT_VERSION,
    audit_navigation_quality,
)


def _record(
    scene_id,
    *,
    matched_ratio=1.0,
    p95_distance=1.0,
    p95_heading=0.1,
    unresolved=0,
    route_valid=True,
    failures=(),
):
    return {
        "scene_id": scene_id,
        "route_valid": route_valid,
        "route_confidence": 0.9 if route_valid else 0.1,
        "sample_count": 100,
        "quality": {
            "matched_pose_ratio": matched_ratio,
            "median_lateral_distance_m": 0.5,
            "p95_lateral_distance_m": p95_distance,
            "median_heading_error_rad": 0.05,
            "p95_heading_error_rad": p95_heading,
            "shortest_path_fill_count": 0,
            "shortest_path_fill_length_m": 0.0,
            "adjacent_transition_count": 0,
            "unresolved_discontinuities": unresolved,
            "failure_reasons": list(failures),
        },
    }


def test_quality_audit_records_acceptance_distribution_and_failures():
    records = [
        _record("accepted"),
        _record(
            "excluded",
            matched_ratio=0.5,
            p95_distance=7.0,
            route_valid=False,
            failures=(
                "matched_pose_ratio_below_threshold",
                "p95_distance_above_threshold",
            ),
        ),
    ]

    audit = audit_navigation_quality(records)

    assert audit["schema_version"] == NAVIGATION_QUALITY_AUDIT_VERSION
    assert audit["policy"] == (
        DEFAULT_NAVIGATION_QUALITY_POLICY.contract()
    )
    assert audit["scene_count"] == 2
    assert audit["accepted_scene_count"] == 1
    assert audit["excluded_scene_count"] == 1
    assert audit["accepted_scene_fraction"] == 0.5
    assert audit["sample_count"] == 200
    assert audit["failure_reason_counts"] == {
        "matched_pose_ratio_below_threshold": 1,
        "p95_distance_above_threshold": 1,
    }
    assert audit["distributions"]["matched_pose_ratio"]["min"] == 0.5
    assert len(audit["source_records_sha256"]) == 64


def test_quality_audit_rejects_persisted_validity_policy_drift():
    with pytest.raises(ValueError, match="persisted route validity"):
        audit_navigation_quality([
            _record(
                "drift",
                matched_ratio=0.1,
                route_valid=True,
                failures=("matched_pose_ratio_below_threshold",),
            )
        ])


def test_quality_audit_rejects_duplicate_scenes():
    with pytest.raises(ValueError, match="duplicate scene IDs"):
        audit_navigation_quality([
            _record("same"),
            _record("same"),
        ])


def test_no_lane_sequence_is_always_excluded():
    audit = audit_navigation_quality([
        _record(
            "empty",
            matched_ratio=1.0,
            route_valid=False,
            failures=("no_lane_sequence",),
        )
    ])

    assert audit["accepted_scene_count"] == 0
    assert audit["failure_reason_counts"] == {"no_lane_sequence": 1}
