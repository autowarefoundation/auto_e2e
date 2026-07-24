"""Route-quality policy and corpus audit tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from navigation.quality import (
    DEFAULT_NAVIGATION_QUALITY_POLICY,
    NAVIGATION_QUALITY_AUDIT_VERSION,
    PACKED_NAVIGATION_QUALITY_AUDIT_VERSION,
    audit_navigation_quality,
    audit_packed_navigation_quality,
    load_packed_navigation_quality,
    verify_packed_navigation_quality_audit,
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


def _write_partition(root, record, *, partition_id=None, total_samples=100):
    root.mkdir()
    quality = {
        **record,
        "quality_policy": DEFAULT_NAVIGATION_QUALITY_POLICY.contract(),
    }
    quality_bytes = json.dumps(
        quality,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    (root / "navigation_quality.json").write_bytes(quality_bytes)
    manifest = {
        "total_samples": total_samples,
        "partition_id": partition_id or f"part-{record['scene_id']}",
        "navigation": {
            "scenes": [
                {
                    "scene_id": record["scene_id"],
                    "path": ".",
                    "hashes": {
                        "navigation_quality.json": hashlib.sha256(
                            quality_bytes
                        ).hexdigest(),
                    },
                }
            ]
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )
    return quality_bytes


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


def test_packed_quality_loader_checks_declared_hash(tmp_path):
    root = tmp_path / "partition"
    quality = {
        **_record("scene-a"),
        "quality_policy": DEFAULT_NAVIGATION_QUALITY_POLICY.contract(),
    }
    quality_bytes = _write_partition(
        root,
        _record("scene-a"),
        partition_id="scene-a",
    )

    records, identities = load_packed_navigation_quality([root])

    assert records == [quality]
    assert identities[0]["scene_id"] == "scene-a"
    (root / "navigation_quality.json").write_bytes(
        quality_bytes + b" "
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_packed_navigation_quality([root])


def test_packed_quality_audit_binds_partition_decisions(tmp_path):
    accepted = tmp_path / "accepted"
    excluded = tmp_path / "excluded"
    _write_partition(accepted, _record("scene-a"))
    _write_partition(
        excluded,
        _record(
            "scene-b",
            matched_ratio=0.5,
            route_valid=False,
        ),
    )

    audit = audit_packed_navigation_quality([excluded, accepted])

    assert (
        audit["schema_version"]
        == PACKED_NAVIGATION_QUALITY_AUDIT_VERSION
    )
    assert audit["accepted_partition_ids"] == ["part-scene-a"]
    assert audit["excluded_partition_ids"] == ["part-scene-b"]
    assert [item["partition_id"] for item in audit["packed_artifacts"]] == [
        "part-scene-a",
        "part-scene-b",
    ]
    assert verify_packed_navigation_quality_audit(
        audit,
        [accepted, excluded],
    ) == audit

    audit["accepted_partition_ids"] = []
    with pytest.raises(ValueError, match="differs from packed artifacts"):
        verify_packed_navigation_quality_audit(
            audit,
            [accepted, excluded],
        )


def test_packed_quality_audit_rejects_sample_count_drift(tmp_path):
    root = tmp_path / "partition"
    _write_partition(
        root,
        _record("scene-a"),
        total_samples=101,
    )

    with pytest.raises(ValueError, match="sample count differs"):
        audit_packed_navigation_quality([root])
