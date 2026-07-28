from __future__ import annotations

import io

import pytest

from data_processing.odd_labeling.parquet import build_parquet_artifacts


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _record(scene_uid: str, offset_ns: int) -> dict:
    evidence_uid = f"oddev-{scene_uid}"
    observation_uid = f"oddobs-{scene_uid}"
    event_uid = f"oddevent-{scene_uid}"
    start_ns = 1_000_000_000 + offset_ns
    end_ns = start_ns + 3_000_000_000
    return {
        "schema_version": "odd_scene_labels_v1",
        "scene_uid": scene_uid,
        "dataset_name": "synthetic",
        "dataset_version": "v1",
        "dataset_manifest_sha256": "a" * 64,
        "capability_manifest_sha256": "b" * 64,
        "start_timestamp_ns": start_ns,
        "end_timestamp_ns": end_ns,
        "distance_m": 42.5,
        "source_artifact_uri": f"s3://example/{scene_uid}",
        "source_artifact_sha256": "c" * 64,
        "provenance": {"labeler_version": "test-v1"},
        "evidence": [
            {
                "schema_version": "odd_label_evidence_v1",
                "evidence_uid": evidence_uid,
                "label_key": "event.ego.maneuver",
                "cardinality": "single",
                "values": ["turn_left"],
                "candidate_values": [
                    {
                        "value": "turn_left",
                        "score": 0.95,
                        "evidence_ref": "trajectory",
                    }
                ],
                "status": "valid",
                "confidence": 0.95,
                "source": "gnss_ins",
                "scope": {
                    "dataset_name": "synthetic",
                    "dataset_version": "v1",
                    "scene_uid": scene_uid,
                    "start_timestamp_ns": start_ns,
                    "end_timestamp_ns": end_ns,
                    "subject_type": "scene",
                    "subject_id": None,
                    "anchor_timestamp_ns": start_ns + 1_000_000_000,
                    "camera_ids": ["front_center"],
                    "coordinate_frame": "ego_flu",
                    "spatial_roi": {"kind": "trajectory"},
                },
                "measurements": [
                    {
                        "name": "heading_change_deg",
                        "value": 91.25,
                        "unit": "degree",
                        "quality": "valid",
                        "aggregation": "interval",
                    }
                ],
                "evidence_refs": [
                    {
                        "artifact_uri": f"s3://example/{scene_uid}/ego.npy",
                        "artifact_sha256": "d" * 64,
                        "timestamp_ns": start_ns + 1_000_000_000,
                        "camera_id": None,
                    }
                ],
                "provenance": {
                    "labeler_name": "trajectory_resolver",
                    "labeler_version": "v1",
                    "code_commit": "e" * 40,
                    "container_image_digest": f"sha256:{'f' * 64}",
                    "config_sha256": "1" * 64,
                    "ontology_sha256": "2" * 64,
                    "input_artifact_sha256s": ["3" * 64],
                    "model_provider": None,
                    "model_name": None,
                    "model_revision": None,
                    "prompt_sha256": None,
                    "decoding_config_sha256": None,
                    "lookback_ns": 500_000_000,
                    "lookahead_ns": 1_000_000_000,
                    "details": {"request_attempt": 1},
                },
            }
        ],
        "observations": [
            {
                "schema_version": "odd_scene_labels_v1",
                "observation_uid": observation_uid,
                "scene_uid": scene_uid,
                "key": "event.ego.maneuver",
                "status": "valid",
                "values": ["turn_left"],
                "confidence": 0.95,
                "source": "fusion",
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
                "evidence_uids": [evidence_uid],
                "conflicting_evidence_uids": [
                    f"oddev-opposed-{scene_uid}"
                ],
                "measurements": {"heading_change_deg": 91.25},
                "provenance": {
                    "fusion_version": "v1",
                    "policy": "authoritative_source_override",
                },
                "camera_id": None,
                "actor_track_uid": None,
                "event_uid": event_uid,
            }
        ],
        "events": [
            {
                "schema_version": "odd_event_instance_v1",
                "event_uid": event_uid,
                "scene_uid": scene_uid,
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
                "primary_event_key": "event.ego.maneuver",
                "actor_track_uids": [],
                "observation_uids": [observation_uid],
                "phases": [
                    {
                        "phase": "onset",
                        "start_timestamp_ns": start_ns,
                        "end_timestamp_ns": start_ns + 1_000_000_000,
                    },
                    {
                        "phase": "active",
                        "start_timestamp_ns": start_ns + 1_000_000_000,
                        "end_timestamp_ns": start_ns + 2_000_000_000,
                    },
                    {
                        "phase": "resolution",
                        "start_timestamp_ns": start_ns + 2_000_000_000,
                        "end_timestamp_ns": end_ns,
                    },
                ],
                "confidence": 0.95,
                "status": "valid",
                "supporting_evidence_uids": [evidence_uid],
                "provenance": {"segmenter_version": "v1"},
            }
        ],
    }


def _statistics() -> dict:
    return {
        "schema_version": "odd_statistics_v1",
        "labelset_id": "oddls-test",
        "scene_count": 2,
        "scene_duration_ns": 6_000_000_000,
        "keys": [
            {
                "key": "event.ego.maneuver",
                "namespace": "event",
                "quality_tier": "experimental",
                "valid_scene_count": 2,
                "eligible_scene_count": 2,
                "observable_scene_coverage": 1.0,
                "eligible_duration_ns": 6_000_000_000,
                "valid_duration_ns": 6_000_000_000,
                "observable_duration_coverage": 1.0,
                "eligible_distance_m": 85.0,
                "valid_distance_m": 85.0,
                "observable_distance_coverage": 1.0,
                "valid_interval_count": 2,
                "attempted_count": 2,
                "successful_count": 2,
                "conflict_count": 0,
                "status_scene_counts": {
                    "valid": 2,
                    "unavailable": 0,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
                "status_duration_ns": {
                    "valid": 6_000_000_000,
                    "unavailable": 0,
                    "not_observable": 0,
                    "ambiguous": 0,
                },
                "status_distance_m": {
                    "valid": 85.0,
                    "unavailable": 0.0,
                    "not_observable": 0.0,
                    "ambiguous": 0.0,
                },
                "source_scene_counts": {"fusion": 2},
                "source_duration_ns": {"fusion": 6_000_000_000},
                "source_distance_m": {"fusion": 85.0},
                "confidence": {
                    "observation_count": 2,
                    "duration_weighted_mean": 0.95,
                    "p10": 0.95,
                    "p50": 0.95,
                    "p90": 0.95,
                    "bins": [],
                },
                "values": [
                    {
                        "value": "turn_left",
                        "scene_count": 2,
                        "scene_ratio": 1.0,
                        "scene_ratio_ci95": {
                            "lower": 0.34,
                            "upper": 1.0,
                            "method": "wilson_scene_95",
                        },
                        "duration_ns": 6_000_000_000,
                        "duration_ratio": 1.0,
                        "duration_ratio_ci95": {
                            "lower": 1.0,
                            "upper": 1.0,
                            "method": "scene_clustered_bootstrap_95",
                            "replicates": 256,
                        },
                        "distance_m": 85.0,
                        "distance_ratio": 1.0,
                        "distance_ratio_ci95": {
                            "lower": 1.0,
                            "upper": 1.0,
                            "method": "scene_clustered_bootstrap_95",
                            "replicates": 256,
                        },
                        "valid_interval_count": 2,
                        "event_instance_count": 2,
                        "confidence": {
                            "observation_count": 2,
                            "duration_weighted_mean": 0.95,
                            "p10": 0.95,
                            "p50": 0.95,
                            "p90": 0.95,
                            "bins": [],
                        },
                    }
                ],
            }
        ],
        "cooccurrences": {
            "minimum_overlap_ns": 100_000_000,
            "odd_pairs": [
                {
                    "left_key": "odd.environment.sky",
                    "left_value": "clear",
                    "right_key": "odd.road.context",
                    "right_value": "urban",
                    "scene_count": 2,
                    "overlap_duration_ns": 6_000_000_000,
                    "overlap_distance_m": 85.0,
                }
            ],
            "odd_event": [
                {
                    "odd_key": "odd.road.context",
                    "odd_value": "urban",
                    "event_key": "event.ego.maneuver",
                    "event_value": "turn_left",
                    "scene_count": 2,
                    "event_instance_count": 2,
                    "overlap_duration_ns": 6_000_000_000,
                    "overlap_distance_m": 85.0,
                }
            ],
        },
    }


def _build(records: list[dict]) -> dict:
    return build_parquet_artifacts(
        records,
        _statistics(),
        labelset_id="oddls-test",
        dataset_name="synthetic",
        dataset_version="v1",
        dataset_manifest_sha256="a" * 64,
        capability_manifest_sha256="b" * 64,
        ontology_sha256="2" * 64,
    )


def _parquet_file(payload: bytes):
    return pq.ParquetFile(io.BytesIO(payload))


def test_odd_parquet_artifacts_are_explicit_and_scene_grouped() -> None:
    artifacts = _build(
        [_record("scene-b", 10_000_000_000), _record("scene-a", 0)]
    )

    assert set(artifacts) == {
        "scene_records",
        "evidence",
        "observations",
        "events",
        "statistics",
        "odd_cooccurrences",
        "odd_event_cooccurrences",
        "conflicts",
    }
    assert {
        name: artifact.row_count for name, artifact in artifacts.items()
    } == {
        "scene_records": 2,
        "evidence": 2,
        "observations": 2,
        "events": 2,
        "statistics": 1,
        "odd_cooccurrences": 1,
        "odd_event_cooccurrences": 1,
        "conflicts": 2,
    }
    for name, artifact in artifacts.items():
        parquet_file = _parquet_file(artifact.payload)
        metadata = parquet_file.schema_arrow.metadata
        assert metadata[b"odd.parquet_schema_version"] == b"odd_parquet_v3"
        assert metadata[b"odd.table_name"] == name.encode("ascii")
        assert metadata[b"odd.labelset_id"] == b"oddls-test"
        assert metadata[b"odd.ontology_sha256"] == b"2" * 64
        assert parquet_file.metadata.num_rows == artifact.row_count

    for name in (
        "scene_records",
        "evidence",
        "observations",
        "events",
        "conflicts",
    ):
        parquet_file = _parquet_file(artifacts[name].payload)
        assert parquet_file.num_row_groups == 2
        assert [
            parquet_file.read_row_group(index, columns=["scene_uid"])
            .column("scene_uid")
            .to_pylist()
            for index in range(2)
        ] == [["scene-a"], ["scene-b"]]


def test_odd_parquet_preserves_nested_evidence_and_event_fields() -> None:
    artifacts = _build([_record("scene-a", 0), _record("scene-b", 10)])

    evidence = pq.read_table(
        pa.BufferReader(artifacts["evidence"].payload)
    ).to_pylist()[0]
    event = pq.read_table(
        pa.BufferReader(artifacts["events"].payload)
    ).to_pylist()[0]
    conflict = pq.read_table(
        pa.BufferReader(artifacts["conflicts"].payload)
    ).to_pylist()[0]

    assert evidence["candidate_values"][0] == {
        "value": "turn_left",
        "score": pytest.approx(0.95),
        "evidence_ref": "trajectory",
    }
    assert evidence["measurements"][0]["value_json"] == "91.25"
    assert evidence["evidence_refs"][0]["artifact_sha256"] == "d" * 64
    assert [phase["phase"] for phase in event["phases"]] == [
        "onset",
        "active",
        "resolution",
    ]
    assert event["supporting_evidence_uids"] == ["oddev-scene-a"]
    assert conflict["conflicting_evidence_uids"] == [
        "oddev-opposed-scene-a"
    ]
    assert conflict["fusion_policy"] == "authoritative_source_override"


def test_odd_parquet_uses_dictionary_encoding_and_is_deterministic() -> None:
    records = [_record("scene-b", 10_000_000_000), _record("scene-a", 0)]
    first = _build(records)
    second = _build(list(reversed(records)))

    assert {
        name: artifact.sha256 for name, artifact in first.items()
    } == {
        name: artifact.sha256 for name, artifact in second.items()
    }
    assert all(
        first[name].payload == second[name].payload for name in first
    )

    parquet_file = _parquet_file(first["evidence"].payload)
    label_key_column = next(
        index
        for index in range(parquet_file.metadata.num_columns)
        if parquet_file.metadata.schema.column(index).path == "label_key"
    )
    for row_group in range(parquet_file.num_row_groups):
        encodings = parquet_file.metadata.row_group(row_group).column(
            label_key_column
        ).encodings
        assert any("DICTIONARY" in encoding for encoding in encodings)
