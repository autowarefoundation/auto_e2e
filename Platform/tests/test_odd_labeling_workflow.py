from __future__ import annotations

import json
from pathlib import Path

import pytest

from Platform.pipelines.odd_labeling_workflow import (
    ODD_LABELER_VERSION,
    ODD_SOURCE_POLICY_VERSIONS,
    _execution_receipt,
    _execution_receipt_key,
    _provider_exchange_key,
    _provider_report,
    _provider_report_key,
    _publication_scope,
    _put_immutable,
    _scene_summary,
    _semantic_output_merkle_root,
    _statistics,
    _union_duration,
    fuse_odd_scene,
    label_odd_bedrock_map,
    label_odd_image_quality,
    label_odd_kinematics,
    label_odd_map_route,
    label_odd_visual,
    odd_dataset_labeler_launch_plan,
    publish_odd_labelset,
    resolve_odd_scenes,
    wf_generate_odd_labelset,
)


class _RecordingS3:
    def __init__(self) -> None:
        self.request = {}

    def put_object(self, **kwargs) -> None:
        self.request = kwargs


def test_union_duration_counts_overlapping_intervals_once() -> None:
    assert _union_duration([(10, 20), (15, 30), (40, 50)]) == 30


def test_latest_publication_requires_complete_scene_inventory() -> None:
    assert _publication_scope(404, 404, "full") == "full"
    assert _publication_scope(1, 404, "smoke") == "smoke"

    try:
        _publication_scope(1, 404, "full")
    except ValueError as error:
        assert "complete scene inventory" in str(error)
    else:
        raise AssertionError("partial LabelSet was accepted as latest")


def test_statistics_deduplicate_camera_and_source_coverage() -> None:
    records = [
        {
            "scene_uid": "scene-1",
            "start_timestamp_ns": 0,
            "end_timestamp_ns": 100,
            "distance_m": 25.0,
            "observations": [
                {
                    "observation_uid": "oddobs-sky-vlm",
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "source": "vlm",
                    "confidence": 0.9,
                    "start_timestamp_ns": 0,
                    "end_timestamp_ns": 100,
                    "measurements": {},
                    "conflicting_evidence_uids": [],
                },
                {
                    "observation_uid": "oddobs-sky-fusion",
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "source": "fusion",
                    "confidence": 0.8,
                    "start_timestamp_ns": 20,
                    "end_timestamp_ns": 80,
                    "measurements": {},
                    "conflicting_evidence_uids": [],
                },
            ],
        }
    ]
    ontology = {
        "statuses": [
            "valid",
            "unavailable",
            "not_observable",
            "ambiguous",
        ],
        "labels": [
            {
                "key": "odd.environment.sky",
                "namespace": "odd",
                "values": [
                    {"value": "clear"},
                    {"value": "partly_cloudy"},
                    {"value": "overcast"},
                ],
            }
        ],
    }

    statistics = _statistics(records, ontology, "oddls-test")
    row = statistics["keys"][0]

    assert statistics["schema_version"] == "odd_statistics_v2"
    assert row["valid_duration_ns"] == 100
    assert row["valid_distance_m"] == 25.0
    assert row["status_duration_ns"]["valid"] == 100
    assert row["values"][0]["duration_ns"] == 100
    assert row["values"][0]["duration_ratio"] == 1.0


def test_scene_summary_pins_record_integrity() -> None:
    summary = _scene_summary(
        {
            "scene_uid": "scene-1",
            "start_timestamp_ns": 0,
            "end_timestamp_ns": 100,
            "distance_m": 12.5,
            "observations": [],
        },
        shard_name="scene-1.tar",
        record_key="kitscenes/v3.0/odd/labelsets/test/scenes/record.json",
        record_sha256="a" * 64,
        record_byte_size=123,
    )

    assert summary["record_sha256"] == "a" * 64
    assert summary["record_byte_size"] == 123


def test_scene_summary_preserves_search_scope_and_events() -> None:
    summary = _scene_summary(
        {
            "scene_uid": "scene-1",
            "start_timestamp_ns": 0,
            "end_timestamp_ns": 100,
            "distance_m": 12.5,
            "observations": [
                {
                    "key": "event.vehicle.interaction",
                    "status": "valid",
                    "values": ["cut_in"],
                    "source": "vlm",
                    "confidence": 0.8,
                    "start_timestamp_ns": 10,
                    "end_timestamp_ns": 30,
                    "camera_id": "front",
                    "actor_track_uid": "vehicle-a",
                    "event_uid": "event-1",
                },
                {
                    "key": "event.vehicle.interaction",
                    "status": "valid",
                    "values": ["cut_in"],
                    "source": "vlm",
                    "confidence": 0.9,
                    "start_timestamp_ns": 40,
                    "end_timestamp_ns": 60,
                    "camera_id": "front",
                    "actor_track_uid": "vehicle-a",
                    "event_uid": "event-1",
                },
            ],
            "events": [
                {
                    "event_uid": "event-1",
                    "primary_event_key": "event.vehicle.interaction",
                    "start_timestamp_ns": 10,
                    "end_timestamp_ns": 60,
                    "status": "valid",
                    "confidence": 0.8,
                    "actor_track_uids": ["vehicle-a"],
                    "provenance": {
                        "primary_values": ["cut_in"],
                        "outcome": "unresolved",
                    },
                }
            ],
        },
        shard_name="scene-1.tar",
        record_key="odd/scenes/scene-1.json",
        record_sha256="a" * 64,
        record_byte_size=123,
    )

    observation = summary["observations"][0]
    assert observation["interval_count"] == 2
    assert observation["duration_ns"] == 40
    assert observation["camera_id"] == "front"
    assert observation["actor_track_uid"] == "vehicle-a"
    assert summary["events"][0]["outcome"] == "unresolved"


def test_immutable_parquet_upload_pins_binary_contract() -> None:
    s3 = _RecordingS3()

    _put_immutable(
        s3,
        "datasets",
        "odd/evidence/part-00000.parquet",
        b"parquet",
        content_type="application/vnd.apache.parquet",
        schema_version="odd_parquet_v1",
        maximum_bytes=7,
    )

    assert s3.request["ContentType"] == "application/vnd.apache.parquet"
    assert s3.request["IfNoneMatch"] == "*"
    assert s3.request["Metadata"]["odd-schema"] == "odd_parquet_v1"
    assert len(s3.request["Metadata"]["sha256"]) == 64
    with pytest.raises(ValueError, match="exceeds size cap"):
        _put_immutable(
            s3,
            "datasets",
            "odd/evidence/part-00000.parquet",
            b"parquet",
            maximum_bytes=6,
        )


def test_semantic_output_merkle_root_covers_all_canonical_outputs() -> None:
    records = [
        {
            "scene_uid": "scene-b",
            "dataset_name": "kitscenes",
            "observations": [
                {
                    "observation_uid": "observation-b",
                    "key": "odd.environment.sky",
                    "values": ["clear"],
                }
            ],
            "evidence": [
                {
                    "evidence_uid": "evidence-b",
                    "label_key": "odd.environment.sky",
                    "values": ["clear"],
                }
            ],
            "events": [],
        },
        {
            "scene_uid": "scene-a",
            "dataset_name": "kitscenes",
            "observations": [],
            "evidence": [],
            "events": [
                {
                    "event_uid": "event-a",
                    "primary_event_key": "event.ego.maneuver",
                }
            ],
        },
    ]
    statistics = {
        "schema_version": "odd_statistics_v2",
        "labelset_id": "oddls-first",
        "scene_count": 2,
    }
    quality = {
        "coverage": {
            "labelset_id": "oddls-first",
            "structural_validation": {"status": "passed"},
        },
        "calibration": {
            "labelset_id": "oddls-first",
            "rows": [],
        },
    }

    first = _semantic_output_merkle_root(records, statistics, quality)
    second = _semantic_output_merkle_root(
        list(reversed(records)),
        {**statistics, "labelset_id": "oddls-second"},
        {
            name: {**document, "labelset_id": "oddls-second"}
            for name, document in reversed(list(quality.items()))
        },
    )

    assert first == second
    changed = [
        records[0],
        {
            **records[1],
            "events": [
                {
                    "event_uid": "event-a",
                    "primary_event_key": "event.ego.strong_response",
                }
            ],
        },
    ]
    assert _semantic_output_merkle_root(
        changed,
        statistics,
        quality,
    ) != first


def test_execution_receipt_is_separate_and_content_addressed() -> None:
    semantic_partition_sha256 = "a" * 64
    receipt = _execution_receipt(
        semantic_partition_sha256,
        {"wall_seconds": 12.5, "observation_count": 42},
        environment={
            "FLYTE_INTERNAL_EXECUTION_ID": "odd-full-run",
            "FLYTE_INTERNAL_TASK_NAME": "fuse_odd_scene",
            "FLYTE_ATTEMPT_NUMBER": "0",
            "HOSTNAME": "odd-full-run-n2-0",
        },
        created_at="2026-07-29T00:00:00Z",
    )

    assert receipt == {
        "receipt_schema_version": "odd_execution_receipt_v1",
        "semantic_partition_sha256": semantic_partition_sha256,
        "created_at": "2026-07-29T00:00:00Z",
        "flyte_execution_id": "odd-full-run",
        "flyte_task_execution_id": "odd-full-run-n2-0",
        "attempt": 1,
        "runtime_metrics": {
            "observation_count": 42,
            "wall_seconds": 12.5,
        },
    }
    key = _execution_receipt_key("kitscenes/v3.0/odd", receipt)
    assert key.startswith(
        "kitscenes/v3.0/odd/execution-receipts/"
        f"semantic-partition={semantic_partition_sha256}/receipt="
    )
    assert key.endswith(".json")
    assert "odd-full-run" not in key


def test_provider_report_aggregates_requests_without_raw_responses() -> None:
    exchanges = [
        {
            "backend": "ORV",
            "provider": "openai_compatible",
            "model": "cosmos",
            "model_revision": "revision-1",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
            "status": "succeeded",
            "attempt": 1,
            "latency_ms": 100.0,
            "input_image_count": 6,
            "request_metadata": {"bundle": "road"},
            "raw_response": {"result": "clear"},
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "error_type": None,
            "schema_version": "odd_provider_exchange_v1",
        },
        {
            "backend": "ORV",
            "provider": "openai_compatible",
            "model": "cosmos",
            "model_revision": "revision-1",
            "request_sha256": "c" * 64,
            "response_sha256": None,
            "status": "transport_error",
            "attempt": 1,
            "latency_ms": 300.0,
            "input_image_count": 6,
            "request_metadata": {"bundle": "road"},
            "raw_response": None,
            "usage": {},
            "error_type": "TimeoutError",
            "schema_version": "odd_provider_exchange_v1",
        },
        {
            "backend": "ORV",
            "provider": "openai_compatible",
            "model": "cosmos",
            "model_revision": "revision-1",
            "request_sha256": "c" * 64,
            "response_sha256": "d" * 64,
            "status": "succeeded",
            "attempt": 2,
            "latency_ms": 200.0,
            "input_image_count": 6,
            "request_metadata": {"bundle": "road"},
            "raw_response": {"result": "overcast"},
            "usage": {"input_tokens": 100, "output_tokens": 30},
            "error_type": None,
            "schema_version": "odd_provider_exchange_v1",
        },
    ]

    report = _provider_report(exchanges)

    assert report["schema_version"] == "odd_provider_report_v1"
    assert "raw_response" not in json.dumps(report)
    assert report["totals"] == {
        "attempt_count": 3,
        "failure_count": 1,
        "input_image_count": 18,
        "request_count": 2,
        "successful_count": 2,
    }
    backend = report["backends"][0]
    assert backend["backend"] == "ORV"
    assert backend["latency_ms"] == {
        "max": 300.0,
        "mean": 200.0,
        "p50": 200.0,
        "p95": 290.0,
        "total": 600.0,
    }
    assert backend["usage"] == {
        "input_tokens": 200,
        "output_tokens": 50,
    }
    assert backend["estimated_cost_usd"] is None
    assert (
        backend["cost_estimation_status"]
        == "unavailable_without_frozen_pricing"
    )


def test_provider_audit_keys_are_backend_separated_and_content_addressed() -> None:
    exchange = {
        "backend": "ORV",
        "provider": "openai_compatible",
        "model": "cosmos",
        "model_revision": "revision-1",
        "request_sha256": "a" * 64,
        "response_sha256": "b" * 64,
        "status": "succeeded",
        "attempt": 1,
        "latency_ms": 100.0,
        "input_image_count": 6,
        "request_metadata": {"bundle": "road"},
        "raw_response": {"result": "clear"},
        "usage": {},
        "error_type": None,
        "schema_version": "odd_provider_exchange_v1",
    }
    key = _provider_exchange_key(
        "kitscenes/v3.0/odd",
        "oddls-test",
        exchange,
    )

    assert "/provider-artifacts/" in key
    assert "/backend=ORV/" in key
    assert f"/request={'a' * 64}/" in key
    assert key.endswith(".json")
    changed = {**exchange, "latency_ms": 101.0}
    assert (
        _provider_exchange_key(
            "kitscenes/v3.0/odd",
            "oddls-test",
            changed,
        )
        != key
    )

    report = {
        **_provider_report([exchange]),
        "labelset_id": "oddls-test",
        "dataset_name": "kitscenes",
        "dataset_version": "v3.0",
        "exchange_artifacts": [{"key": key}],
    }
    report_key = _provider_report_key(
        "kitscenes/v3.0/odd",
        "oddls-test",
        report,
    )
    assert "/provider-reports/labelset=oddls-test/" in report_key
    assert report_key.endswith(".json")
    assert "raw_response" not in json.dumps(report)
    assert _provider_report([])["backends"] == []


def test_workflow_interface_does_not_expose_endpoint_url() -> None:
    inputs = wf_generate_odd_labelset.python_interface.inputs
    assert {
        "openai_base_url",
        "openai_model",
        "openai_model_revision",
        "bedrock_map_model_id",
        "bedrock_map_model_revision",
        "publish_latest",
    }.isdisjoint(inputs)
    assert {
        "ontology_version",
        "ontology_sha256",
        "labeler_bundle_version",
        "labeler_config_uri",
        "labeler_config_sha256",
        "enabled_sources",
        "road_vlm_provider",
        "road_vlm_model",
        "road_vlm_model_revision",
        "road_vlm_prompt_bundle_sha256",
        "road_vlm_decoding_config_sha256",
        "map_resolver_provider",
        "map_resolver_model_id",
        "map_resolver_model_revision",
        "map_resolver_prompt_bundle_sha256",
        "map_resolver_decoding_config_sha256",
        "fusion_config_sha256",
        "calibration_bundle_sha256",
        "publication_prefix",
        "labeler_image_digest",
        "labeler_source_revision",
        "openai_concurrency",
        "bedrock_concurrency",
        "publication_scope",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(inputs)
    assert ODD_LABELER_VERSION == "odd_dataset_labeler_v6"
    assert ODD_SOURCE_POLICY_VERSIONS["gnss_ins"] == "odd_gnss_ins_policy_v2"
    assert ODD_SOURCE_POLICY_VERSIONS["vlm"] == "odd_road_vlm_policy_v4"
    assert ODD_SOURCE_POLICY_VERSIONS["image_qc"] == "odd_image_qc_policy_v2"


def test_dataset_labeler_has_dedicated_launch_plan() -> None:
    assert odd_dataset_labeler_launch_plan.name == "odd-dataset-labeler"
    assert odd_dataset_labeler_launch_plan.workflow == wf_generate_odd_labelset
    assert odd_dataset_labeler_launch_plan.fixed_inputs.literals == {}


def test_source_labelers_have_independent_semantic_interfaces() -> None:
    assert set(resolve_odd_scenes.python_interface.outputs) == {
        "descriptors",
        "capability_manifest_json",
    }
    for source_task in (
        label_odd_map_route,
        label_odd_kinematics,
        label_odd_image_quality,
        label_odd_visual,
        label_odd_bedrock_map,
    ):
        assert "capability_manifest_json" in (
            source_task.python_interface.inputs
        )
    for deterministic_task in (
        label_odd_map_route,
        label_odd_kinematics,
        label_odd_image_quality,
    ):
        assert {
            "road_vlm_model",
            "map_resolver_model_id",
        }.isdisjoint(deterministic_task.python_interface.inputs)
    assert {
        "map_route_file",
        "kinematics_file",
        "image_quality_file",
        "enabled_sources",
        "ontology_sha256",
        "labeler_config_sha256",
        "road_vlm_provider",
        "road_vlm_model",
        "road_vlm_model_revision",
        "road_vlm_prompt_bundle_sha256",
        "road_vlm_decoding_config_sha256",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(label_odd_visual.python_interface.inputs)
    assert "map_resolver_model_id" not in (
        label_odd_visual.python_interface.inputs
    )
    assert {
        "map_route_file",
        "enabled_sources",
        "ontology_sha256",
        "labeler_config_sha256",
        "map_resolver_provider",
        "map_resolver_model_id",
        "map_resolver_model_revision",
        "map_resolver_prompt_bundle_sha256",
        "map_resolver_decoding_config_sha256",
    }.issubset(label_odd_bedrock_map.python_interface.inputs)
    assert "road_vlm_model" not in (
        label_odd_bedrock_map.python_interface.inputs
    )
    assert {
        "capability_manifest_json",
        "map_route_file",
        "kinematics_file",
        "image_quality_file",
        "visual_file",
        "bedrock_map_file",
        "enabled_sources",
        "ontology_sha256",
        "labeler_config_sha256",
        "fusion_config_sha256",
        "calibration_bundle_sha256",
        "labeler_image_digest",
        "labeler_source_revision",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(fuse_odd_scene.python_interface.inputs)
    assert {
        "capability_manifest_json",
        "semantic_contract_json",
        "ontology_version",
        "ontology_sha256",
        "labeler_bundle_version",
        "labeler_config_uri",
        "labeler_config_sha256",
        "enabled_sources",
        "road_vlm_provider",
        "road_vlm_model",
        "road_vlm_model_revision",
        "road_vlm_prompt_bundle_sha256",
        "road_vlm_decoding_config_sha256",
        "map_resolver_provider",
        "map_resolver_model_id",
        "map_resolver_model_revision",
        "map_resolver_prompt_bundle_sha256",
        "map_resolver_decoding_config_sha256",
        "fusion_config_sha256",
        "calibration_bundle_sha256",
        "publication_prefix",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(publish_odd_labelset.python_interface.inputs)


def test_launcher_uses_full_rate_visual_sampling_contract() -> None:
    launcher = Path("Platform/buildspec-launch-odd-labeling.yml").read_text(
        encoding="utf-8"
    )

    assert 'CAMERA_ANCHOR_INTERVAL_S: "1.0"' in launcher
    assert 'MAXIMUM_CAMERA_ANCHORS: "128"' in launcher
    assert 'TRIGGER_CONTEXT_S: "1.0"' in launcher
    assert 'REFINEMENT_CONFIDENCE_THRESHOLD: "0.65"' in launcher
    assert '"trigger_context_s"' in launcher
    assert '"refinement_confidence_threshold"' in launcher
    assert "odd_labeler_config_document" in launcher
    assert 'IfNoneMatch="*"' in launcher
    assert all(
        name in launcher
        for name in {
            '"ontology_version"',
            '"ontology_sha256"',
            '"labeler_bundle_version"',
            '"labeler_config_uri"',
            '"labeler_config_sha256"',
            '"enabled_sources"',
            '"road_vlm_provider"',
            '"road_vlm_model"',
            '"road_vlm_model_revision"',
            '"road_vlm_prompt_bundle_sha256"',
            '"road_vlm_decoding_config_sha256"',
            '"map_resolver_provider"',
            '"map_resolver_model_id"',
            '"map_resolver_model_revision"',
            '"map_resolver_prompt_bundle_sha256"',
            '"map_resolver_decoding_config_sha256"',
            '"fusion_config_sha256"',
            '"calibration_bundle_sha256"',
            '"publication_prefix"',
        }
    )
    assert '"openai_model"' not in launcher
    assert '"bedrock_map_model_id"' not in launcher
    assert "odd_dataset_labeler_launch_plan" in launcher
