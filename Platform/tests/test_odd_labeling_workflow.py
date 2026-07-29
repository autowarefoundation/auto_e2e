from __future__ import annotations

from pathlib import Path

import pytest

from Platform.pipelines.odd_labeling_workflow import (
    ODD_LABELER_VERSION,
    _publication_scope,
    _put_immutable,
    _scene_summary,
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


def test_workflow_interface_does_not_expose_endpoint_url() -> None:
    assert "openai_base_url" not in wf_generate_odd_labelset.python_interface.inputs
    assert "publish_latest" not in wf_generate_odd_labelset.python_interface.inputs
    assert {
        "bedrock_map_model_id",
        "bedrock_map_model_revision",
        "labeler_image_digest",
        "labeler_source_revision",
        "openai_concurrency",
        "bedrock_concurrency",
        "publication_scope",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(wf_generate_odd_labelset.python_interface.inputs)
    assert ODD_LABELER_VERSION == "odd_dataset_labeler_v5"


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
    assert "openai_model" not in label_odd_map_route.python_interface.inputs
    assert "openai_model" not in label_odd_kinematics.python_interface.inputs
    assert "openai_model" not in (
        label_odd_image_quality.python_interface.inputs
    )
    assert {
        "map_route_file",
        "kinematics_file",
        "image_quality_file",
        "openai_model",
        "openai_model_revision",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(label_odd_visual.python_interface.inputs)
    assert "bedrock_map_model_id" not in (
        label_odd_visual.python_interface.inputs
    )
    assert {
        "bedrock_map_model_id",
        "bedrock_map_model_revision",
        "map_route_file",
    }.issubset(label_odd_bedrock_map.python_interface.inputs)
    assert "openai_model" not in (
        label_odd_bedrock_map.python_interface.inputs
    )
    assert {
        "capability_manifest_json",
        "map_route_file",
        "kinematics_file",
        "image_quality_file",
        "visual_file",
        "bedrock_map_file",
        "labeler_image_digest",
        "labeler_source_revision",
        "camera_anchor_interval_s",
        "maximum_camera_anchors",
        "trigger_context_s",
        "refinement_confidence_threshold",
    }.issubset(fuse_odd_scene.python_interface.inputs)
    assert {
        "bedrock_map_model_id",
        "bedrock_map_model_revision",
        "capability_manifest_json",
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
    assert "odd_dataset_labeler_launch_plan" in launcher
