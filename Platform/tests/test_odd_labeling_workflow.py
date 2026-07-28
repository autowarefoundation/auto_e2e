from __future__ import annotations

from Platform.pipelines.odd_labeling_workflow import (
    _publication_scope,
    _scene_summary,
    _statistics,
    _union_duration,
    label_odd_scene,
    publish_odd_labelset,
    resolve_odd_scenes,
    wf_generate_odd_labelset,
)


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
            "observations": [
                {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "source": "vlm",
                    "start_timestamp_ns": 0,
                    "end_timestamp_ns": 100,
                },
                {
                    "key": "odd.environment.sky",
                    "status": "valid",
                    "values": ["clear"],
                    "source": "fusion",
                    "start_timestamp_ns": 20,
                    "end_timestamp_ns": 80,
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

    assert row["valid_duration_ns"] == 100
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


def test_workflow_interface_does_not_expose_endpoint_url() -> None:
    assert "openai_base_url" not in wf_generate_odd_labelset.python_interface.inputs
    assert "publish_latest" not in wf_generate_odd_labelset.python_interface.inputs
    assert {
        "labeler_image_digest",
        "labeler_source_revision",
        "publication_scope",
    }.issubset(wf_generate_odd_labelset.python_interface.inputs)


def test_capability_manifest_is_required_across_publication_tasks() -> None:
    assert set(resolve_odd_scenes.python_interface.outputs) == {
        "descriptors",
        "capability_manifest_json",
    }
    assert {
        "capability_manifest_json",
        "labeler_image_digest",
        "labeler_source_revision",
    }.issubset(label_odd_scene.python_interface.inputs)
    assert (
        "capability_manifest_json"
        in publish_odd_labelset.python_interface.inputs
    )
