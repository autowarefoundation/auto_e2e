from __future__ import annotations

from Platform.pipelines.odd_labeling_workflow import (
    _scene_summary,
    _statistics,
    _union_duration,
    wf_generate_odd_labelset,
)


def test_union_duration_counts_overlapping_intervals_once() -> None:
    assert _union_duration([(10, 20), (15, 30), (40, 50)]) == 30


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
