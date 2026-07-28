from __future__ import annotations

import pytest

from data_processing.odd_labeling.statistics import (
    build_statistics,
    union_duration,
)


SECOND = 1_000_000_000


def _observation(
    uid: str,
    key: str,
    value: str | None,
    start_s: int,
    end_s: int,
    *,
    status: str = "valid",
    confidence: float = 0.9,
    source: str = "fusion",
    measurements: dict | None = None,
) -> dict:
    return {
        "observation_uid": uid,
        "scene_uid": "",
        "key": key,
        "status": status,
        "values": [value] if value is not None else [],
        "confidence": confidence,
        "source": source,
        "start_timestamp_ns": start_s * SECOND,
        "end_timestamp_ns": end_s * SECOND,
        "measurements": measurements or {},
        "conflicting_evidence_uids": [],
    }


def _records() -> list[dict]:
    scene_a = {
        "scene_uid": "scene-a",
        "start_timestamp_ns": 0,
        "end_timestamp_ns": 10 * SECOND,
        "distance_m": 100.0,
        "evidence": [
            {"label_key": "odd.road.context", "status": "valid"},
            {"label_key": "odd.environment.sky", "status": "valid"},
            {"label_key": "event.ego.maneuver", "status": "valid"},
        ],
        "observations": [
            _observation(
                "speed-a-1",
                "odd.ego.speed_bin",
                "low_speed",
                0,
                5,
                source="gnss_ins",
                measurements={"ego_speed_mps": 5.0},
            ),
            _observation(
                "speed-a-2",
                "odd.ego.speed_bin",
                "medium_speed",
                5,
                10,
                source="gnss_ins",
                measurements={"ego_speed_mps": 15.0},
            ),
            _observation(
                "road-a-1",
                "odd.road.context",
                "urban",
                0,
                5,
            ),
            _observation(
                "road-a-2",
                "odd.road.context",
                "rural",
                5,
                10,
            ),
            _observation(
                "sky-a-1",
                "odd.environment.sky",
                "clear",
                0,
                10,
                source="vlm",
            ),
            _observation(
                "sky-a-camera-overlap",
                "odd.environment.sky",
                "clear",
                2,
                8,
                source="vlm",
            ),
            _observation(
                "event-a",
                "event.ego.maneuver",
                "turn_left",
                4,
                6,
            ),
        ],
        "events": [
            {
                "event_uid": "event-instance-a",
                "primary_event_key": "event.ego.maneuver",
                "start_timestamp_ns": 4 * SECOND,
                "end_timestamp_ns": 6 * SECOND,
                "observation_uids": ["event-a"],
                "provenance": {"primary_values": ["turn_left"]},
            }
        ],
    }
    scene_b = {
        "scene_uid": "scene-b",
        "start_timestamp_ns": 0,
        "end_timestamp_ns": 10 * SECOND,
        "distance_m": 20.0,
        "evidence": [
            {"label_key": "odd.road.context", "status": "valid"},
            {
                "label_key": "odd.environment.sky",
                "status": "not_observable",
            },
        ],
        "observations": [
            _observation(
                "road-b",
                "odd.road.context",
                "urban",
                0,
                10,
            ),
            _observation(
                "sky-b",
                "odd.environment.sky",
                None,
                0,
                10,
                status="not_observable",
                confidence=1.0,
                source="vlm",
            ),
        ],
        "events": [],
    }
    for record in (scene_a, scene_b):
        for observation in record["observations"]:
            observation["scene_uid"] = record["scene_uid"]
    return [scene_a, scene_b]


def _ontology() -> dict:
    return {
        "labels": [
            {
                "key": "odd.road.context",
                "namespace": "odd",
                "quality_tier": "experimental",
                "values": [{"value": "urban"}, {"value": "rural"}],
            },
            {
                "key": "odd.environment.sky",
                "namespace": "odd",
                "quality_tier": "experimental",
                "values": [{"value": "clear"}, {"value": "overcast"}],
            },
            {
                "key": "event.ego.maneuver",
                "namespace": "event",
                "quality_tier": "experimental",
                "values": [
                    {"value": "turn_left"},
                    {"value": "turn_right"},
                ],
            },
        ]
    }


def _key(statistics: dict, key: str) -> dict:
    return next(row for row in statistics["keys"] if row["key"] == key)


def _value(key: dict, value: str) -> dict:
    return next(row for row in key["values"] if row["value"] == value)


def test_union_duration_deduplicates_overlapping_intervals() -> None:
    assert union_duration([(0, 10), (2, 8), (10, 12)]) == 12


def test_statistics_use_scene_duration_and_trajectory_distance() -> None:
    statistics = build_statistics(_records(), _ontology(), "oddls-test")
    road = _key(statistics, "odd.road.context")
    urban = _value(road, "urban")
    rural = _value(road, "rural")
    sky = _key(statistics, "odd.environment.sky")

    assert statistics["schema_version"] == "odd_statistics_v2"
    assert statistics["scene_duration_ns"] == 20 * SECOND
    assert statistics["scene_distance_m"] == pytest.approx(120.0)
    assert statistics["distance_weighting"]["method_scene_counts"] == {
        "duration_proportional": 1,
        "speed_integrated_scene_normalized": 1,
    }
    assert road["valid_interval_count"] == 2
    assert road["valid_duration_ns"] == 20 * SECOND
    assert road["valid_distance_m"] == pytest.approx(120.0)
    assert urban["scene_count"] == 2
    assert urban["scene_ratio"] == 1.0
    assert urban["duration_ns"] == 15 * SECOND
    assert urban["duration_ratio"] == 0.75
    assert urban["distance_m"] == pytest.approx(45.0)
    assert urban["distance_ratio"] == pytest.approx(0.375)
    assert rural["duration_ratio"] == 0.25
    assert rural["distance_ratio"] == pytest.approx(0.625)
    assert urban["scene_ratio_ci95"]["method"] == "wilson_scene_95"
    assert urban["distance_ratio_ci95"]["replicates"] == 256

    assert sky["valid_duration_ns"] == 10 * SECOND
    assert sky["valid_distance_m"] == pytest.approx(100.0)
    assert sky["observable_scene_coverage"] == 0.5
    assert sky["status_duration_ns"]["not_observable"] == 10 * SECOND
    assert _value(sky, "clear")["valid_interval_count"] == 1


def test_statistics_count_events_and_interval_cooccurrence() -> None:
    statistics = build_statistics(_records(), _ontology(), "oddls-test")
    maneuver = _key(statistics, "event.ego.maneuver")
    turn_left = _value(maneuver, "turn_left")

    assert turn_left["event_instance_count"] == 1
    assert turn_left["confidence"]["observation_count"] == 1

    odd_pairs = statistics["cooccurrences"]["odd_pairs"]
    urban_clear = next(
        row
        for row in odd_pairs
        if row["left_value"] == "clear" and row["right_value"] == "urban"
    )
    assert urban_clear["scene_count"] == 1
    assert urban_clear["overlap_duration_ns"] == 5 * SECOND
    assert urban_clear["overlap_distance_m"] == pytest.approx(25.0)

    odd_event = statistics["cooccurrences"]["odd_event"]
    urban_turn = next(
        row
        for row in odd_event
        if row["odd_value"] == "urban"
        and row["event_value"] == "turn_left"
    )
    rural_turn = next(
        row
        for row in odd_event
        if row["odd_value"] == "rural"
        and row["event_value"] == "turn_left"
    )
    assert urban_turn["event_instance_count"] == 1
    assert urban_turn["overlap_duration_ns"] == SECOND
    assert urban_turn["overlap_distance_m"] == pytest.approx(5.0)
    assert rural_turn["overlap_distance_m"] == pytest.approx(15.0)


def test_statistics_are_order_independent_and_reject_duplicate_scenes() -> None:
    records = _records()

    assert build_statistics(
        records,
        _ontology(),
        "oddls-test",
    ) == build_statistics(
        list(reversed(records)),
        _ontology(),
        "oddls-test",
    )

    with pytest.raises(ValueError, match="unique scene"):
        build_statistics(
            [records[0], records[0]],
            _ontology(),
            "oddls-test",
        )
