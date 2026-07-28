from __future__ import annotations

import numpy as np
import pytest

from data_processing.odd_labeling.deterministic import _interval_slices
from data_processing.odd_labeling.ontology import (
    LABEL_STATUSES,
    ONTOLOGY,
    ontology_document,
)
from data_processing.odd_labeling.schema import (
    coalesce_observations,
    make_observation,
)


def test_interval_slices_carry_forward_final_sample() -> None:
    timestamps = np.array(
        [index * 100_000_000 for index in range(20)]
        + [1_950_000_000],
        dtype=np.int64,
    )

    slices = _interval_slices(timestamps)

    assert slices[-1] == (2_000_000_000, 2_050_000_000, 20, 21)


def test_ontology_contains_complete_scene_label_catalog() -> None:
    counts = {"odd": 0, "event": 0, "perception": 0}
    for definition in ONTOLOGY.values():
        counts[definition.namespace] += 1

    assert counts == {"odd": 32, "event": 13, "perception": 21}
    assert tuple(LABEL_STATUSES) == (
        "valid",
        "unavailable",
        "not_observable",
        "ambiguous",
    )
    document = ontology_document()
    assert len(document["ontology_sha256"]) == 64
    assert len(document["labels"]) == 66


def test_route_plan_and_actual_maneuver_remain_distinct() -> None:
    planned = ONTOLOGY["odd.route.action"]
    actual = ONTOLOGY["event.ego.maneuver"]

    assert "turn_left" in planned.values
    assert "turn_left" in actual.values
    assert planned.primary_sources == ("map_route",)
    assert "gnss_ins" in actual.primary_sources


def test_non_valid_status_cannot_encode_none() -> None:
    with pytest.raises(ValueError, match="must not carry resolved values"):
        make_observation(
            scene_uid="scene-1",
            key="odd.road.workzone_state",
            status="not_observable",
            values=("none",),
            confidence=0.0,
            source="vlm",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
        )


def test_none_and_normal_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="none cannot coexist"):
        make_observation(
            scene_uid="scene-1",
            key="odd.road.workzone_state",
            status="valid",
            values=("none", "cones"),
            confidence=0.8,
            source="vlm",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
        )
    with pytest.raises(ValueError, match="normal cannot coexist"):
        make_observation(
            scene_uid="scene-1",
            key="perception.visual.lighting",
            status="valid",
            values=("normal", "backlit"),
            confidence=0.8,
            source="image_qc",
            start_timestamp_ns=1,
            end_timestamp_ns=2,
            camera_id="front",
        )


def test_speed_observation_retains_continuous_measurement() -> None:
    observation = make_observation(
        scene_uid="scene-1",
        key="odd.ego.speed_bin",
        status="valid",
        values=("low_speed",),
        confidence=0.99,
        source="gnss_ins",
        start_timestamp_ns=1,
        end_timestamp_ns=2,
        measurements={"ego_speed_kph": 12.5},
    )

    assert observation.measurements["ego_speed_kph"] == 12.5


def test_adjacent_equal_observations_coalesce() -> None:
    observations = [
        make_observation(
            scene_uid="scene-1",
            key="odd.environment.sky",
            status="valid",
            values=("clear",),
            confidence=0.9,
            source="vlm",
            start_timestamp_ns=10,
            end_timestamp_ns=20,
        ),
        make_observation(
            scene_uid="scene-1",
            key="odd.environment.sky",
            status="valid",
            values=("clear",),
            confidence=0.8,
            source="vlm",
            start_timestamp_ns=20,
            end_timestamp_ns=30,
        ),
    ]

    merged = coalesce_observations(observations)

    assert len(merged) == 1
    assert merged[0].start_timestamp_ns == 10
    assert merged[0].end_timestamp_ns == 30
    assert merged[0].confidence == 0.8
