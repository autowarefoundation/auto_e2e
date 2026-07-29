from __future__ import annotations

import numpy as np
import pytest

from data_processing.odd_labeling.deterministic import (
    KINEMATIC_KEYS,
    _speed_bin,
    label_kinematics,
)
from data_processing.odd_labeling.published_snapshot import (
    CameraObject,
    CanonicalSceneEvidence,
    PublishedSceneDescriptor,
)


EARTH_RADIUS_M = 6_371_008.8


def _scene(path: np.ndarray) -> CanonicalSceneEvidence:
    descriptor = PublishedSceneDescriptor(
        dataset_name="kinematics-fixture",
        dataset_version="v1",
        dataset_manifest_uri="s3://fixture/manifest.json",
        dataset_manifest_sha256="1" * 64,
        partition_id="partition-1",
        scene_uid="scene-1",
        source_uri="s3://fixture/scene-1",
        source_manifest_sha256="2" * 64,
        shard_name="scene-1.tar",
        camera_count=1,
        endpoint_exclusion_frames=0,
    )
    camera = CameraObject(
        frame_index=0,
        camera_index=0,
        camera_role="front_center",
        timestamp_ns=int(path[0, 3]),
        bucket="fixture",
        key="front.jpg",
        byte_size=1,
    )
    return CanonicalSceneEvidence(
        descriptor=descriptor,
        path_latlon_heading_timestamp=path,
        navigation_map=None,
        navigation_route=None,
        navigation_quality={},
        camera_objects=(camera,),
    )


def _northbound_path(
    timestamps_ns: np.ndarray,
    speed_mps: np.ndarray,
    *,
    headings_deg: np.ndarray | None = None,
) -> np.ndarray:
    seconds = timestamps_ns.astype(np.float64) / 1e9
    distances = np.zeros(len(timestamps_ns), dtype=np.float64)
    distances[1:] = np.cumsum(
        (speed_mps[:-1] + speed_mps[1:]) * 0.5 * np.diff(seconds)
    )
    latitudes = 49.0 + np.degrees(distances / EARTH_RADIUS_M)
    headings = (
        np.zeros(len(timestamps_ns), dtype=np.float64)
        if headings_deg is None
        else headings_deg
    )
    return np.column_stack(
        (
            latitudes,
            np.full(len(timestamps_ns), 8.0),
            headings,
            timestamps_ns,
        )
    )


def _observations_for_key(
    path: np.ndarray,
    key: str,
) -> list:
    return [
        item
        for item in label_kinematics(_scene(path))
        if item.key == key
    ]


@pytest.mark.parametrize(
    ("speed_kph", "stationary_dwell_met", "expected"),
    [
        (0.0, False, "creeping"),
        (0.0, True, "stationary"),
        (0.5, True, "stationary"),
        (0.500_001, True, "creeping"),
        (4.999_999, True, "creeping"),
        (5.0, True, "low_speed"),
        (29.999_999, True, "low_speed"),
        (30.0, True, "medium_speed"),
        (60.0, True, "medium_speed"),
        (60.000_001, True, "high_speed"),
    ],
)
def test_speed_bin_boundaries_are_explicit(
    speed_kph: float,
    stationary_dwell_met: bool,
    expected: str,
) -> None:
    assert (
        _speed_bin(
            speed_kph,
            stationary_dwell_met=stationary_dwell_met,
        )
        == expected
    )


def test_stationary_requires_dwell_before_stop_labels() -> None:
    short_timestamps = np.arange(6, dtype=np.int64) * 100_000_000
    short_path = _northbound_path(
        short_timestamps,
        np.zeros(len(short_timestamps)),
    )
    short = label_kinematics(_scene(short_path))

    assert next(item for item in short if item.key == "odd.ego.speed_bin").values == (
        "creeping",
    )
    assert next(
        item for item in short if item.key == "event.ego.motion_state"
    ).values == ("creeping",)
    assert next(
        item for item in short if item.key == "event.ego.maneuver"
    ).values == ("lane_follow",)

    dwell_timestamps = np.arange(11, dtype=np.int64) * 100_000_000
    dwell_path = _northbound_path(
        dwell_timestamps,
        np.zeros(len(dwell_timestamps)),
    )
    dwell = label_kinematics(_scene(dwell_path))

    assert next(
        item for item in dwell if item.key == "odd.ego.speed_bin"
    ).values == ("stationary",)
    assert next(
        item for item in dwell if item.key == "event.ego.motion_state"
    ).values == ("stopped",)
    assert next(
        item for item in dwell if item.key == "event.ego.maneuver"
    ).values == ("stop",)


def test_timestamp_gap_is_not_observable_and_does_not_leak_heading() -> None:
    timestamps = np.asarray(
        [
            0,
            100_000_000,
            200_000_000,
            2_000_000_000,
            2_100_000_000,
            2_200_000_000,
        ],
        dtype=np.int64,
    )
    headings = np.asarray([0.0, 0.0, 0.0, 180.0, 180.0, 180.0])
    path = _northbound_path(
        timestamps,
        np.full(len(timestamps), 2.0),
        headings_deg=headings,
    )
    observations = label_kinematics(_scene(path))

    missing = [item for item in observations if item.status == "not_observable"]
    assert len(missing) == 2 * len(KINEMATIC_KEYS)
    assert {item.key for item in missing} == set(KINEMATIC_KEYS)
    assert all(item.values == () for item in missing)
    assert all(item.provenance["reason"] == "timestamp_gap" for item in missing)
    assert all(item.provenance["maximum_gap_ns"] == 500_000_000 for item in missing)

    post_gap_maneuver = next(
        item
        for item in observations
        if item.key == "event.ego.maneuver" and item.start_timestamp_ns == 2_000_000_000
    )
    assert post_gap_maneuver.status == "valid"
    assert post_gap_maneuver.values == ("lane_follow",)


def test_actual_heading_change_produces_left_turn() -> None:
    timestamps = np.arange(31, dtype=np.int64) * 100_000_000
    headings = np.linspace(0.0, -45.0, len(timestamps))
    path = _northbound_path(
        timestamps,
        np.full(len(timestamps), 5.0),
        headings_deg=headings,
    )

    maneuvers = _observations_for_key(path, "event.ego.maneuver")

    assert all(item.status == "valid" for item in maneuvers)
    assert any(item.values == ("turn_left",) for item in maneuvers)


def test_sustained_deceleration_produces_hard_brake() -> None:
    timestamps = np.arange(51, dtype=np.int64) * 100_000_000
    seconds = timestamps.astype(np.float64) / 1e9
    speed = np.full(len(timestamps), 20.0)
    braking = (seconds >= 2.0) & (seconds <= 2.8)
    speed[braking] = 20.0 * (2.8 - seconds[braking]) / 0.8
    speed[seconds > 2.8] = 0.0
    path = _northbound_path(timestamps, speed)

    responses = _observations_for_key(path, "event.ego.strong_response")

    assert all(item.status == "valid" for item in responses)
    assert any(item.values == ("hard_brake",) for item in responses)
    assert responses[0].values == ("none",)
    braking_measurements = [
        item.measurements["longitudinal_acceleration_mps2"]
        for item in responses
        if item.values == ("hard_brake",)
    ]
    assert min(braking_measurements) < -3.0
