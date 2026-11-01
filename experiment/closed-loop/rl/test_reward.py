"""Tests for the gated driving reward on a synthetic straight road.

Lane "right" runs along +x at y = 0, lane "left" at y = 3.5 m; both 3.5 m wide.
The recorded ego drives 10 m/s along y = 0 for 10 s.

    python -m pytest experiment/closed-loop/rl/test_reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward import Box, Lane, SceneScorer, Track  # noqa: E402

EGO = Box(length=5.0, width=2.0, centre_offset=1.4)
T0_US = 1_000_000


def _lane(y: float, heading: int) -> Lane:
    x = np.linspace(-50.0, 250.0, 61)[:: heading]
    polygon = Polygon([(-50, y - 1.75), (250, y - 1.75), (250, y + 1.75), (-50, y + 1.75)])
    return Lane(polygon, np.column_stack([x, np.full_like(x, y)]))


def _scorer(left_heading: int = 1, tracks: list[Track] = ()) -> SceneScorer:
    times = np.arange(0, 11_000_001, 100_000, dtype=np.int64)
    xy = np.column_stack([10.0 * times / 1e6, np.zeros(len(times))])
    return SceneScorer([_lane(0.0, 1), _lane(3.5, left_heading)], list(tracks), EGO, times, xy, np.zeros(len(times)))


def _plan(y_end: float = 0.0, speed: float = 10.0, accel: float = 0.0) -> tuple[np.ndarray, ...]:
    """64 steps from x = 10 m (the recording at t0), easing to lateral offset ``y_end``."""
    t = np.arange(1, 65) * 0.1
    speeds = np.maximum(speed + accel * t, 0.0)
    x = 10.0 + np.cumsum(speeds * 0.1)
    y = y_end * np.clip(t / 3.0, 0, 1) ** 2 * (3 - 2 * np.clip(t / 3.0, 0, 1))
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    controls = np.column_stack([np.full(64, accel), np.zeros(64)])
    return np.column_stack([x, y, heading]), speeds, controls


def test_following_the_recording_scores_one() -> None:
    terms = _scorer().score(*_plan(), T0_US)
    assert terms["reward"] == pytest.approx(1.0)
    assert terms["gated"] == 0.0


def test_stopping_scores_zero_progress() -> None:
    terms = _scorer().score(*_plan(speed=0.0), T0_US)
    assert terms["ep"] == 0.0 and terms["reward"] == 0.0
    assert terms["nc"] == 1.0  # stopped: nothing is at fault


def test_driving_faster_than_the_recording_earns_no_extra_progress() -> None:
    assert _scorer().score(*_plan(speed=14.0), T0_US)["ep"] == 1.0


def test_hitting_a_stopped_car_ahead_is_gated() -> None:
    times = np.array([0, 11_000_000])
    parked = Track(Box(4.5, 2.0), times, np.array([[40.0, 0.0, 0.0], [40.0, 0.0, 0.0]]), is_static=False)
    terms = _scorer(tracks=[parked]).score(*_plan(), T0_US)
    assert terms["nc"] == 0.0 and terms["gated"] == 1.0 and terms["reward"] == 0.0
    static = Track(parked.box, times, parked.poses, is_static=True)
    assert _scorer(tracks=[static]).score(*_plan(), T0_US)["nc"] == 0.5


def test_being_rear_ended_is_not_at_fault() -> None:
    times = np.array([T0_US, T0_US + 4_000_000])
    # Starts 8 m behind the ego's rear axle and closes in at 14 m/s.
    follower = Track(Box(4.5, 2.0), times, np.array([[2.0, 0.0, 0.0], [58.0, 0.0, 0.0]]), is_static=False)
    terms = _scorer(tracks=[follower]).score(*_plan(), T0_US)
    assert terms["nc"] == 1.0


def test_time_to_collision_halves_the_reward() -> None:
    # A car crossing 1 m beyond where the ego's box reaches 0.6 s later.
    times = np.array([0, 11_000_000])
    ahead = Track(Box(4.5, 2.0), times, np.array([[22.5, 0.0, 0.0], [22.5, 0.0, 0.0]]), is_static=False)
    plan, speeds, controls = _plan()
    plan, speeds, controls = plan[:5], speeds[:5], controls[:5]  # stops planning before contact
    terms = _scorer(tracks=[ahead]).score(plan, speeds, controls, T0_US)
    assert terms["nc"] == 1.0 and terms["ttc"] == 0.5


def test_leaving_the_road_is_gated_unless_the_recording_does_too() -> None:
    terms = _scorer().score(*_plan(y_end=-2.5), T0_US)
    assert terms["dac"] == 0.0 and terms["gated"] == 1.0
    # The recorded drive itself leaves the mapped lanes: the map is incomplete there.
    off_map = _scorer()
    off_map.recorded_xy = off_map.recorded_xy + [0.0, -3.0]
    assert off_map.score(*_plan(y_end=-2.5), T0_US)["dac"] == 1.0


def test_lane_change_keeps_the_road_but_loses_some_lane_keeping() -> None:
    terms = _scorer().score(*_plan(y_end=2.0), T0_US)
    assert terms["dac"] == 1.0
    assert 0.0 < terms["lk"] < 1.0


def test_renderer_bound_fades_from_2_5_to_3_m() -> None:
    assert _scorer().score(*_plan(y_end=2.75), T0_US)["valid"] == pytest.approx(0.5, abs=0.02)
    assert _scorer().score(*_plan(y_end=3.5), T0_US)["valid"] == 0.0


def test_driving_into_the_oncoming_lane_is_gated() -> None:
    oncoming = _scorer(left_heading=-1)
    terms = oncoming.score(*_plan(y_end=3.5), T0_US)
    assert terms["ddc"] == 0.0
    assert _scorer().score(*_plan(y_end=3.5), T0_US)["ddc"] == 1.0


def test_harsh_braking_costs_comfort() -> None:
    terms = _scorer().score(*_plan(accel=-6.0), T0_US)
    assert terms["comfort"] == pytest.approx(1.0 - 0.5 / 6.0)
