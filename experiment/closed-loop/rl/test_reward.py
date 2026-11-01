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


def _plan(
    y_end: float = 0.0, speed: float = 10.0, accel: float = 0.0, x0: float = 10.0
) -> tuple[np.ndarray, ...]:
    """Plan, speeds, controls and start pose: 64 steps from ``x0`` (the recording is at
    x = 10 m at t0), easing to lateral offset ``y_end``."""
    t = np.arange(1, 65) * 0.1
    speeds = np.maximum(speed + accel * t, 0.0)
    x = x0 + np.cumsum(speeds * 0.1)
    y = y_end * np.clip(t / 3.0, 0, 1) ** 2 * (3 - 2 * np.clip(t / 3.0, 0, 1))
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    controls = np.column_stack([np.full(64, accel), np.zeros(64)])
    return np.column_stack([x, y, heading]), speeds, controls, np.array([x0, 0.0, 0.0])


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
    parked = Track(Box(4.5, 2.0), times, np.array([[40.0, 0.0, 0.0], [40.0, 0.0, 0.0]]), road_user=True)
    terms = _scorer(tracks=[parked]).score(*_plan(), T0_US)
    assert terms["nc"] == 0.0 and terms["gated"] == 1.0 and terms["reward"] == 0.0
    obstacle = Track(parked.box, times, parked.poses, road_user=False)
    assert _scorer(tracks=[obstacle]).score(*_plan(), T0_US)["nc"] == 0.5


def test_being_rear_ended_is_not_at_fault() -> None:
    times = np.array([T0_US, T0_US + 4_000_000])
    # Starts 8 m behind the ego's rear axle and closes in at 14 m/s.
    follower = Track(Box(4.5, 2.0), times, np.array([[2.0, 0.0, 0.0], [58.0, 0.0, 0.0]]), road_user=True)
    terms = _scorer(tracks=[follower]).score(*_plan(), T0_US)
    assert terms["nc"] == 1.0


def test_time_to_collision_halves_the_reward() -> None:
    # A car crossing 1 m beyond where the ego's box reaches 0.6 s later.
    times = np.array([0, 11_000_000])
    ahead = Track(Box(4.5, 2.0), times, np.array([[22.5, 0.0, 0.0], [22.5, 0.0, 0.0]]), road_user=True)
    plan, speeds, controls, start = _plan()
    plan, speeds, controls = plan[:5], speeds[:5], controls[:5]  # stops planning before contact
    terms = _scorer(tracks=[ahead]).score(plan, speeds, controls, start, T0_US)
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
    # -4.5 m/s^2 from 20 m/s does not stop within 4 s: only the acceleration bound breaks.
    terms = _scorer().score(*_plan(speed=20.0, accel=-4.5), T0_US)
    assert terms["comfort"] == pytest.approx(1.0 - 0.5 / 6.0)


def test_commanded_braking_of_a_stopped_plan_is_not_felt() -> None:
    assert _scorer().score(*_plan(speed=0.0, accel=-5.0), T0_US)["comfort"] == 1.0


def test_cutting_into_a_car_alongside_is_at_fault() -> None:
    times = np.array([T0_US, T0_US + 10_000_000])
    # In the left lane, its centre 0.4 m behind the ego's rear axle at t0, at the ego's speed.
    alongside = Track(Box(4.5, 2.0), times, np.array([[9.6, 3.5, 0.0], [109.6, 3.5, 0.0]]), road_user=True)
    terms = _scorer(tracks=[alongside]).score(*_plan(y_end=3.5), T0_US)
    assert terms["nc"] == 0.0


def test_progress_is_measured_from_the_ego() -> None:
    ahead = _scorer().score(*_plan(x0=12.0), T0_US)["ep"]
    slower = _scorer().score(*_plan(x0=12.0, speed=9.5), T0_US)["ep"]
    assert ahead == pytest.approx(1.0) and slower == pytest.approx(0.95, abs=0.01)


def test_the_end_of_the_recording_is_rejected() -> None:
    plan, speeds, controls, start = _plan()
    with pytest.raises(ValueError):
        _scorer().score(plan, speeds, controls, start, 10_950_000)  # one step left
