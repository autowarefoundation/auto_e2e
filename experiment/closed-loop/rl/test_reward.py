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


def _scorer(left_heading: int = 1, tracks: list[Track] = (), recorded_y: float = 0.0,
            lanes: list[Lane] | None = None) -> SceneScorer:
    times = np.arange(0, 11_000_001, 100_000, dtype=np.int64)
    xy = np.column_stack([10.0 * times / 1e6, np.full(len(times), recorded_y)])
    lanes = lanes if lanes is not None else [_lane(0.0, 1), _lane(3.5, left_heading)]
    return SceneScorer(lanes, [], list(tracks), EGO, times, xy, np.zeros(len(times)))


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
    return np.column_stack([x, y, heading]), speeds, controls, np.array([x0, 0.0, 0.0]), speed


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


def test_a_car_sideswiping_the_ego_in_its_lane_is_not_at_fault() -> None:
    times = np.array([T0_US, T0_US + 10_000_000])
    # Drifts from the left lane into the ego's while level with it.
    swiper = Track(Box(4.5, 2.0), times, np.array([[11.4, 3.5, 0.0], [111.4, -6.5, 0.0]]), road_user=True)
    assert _scorer(tracks=[swiper]).score(*_plan(), T0_US)["nc"] == 1.0


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
    # Brakes at 8 m/s^2 and stops with its front 0.1 m short of the car.
    terms = _scorer(tracks=[ahead]).score(*_plan(accel=-8.0), T0_US)
    assert terms["nc"] == 1.0 and terms["ttc"] == 0.5


def test_leaving_the_road_is_gated_unless_the_recording_does_too() -> None:
    terms = _scorer().score(*_plan(y_end=-2.5), T0_US)
    assert terms["dac"] == 0.0 and terms["gated"] == 1.0
    # The recorded drive itself runs there, off the mapped lanes: the map is incomplete.
    assert _scorer(recorded_y=-2.5).score(*_plan(y_end=-2.5), T0_US)["dac"] == 1.0


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
    plan, speeds, controls, start, speed = _plan()
    with pytest.raises(ValueError):
        _scorer().score(plan, speeds, controls, start, speed, 7_500_000)  # 3.5 s left


def test_gated_plans_keep_the_order_of_their_first_violation() -> None:
    scorer = _scorer()
    assert scorer.score(*_plan(), T0_US)["safe_fraction"] == 1.0
    sooner = scorer.score(*_plan(y_end=-6.0), T0_US)
    later = scorer.score(*_plan(y_end=-2.5), T0_US)
    assert sooner["gated"] == later["gated"] == 1.0
    assert 0.0 < sooner["safe_fraction"] < later["safe_fraction"] < 1.0


def test_human_filter_matches_by_place_not_by_time() -> None:
    """A gap in the lane polygons that the recording drives through is forgiven at any speed."""
    gap = Polygon([(40, -1.75), (46, -1.75), (46, -0.5), (40, -0.5)])
    narrowed = Lane(_lane(0.0, 1).polygon.difference(gap), _lane(0.0, 1).centreline)
    scorer = _scorer(lanes=[narrowed, _lane(3.5, 1)])
    for speed in (10.0, 8.0, 6.0):
        assert scorer.score(*_plan(speed=speed), T0_US)["dac"] == 1.0


def test_lane_keeping_is_forgiven_where_the_recording_is_off_centre() -> None:
    assert _scorer(recorded_y=-1.5).score(*_plan(y_end=-1.5), T0_US)["lk"] == 1.0
    assert _scorer().score(*_plan(y_end=-1.5), T0_US)["lk"] < 1.0


def test_comfort_sees_the_step_out_of_t0() -> None:
    plan, speeds, controls, start, _ = _plan(speed=19.0)  # holds 19 m/s ...
    terms = _scorer().score(plan, speeds, controls, start, 20.0, T0_US)  # ... after 20 m/s at t0
    assert terms["comfort"] < 1.0


def test_human_filter_holds_between_recorded_samples() -> None:
    """A plan on the recorded line, starting between two recorded samples, is still forgiven."""
    gap = Polygon([(40, -1.75), (46, -1.75), (46, -0.5), (40, -0.5)])
    narrowed = Lane(_lane(0.0, 1).polygon.difference(gap), _lane(0.0, 1).centreline)
    scorer = _scorer(lanes=[narrowed, _lane(3.5, 1)])
    for x0 in (10.05, 10.5, 10.95):
        assert scorer.score(*_plan(x0=x0), T0_US)["dac"] == 1.0


def test_a_moving_track_is_not_stopped_at_its_last_sample() -> None:
    times = np.array([0, 2_000_000])
    track = Track(Box(4.5, 2.0), times, np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), road_user=True)
    assert track.speed(2_000_000) == pytest.approx(10.0)
    assert track.speed(0) == pytest.approx(10.0)


def test_progress_follows_the_log_from_the_ego_s_place() -> None:
    """The log stopped 30 m ahead of where the ego is now; from the ego's place it was driving."""
    times = np.arange(0, 11_000_001, 100_000, dtype=np.int64)
    x = np.minimum(10.0 * times / 1e6, 40.0)  # reaches x = 40 m at 4 s, then stops
    scorer = SceneScorer([_lane(0.0, 1), _lane(3.5, 1)], [], [], EGO, times,
                         np.column_stack([x, np.zeros(len(times))]), np.zeros(len(times)))
    t0 = 6_000_000  # the log is stopped at x = 40 m; the ego is still at x = 10 m
    assert scorer.score(*_plan(speed=0.0), t0)["ep"] == 0.0
    assert scorer.score(*_plan(speed=7.5), t0)["ep"] == pytest.approx(1.0)


def test_a_track_shorter_than_a_tenth_of_a_second_has_no_speed() -> None:
    stub = Track(Box(4.5, 2.0), np.array([0, 50_000]), np.zeros((2, 3)), road_user=True)
    assert stub.speed(25_000) == 0.0


def test_waiting_where_the_log_waited_earns_no_progress_after_it_left() -> None:
    """The log waited at x = 40 m from 4 s to 8 s; the ego stands just short of it later."""
    times = np.arange(0, 15_000_001, 100_000, dtype=np.int64)
    t = times / 1e6
    x = np.where(t < 4, 10 * t, np.where(t < 8, 40.0, 40 + 10 * (t - 8)))
    scorer = SceneScorer([_lane(0.0, 1), _lane(3.5, 1)], [], [], EGO, times,
                         np.column_stack([x, np.zeros(len(times))]), np.zeros(len(times)))
    stay = scorer.score(*_plan(speed=0.0, x0=39.5), 9_000_000)
    drive = scorer.score(*_plan(speed=10.0, x0=39.5), 9_000_000)
    assert stay["ep"] == 0.0 and drive["ep"] == pytest.approx(1.0)
