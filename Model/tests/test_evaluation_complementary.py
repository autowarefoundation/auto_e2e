"""Tests for the complementary open-loop evaluation pieces (#66).

These cover only what this PR *adds* on top of the existing
``compute_open_loop_metrics`` / ``gate_check`` already in ``evaluation``:
comfort metrics from (a,κ), an off-road proxy, training-free baselines, and
validation splits. Pure-numpy, no model or dataset needed.
"""

import numpy as np
import pytest

from evaluation import (
    COMFORT_THRESHOLDS,
    compute_comfort_metrics,
    compute_open_loop_metrics,
    constant_velocity_baseline,
    episode_range_split,
    geographic_holdout_split,
    hold_last_action_baseline,
    offroad_rate,
)

B, T = 4, 64


def _zeros():
    return np.zeros((B, T)), np.zeros((B, T))


# ---- comfort (#66 §3) ------------------------------------------------------

def test_comfort_smooth_trajectory_no_violations():
    a, k = _zeros()
    m = compute_comfort_metrics(a, k, np.full(B, 10.0))
    assert m["comfort_violation_rate"] == 0.0
    assert m["max_lon_jerk"] == 0.0 and m["max_yaw_rate"] == 0.0


def test_comfort_aggressive_trajectory_violates():
    # Alternating hard accel → huge longitudinal jerk.
    a = np.tile(np.array([5.0, -5.0]), (B, T // 2))
    k = np.zeros((B, T))
    m = compute_comfort_metrics(a, k, np.full(B, 10.0))
    assert m["max_lon_jerk"] > COMFORT_THRESHOLDS["lon_jerk"]
    assert m["lon_jerk_violation_rate"] == 1.0
    assert m["comfort_violation_rate"] == 1.0


def test_comfort_high_curvature_violates_lateral():
    a, k = np.zeros((B, T)), np.full((B, T), 0.2)   # tight curve at speed
    m = compute_comfort_metrics(a, k, np.full(B, 10.0))
    assert m["max_lat_accel"] > COMFORT_THRESHOLDS["lat_accel"]
    assert m["max_yaw_rate"] > COMFORT_THRESHOLDS["yaw_rate"]


# ---- off-road proxy (#66 §2) ----------------------------------------------

def test_offroad_rate_all_drivable_is_zero():
    positions = np.random.randn(B, T, 2) * 2.0
    mask = np.ones((64, 64), dtype=bool)
    assert offroad_rate(positions, mask, meters_per_pixel=0.5) == 0.0


def test_offroad_rate_nondrivable_is_one():
    positions = np.ones((B, T, 2)) * 5.0            # drive well forward/left
    mask = np.zeros((64, 64), dtype=bool)           # nothing drivable
    assert offroad_rate(positions, mask, meters_per_pixel=0.5) == 1.0


# ---- baselines (#66 §5) ----------------------------------------------------

def test_constant_velocity_baseline_is_zeros():
    a, k = constant_velocity_baseline(B, T)
    assert a.shape == (B, T) and np.all(a == 0) and np.all(k == 0)


def test_hold_last_action_baseline_tiles():
    a, k = hold_last_action_baseline(np.array([1.0, -2.0]), np.array([0.1, 0.0]), T)
    assert a.shape == (2, T)
    assert np.all(a[0] == 1.0) and np.all(a[1] == -2.0)
    assert np.all(k[0] == 0.1)


def test_baseline_scored_by_existing_metrics_runs():
    # Baselines must plug straight into the existing compute_open_loop_metrics.
    a, k = constant_velocity_baseline(B, T)
    gt_a, gt_k = np.full((B, T), 0.3), np.zeros((B, T))
    m = compute_open_loop_metrics(a, k, gt_a, gt_k, np.full(B, 6.0))
    assert m["ADE@6.4s"] > 0.0  # const-vel diverges from an accelerating gt


# ---- splits (#66 §4) -------------------------------------------------------

def test_episode_range_split():
    train, val = episode_range_split(100, val_fraction=0.1)
    assert len(val) == 10 and len(train) == 90
    assert set(train).isdisjoint(val)
    assert max(train) < min(val)              # val is the tail
    with pytest.raises(ValueError):
        episode_range_split(100, val_fraction=1.5)


def test_geographic_holdout_split():
    cities = ["berlin", "munich", "berlin", "hamburg", "munich"]
    train, val = geographic_holdout_split(cities, holdout_cities=["munich"])
    assert val == [1, 4] and train == [0, 2, 3]    # all munich episodes held out
