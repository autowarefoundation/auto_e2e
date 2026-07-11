import torch
import numpy as np

from Tools.trajectory_visualization.kinematics import (
    accel_and_curv_to_meters_trajectory,
    get_cumulative_distances,
    get_trajectory_boundaries_3d
)

import pytest

_DT = 0.1
_FUTURE_TIMESTEPS = 64

def test_accel_and_curv_to_meters_trajectory_straight_no_accel():
    # 1. Create a dummy action sequence for going straight with no acceleration
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    current_speed = 10.0  # 10 m/s

    # 2. Run the function
    trajectory_m = accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    # 3. Assertions
    assert trajectory_m.shape == (_FUTURE_TIMESTEPS + 1, 2), "Shape of trajectory tensor is incorrect"
    # The car should move straight along the y-axis (forward)
    # X should be 0, Y should increase based on speed
    v = current_speed
    for i in range(1, _FUTURE_TIMESTEPS + 1):
        # Note: In the function, positive Y is up, positive X is right.
        assert trajectory_m[i, 0].item() == pytest.approx(0.0), "X should be 0"
        assert trajectory_m[i, 1].item() > trajectory_m[i-1, 1].item(), "Y should be increasing"
        assert trajectory_m[i, 1].item() == pytest.approx(trajectory_m[i-1, 1].item() + v * _DT), "Integration is incorrect"

def test_accel_and_curv_to_meters_trajectory_stationary():
    # Edge case: 0 speed, 0 acceleration -> Car should remain at origin (0, 0)
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    current_speed = 0.0

    trajectory_m = accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    for i in range(_FUTURE_TIMESTEPS + 1):
        assert trajectory_m[i, 0].item() == pytest.approx(0.0)
        assert trajectory_m[i, 1].item() == pytest.approx(0.0)

def test_accel_and_curv_to_meters_trajectory_constant_acceleration_from_standstill():
    # Edge case: starting from 0 speed, but applying constant acceleration
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    action_sequence[0::2] = 2.0  # Constant 2.0 m/s^2 acceleration (every even index is accel)
    current_speed = 0.0

    trajectory_m = accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    assert trajectory_m[0, 0].item() == pytest.approx(0.0)
    assert trajectory_m[0, 1].item() == pytest.approx(0.0)
    
    # Check that distance covered in each timestep is strictly increasing
    for i in range(2, _FUTURE_TIMESTEPS + 1):
        dist_prev = trajectory_m[i-1, 1].item() - trajectory_m[i-2, 1].item()
        dist_curr = trajectory_m[i, 1].item() - trajectory_m[i-1, 1].item()
        
        assert trajectory_m[i, 0].item() == pytest.approx(0.0), "X should be 0, no curvature applied"
        assert dist_curr > dist_prev, "Distance per timestep should increase under constant acceleration"

def test_accel_and_curv_to_meters_trajectory_turning():
    # Edge case: turning left with constant speed
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    action_sequence[1::2] = 0.1  # Constant positive curvature (left turn)
    current_speed = 10.0

    trajectory_m = accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    # After 64 timesteps, X should be negative (left of the starting Y-axis) and Y should be positive
    assert trajectory_m[-1, 0].item() < -0.1, "Car should have turned left (negative X)"
    assert trajectory_m[-1, 1].item() > 0.1, "Car should have moved forward (positive Y)"

def test_accel_and_curv_to_meters_trajectory_extreme_spiral():
    # Edge case: extreme spiral
    # Constant acceleration and linearly increasing curvature.
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    action_sequence[0::2] = 0.5  # Constant acceleration
    action_sequence[1::2] = torch.linspace(0.5, 1.0, _FUTURE_TIMESTEPS)  # Increasing curvature
    current_speed = 5.0

    trajectory_m = accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    assert not torch.isnan(trajectory_m).any(), "Trajectory contains NaNs"
    assert not torch.isinf(trajectory_m).any(), "Trajectory contains Infs"

    # A tight spiral with these parameters will complete multiple full 360-degree rotations.
    # This means the vehicle must travel "backwards" relative to its start at some point.
    assert trajectory_m[:, 1].min().item() < -0.5, "Car did not loop backwards significantly"

def test_get_cumulative_distances():
    traj = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [3.0, 2.0]
    ])
    dists = get_cumulative_distances(traj)
    expected_dists = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(dists, expected_dists)

def test_get_trajectory_boundaries_3d():
    traj = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0]
    ])
    left_bound, right_bound = get_trajectory_boundaries_3d(traj, width_m=2.0)
    
    # direction is [0, 1] (y-axis)
    # normal is [-1, 0] (x-axis)
    # left should be [-1, y], right should be [1, y]
    expected_left = torch.tensor([
        [-1.0, 0.0],
        [-1.0, 1.0],
        [-1.0, 2.0]
    ])
    expected_right = torch.tensor([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0]
    ])
    
    assert torch.allclose(left_bound, expected_left)
    assert torch.allclose(right_bound, expected_right)
