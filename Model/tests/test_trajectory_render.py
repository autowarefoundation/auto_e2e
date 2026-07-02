import sys
sys.path.append('..')
from visualization.trajectory_rendering import Visualization, _DT, _FUTURE_TIMESTEPS
import torch
import pytest
import cv2
import numpy as np
from pathlib import Path
import os

def test_visualization_with_dummy_data(tmp_path: Path):

    # 1. Create a dummy action sequence (64 timesteps * 2 signals = 128 flat)
    # Let's mock a constant acceleration and a slight left turn (positive curvature)
    mock_actions = torch.zeros(128)
    mock_actions = mock_actions.view(64, 2)
    mock_actions[:, 0] = 0.5  # Constant acceleration of 0.5 m/s^2
    mock_actions[:, 1] = 0.01  # Constant left curvature
    mock_actions = mock_actions.flatten()  # Flatten back to match network output

    # 2. Set baseline parameters
    mock_speed = 10.0  # Starting at 10 m/s (36 km/h)
    mock_radius = 800.0  # Just like in gps_to_map.py

    # 3. Create a clean mock map image, following L2D format
    mock_map = np.full((360, 640, 3), (17, 17, 17), dtype=np.uint8) # equivalent to #111111
    map_copy = mock_map.copy()

    print("Executing render_trajectory...")
    # Run the visualization function
    result_img = Visualization.render_trajectory_map_tile(
        action_sequence=mock_actions,
        current_speed=mock_speed,
        map_image=mock_map,
        radius_m=mock_radius
    )

    # 4. Save and inspect the result
    output_path = tmp_path / "output.png"
    cv2.imwrite(str(output_path), result_img)

    assert result_img is not None, "Visualization function returned None"
    assert isinstance(result_img, np.ndarray), "Visualization function did not return a numpy array"
    assert result_img.shape == mock_map.shape, "Shape does not match"
    assert np.array_equal(map_copy, mock_map), "Original image mutated"
    assert not np.array_equal(result_img, mock_map), "Image was not modified"
    assert os.path.isfile(output_path), "Image file was not created in the target directory"

def test_accel_and_curv_to_meters_trajectory_straight_no_accel():
    # 1. Create a dummy action sequence for going straight with no acceleration
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    current_speed = 10.0  # 10 m/s

    # 2. Run the function
    trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

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

    trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    for i in range(_FUTURE_TIMESTEPS + 1):
        assert trajectory_m[i, 0].item() == pytest.approx(0.0)
        assert trajectory_m[i, 1].item() == pytest.approx(0.0)

def test_accel_and_curv_to_meters_trajectory_constant_acceleration_from_standstill():
    # Edge case: starting from 0 speed, but applying constant acceleration
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    action_sequence[0::2] = 2.0  # Constant 2.0 m/s^2 acceleration (every even index is accel)
    current_speed = 0.0

    trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

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

    trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

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

    trajectory_m = Visualization.accel_and_curv_to_meters_trajectory(action_sequence, current_speed, _FUTURE_TIMESTEPS)

    assert not torch.isnan(trajectory_m).any(), "Trajectory contains NaNs"
    assert not torch.isinf(trajectory_m).any(), "Trajectory contains Infs"

    # A tight spiral with these parameters will complete multiple full 360-degree rotations.
    # This means the vehicle must travel "backwards" relative to its start at some point.
    assert trajectory_m[:, 1].min().item() < -0.5, "Car did not loop backwards significantly"

def test_meters_to_pixels_trajectory():
    trajectory_m = torch.tensor([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ])
    radius_m = 20.0
    map_image = np.zeros((400, 400, 3), dtype=np.uint8)

    trajectory_px = Visualization.meters_to_pixels_trajectory(trajectory_m, radius_m, map_image)

    assert trajectory_px.shape == trajectory_m.shape
    # Check pixel coordinates
    # Origin (0,0) in meters is at the top-center of the image. Y is increasing down.
    # Image dimensions: 400x400. Center X is 200.
    # Meter to pixel scale: 400 pixels / (2 * 20m) = 10 pixels/meter
    assert trajectory_px[0, 0] == 200 and trajectory_px[0, 1] == 200 # Origin
    assert trajectory_px[1, 0] == 300 and trajectory_px[1, 1] == 200 # 10m right
    assert trajectory_px[2, 0] == 300 and trajectory_px[2, 1] == 100 # 10m right, 10m up
    assert trajectory_px[3, 0] == 200 and trajectory_px[3, 1] == 100 # 10m up

def test_overlay_the_trajectory_with_map():
    map_image = np.zeros((400, 400, 3), dtype=np.uint8)
    trajectory_px = torch.tensor([
        [200, 399], # Start at bottom center, slightly off edge
        [300, 399],
        [300, 300],
    ])

    overlaid_image = Visualization.overlay_the_trajectory_with_map(trajectory_px, map_image)

    assert overlaid_image is not None
    assert isinstance(overlaid_image, np.ndarray)
    assert overlaid_image.shape == map_image.shape

    # Check if pixels are colored correctly
    # The trajectory should be a non-black color
    # We check points along the drawn line segments
    p1 = (trajectory_px[0,1].item(), trajectory_px[0,0].item()) # (y, x) for numpy
    p2 = (trajectory_px[1,1].item(), trajectory_px[1,0].item())
    p3 = (trajectory_px[2,1].item(), trajectory_px[2,0].item())

    assert not np.array_equal(overlaid_image[p1], [0, 0, 0])
    assert not np.array_equal(overlaid_image[p2], [0, 0, 0])
    assert not np.array_equal(overlaid_image[p3], [0, 0, 0])

    # Check a point on the line between p1 and p2
    mid_p1_p2 = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
    assert not np.array_equal(overlaid_image[mid_p1_p2], [0, 0, 0])