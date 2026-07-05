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
    mock_resolution = 0.4  # Default resolution is 0.4 m/px

    # 3. Create a clean mock map image, following L2D format
    mock_map = np.full((360, 640, 3), (17, 17, 17), dtype=np.uint8) # equivalent to #111111
    map_copy = mock_map.copy()

    print("Executing render_trajectory...")
    # Run the visualization function
    result_img = Visualization.render_trajectory_map_tile(
        action_sequence=mock_actions,
        current_speed=mock_speed,
        map_image=mock_map,
        resolution_m_px=mock_resolution
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
    resolution_m_px = 0.1  # 10 pixels/meter -> 0.1 m/pixel
    map_image = np.zeros((400, 400, 3), dtype=np.uint8)

    trajectory_px = Visualization.meters_to_pixels_trajectory(trajectory_m, resolution_m_px, map_image)

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
    p1 = (int(trajectory_px[0,1].item()), int(trajectory_px[0,0].item())) # (y, x) for numpy
    p2 = (int(trajectory_px[1,1].item()), int(trajectory_px[1,0].item()))
    p3 = (int(trajectory_px[2,1].item()), int(trajectory_px[2,0].item()))

    assert not np.array_equal(overlaid_image[p1], [0, 0, 0])
    assert not np.array_equal(overlaid_image[p2], [0, 0, 0])
    assert not np.array_equal(overlaid_image[p3], [0, 0, 0])

    # Check a point on the line between p1 and p2
    mid_p1_p2 = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
    assert not np.array_equal(overlaid_image[mid_p1_p2], [0, 0, 0])

def test_generate_grid_with_prediction_only():
    # 1. Create a dummy trajectory prediction in meters
    prediction_m = torch.tensor([
        [0.0, 0.0],
        [1.0, 10.0],
        [2.0, 20.0],
    ])
    
    # 2. Run the function
    grid_img = Visualization.generate_grid(prediction_m)
    
    # 3. Assertions
    assert grid_img is not None, "generate_grid returned None"
    assert isinstance(grid_img, np.ndarray), "generate_grid did not return a numpy array"
    assert grid_img.shape == (1080, 480, 3), "Shape of grid image is incorrect"
    # Basic check to ensure it's not all black or background
    assert not np.all(grid_img == grid_img[0][0]), "Grid image appears to be empty"

def test_generate_grid_with_prediction_and_actual():
    # 1. Create dummy trajectories
    prediction_m = torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    actual_m = torch.tensor([[0.0, 0.0], [0.5, 10.0], [1.0, 20.0]])
    
    # 2. Run the function
    grid_img = Visualization.generate_grid(prediction_m, actual_trajectory_m=actual_m)
    
    # 3. Assertions
    assert grid_img is not None
    assert isinstance(grid_img, np.ndarray)
    assert grid_img.shape == (1080, 480, 3)

def test_render_trajectory_on_a_grid():
    # 1. Create a dummy action sequence
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    current_speed = 10.0
    
    # 2. Run the function
    grid_img = Visualization.render_trajectory_on_a_grid(action_sequence, current_speed)
    
    # 3. Assertions
    assert grid_img is not None, "render_trajectory_on_a_grid returned None"
    assert isinstance(grid_img, np.ndarray), "render_trajectory_on_a_grid did not return a numpy array"
    assert grid_img.shape == (1080, 480, 3), "Shape of grid image is incorrect"

def test_get_camera_projection_matrix():
    K = np.eye(3)
    R = np.eye(3)
    t = np.array([[1.0], [2.0], [3.0]])
    
    P = Visualization.get_camera_projection_matrix(K, R, t)
    
    assert P.shape == (3, 4)
    expected_P = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0]
    ])
    assert np.allclose(P, expected_P)

def test_project_BEV_to_CameraView():
    # 3D points in BEV: (x, y) where x is lateral, y is longitudinal
    trajectory_m = torch.tensor([
        [0.0, 10.0],  # Valid point in front
        [2.0, 20.0],  # Valid point in front
        [0.0, -5.0],  # Invalid point behind camera
        [0.0, 0.05]   # Invalid point too close/behind camera (z <= 0.1)
    ])
    
    # Simple projection matrix P = [I | 0]
    P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    points_2d = Visualization.project_BEV_to_CameraView(trajectory_m, P)
    
    assert points_2d.shape == (4, 2)
    # Check valid points
    # 3D point is [x, 1.5, z] -> [0.0, 1.5, 10.0] -> 2D is [0/10, 1.5/10] = [0.0, 0.15]
    assert np.allclose(points_2d[0], [0.0, 0.15])
    # 3D point [2.0, 1.5, 20.0] -> 2D is [2/20, 1.5/20] = [0.1, 0.075]
    assert np.allclose(points_2d[1], [0.1, 0.075])
    
    # Check invalid points
    assert np.allclose(points_2d[2], [-1.0, -1.0])
    assert np.allclose(points_2d[3], [-1.0, -1.0])

def test_render_trajectory_on_camera_view():
    camera_image = np.zeros((400, 400, 3), dtype=np.uint8)
    # Valid points within the image, and some outside
    trajectory_2d = np.array([
        [200, 200],  # Inside
        [250, 250],  # Inside
        [-10, -10],  # Outside
        [500, 500]   # Outside
    ])
    
    test_color = (123, 45, 67) # BGR
    img_with_traj = Visualization.render_trajectory_on_camera_view(
        camera_image, trajectory_2d, color=test_color, thickness=3
    )
    
    assert img_with_traj.shape == (400, 400, 3)
    
    # Check that color is present in the image where the line is drawn
    # The line from (200, 200) to (250, 250) should be drawn with `test_color`
    color_matches = np.all(img_with_traj == test_color, axis=-1)
    assert np.any(color_matches), "Expected color not found in the rendered image"
    
def test_complete_front_camera_view_with_trajectory():
    action_sequence_target = torch.zeros(128)
    action_sequence_target[0::2] = 0.5  # acceleration
    action_sequence_target[1::2] = 0.1  # curvature (left turn)
    action_sequence_pred = torch.zeros(128)
    action_sequence_pred[0::2] = -0.5 # deceleration
    action_sequence_pred[1::2] = -0.1 # curvature (right turn)
    current_speed = 10.0
    front_camera_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Simple dummy projection matrix to make projection work without math domain errors
    # P must map [x, 1.5, z] to something reasonable
    K = np.array([[500, 0, 300], [0, 500, 200], [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros((3, 1))
    
    combined_img = Visualization.complete_front_camera_view_with_trajectory(
        action_sequence_target,
        action_sequence_pred,
        current_speed,
        front_camera_image,
        K=K, R=R, t=t
    )
    
    assert combined_img is not None
    # Grid is 1080x480. Camera is resized to height 1080.
    # scale = 1080 / 400 = 2.7
    # new_cam_w = int(600 * 2.7) = 1620
    # Expected combined width = 480 + 1620 = 2100
    assert combined_img.shape == (1080, 2100, 3)
    
    # The camera image part is the right side: combined_img[:, 480:]
    cam_part = combined_img[:, 480:]
    
    # Let's check for the presence of target color (59, 108, 255) and pred color (52, 217, 164)
    target_color = (59, 108, 255)
    pred_color = (52, 217, 164)
    
    target_matches = np.all(cam_part == target_color, axis=-1)
    pred_matches = np.all(cam_part == pred_color, axis=-1)
    
    assert np.any(target_matches), "Target trajectory color not found in the combined image"
    assert np.any(pred_matches), "Predicted trajectory color not found in the combined image"
