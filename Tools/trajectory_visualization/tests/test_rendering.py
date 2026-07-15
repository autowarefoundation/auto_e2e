import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..', '..')))

from Tools.trajectory_visualization import kinematics  # noqa: E402
from Tools.trajectory_visualization import rendering  # noqa: E402

_DT = 0.1
_FUTURE_TIMESTEPS = 64

def get_dummy_contract():
    return kinematics.ModelOutputContract(
        num_timesteps=_FUTURE_TIMESTEPS,
        num_signals=2,
        sampling_interval_dt=_DT,
        acceleration_unit="m/s^2",
        curvature_unit="rad/m",
        speed_unit="m/s",
        coordinate_handedness="right-handed"
    )

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
    # Generate trajectory first
    prediction_xy = kinematics.controls_to_metric_trajectory(mock_actions, mock_speed, contract=get_dummy_contract())
    
    # Run the visualization function
    geometry = rendering.MapGeometry(
        meters_per_pixel_x=mock_resolution,
        meters_per_pixel_y=mock_resolution,
        ego_pixel_x=mock_map.shape[1] / 2.0,
        ego_pixel_y=mock_map.shape[0] / 2.0,
        rotation_rad=0.0
    )
    result_img = rendering.render_trajectory_map_tile(
        prediction_xy=prediction_xy,
        map_image=mock_map,
        geometry=geometry
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


def test_meters_to_pixels_trajectory():
    trajectory_m = torch.tensor([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ])
    resolution_m_px = 0.1  # 10 pixels/meter -> 0.1 m/pixel
    geometry = rendering.MapGeometry(
        meters_per_pixel_x=resolution_m_px,
        meters_per_pixel_y=resolution_m_px,
        ego_pixel_x=200.0,
        ego_pixel_y=200.0,
        rotation_rad=0.0
    )

    trajectory_px = rendering.meters_to_pixels_trajectory(trajectory_m, geometry)

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
    geometry = rendering.MapGeometry(
        meters_per_pixel_x=0.1,
        meters_per_pixel_y=0.1,
        ego_pixel_x=200.0,
        ego_pixel_y=200.0,
        rotation_rad=0.0
    )
    trajectory_px = torch.tensor([
        [200, 399], # Start at bottom center, slightly off edge
        [300, 399],
        [300, 300],
    ])

    overlaid_image = rendering.overlay_the_trajectory_with_map(trajectory_px, map_image, geometry)

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
    grid_img = rendering.generate_grid(prediction_xy=prediction_m)
    
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
    grid_img = rendering.generate_grid(prediction_xy=prediction_m, target_xy=actual_m)
    
    # 3. Assertions
    assert grid_img is not None
    assert isinstance(grid_img, np.ndarray)
    assert grid_img.shape == (1080, 480, 3)

def test_generate_grid_clipping():
    # 1. Create a trajectory that goes WAY out of bounds
    prediction_m = torch.tensor([[0.0, 0.0], [500.0, 500.0], [-500.0, -500.0]])
    actual_m = torch.tensor([[0.0, 0.0], [1000.0, -1000.0], [-1000.0, 1000.0]])
    
    # 2. Run the function with unique colors
    unique_pred_color = (1, 2, 3)
    unique_act_color = (4, 5, 6)
    grid_img = rendering.generate_grid(
        prediction_xy=prediction_m, 
        target_xy=actual_m,
        prediction_color=unique_pred_color,
        actual_trajectory_color=unique_act_color
    )
    
    # 3. Assertions
    margin_left, margin_right = 50, 20
    margin_top, margin_bottom = 60, 50
    width, height = 480, 1080
    
    # The plot area is:
    # x: margin_left to width - margin_right
    # y: margin_top to height - margin_bottom
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    # Extract the three margins
    bottom_margin = grid_img[margin_top + plot_h:, :]
    left_margin = grid_img[margin_top:margin_top + plot_h, 0:margin_left]
    right_margin = grid_img[margin_top:margin_top + plot_h, margin_left + plot_w:]
    
    # The top margin contains the legend which draws a circle with the prediction/actual color!
    # So we must just check bottom/left/right margins where no legend is drawn.
    for margin, name in [(bottom_margin, "bottom"), (left_margin, "left"), (right_margin, "right")]:
        assert not np.any(np.all(margin == unique_pred_color, axis=-1)), f"Prediction color bled into {name} margin"
        assert not np.any(np.all(margin == unique_act_color, axis=-1)), f"Actual color bled into {name} margin"

def test_render_trajectory_on_a_grid():
    # 1. Create a dummy action sequence
    action_sequence = torch.zeros(_FUTURE_TIMESTEPS * 2)
    current_speed = 10.0
    
    # 2. Convert to metric and run the function
    prediction_xy = kinematics.controls_to_metric_trajectory(action_sequence, current_speed, contract=get_dummy_contract())
    grid_img = rendering.render_trajectory_on_a_grid(prediction_xy=prediction_xy)
    
    # 3. Assertions
    assert grid_img is not None, "render_trajectory_on_a_grid returned None"
    assert isinstance(grid_img, np.ndarray), "render_trajectory_on_a_grid did not return a numpy array"
    assert grid_img.shape == (1080, 480, 3), "Shape of grid image is incorrect"

def test_get_camera_projection_matrix():
    K = np.eye(3)
    R = np.eye(3)
    t = np.array([[1.0], [2.0], [3.0]])
    
    P = rendering.get_camera_projection_matrix(K, R, t)
    
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
    
    points_2d = rendering.project_BEV_to_CameraView(trajectory_m, P)
    
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
    left_2d = np.array([
        [190, 200],  # Inside
        [240, 250],  # Inside
        [-10, -10],  # Outside
        [490, 500]   # Outside
    ])
    right_2d = np.array([
        [210, 200],  # Inside
        [260, 250],  # Inside
        [-10, -10],  # Outside
        [510, 500]   # Outside
    ])
    
    test_color = (123, 45, 67) # BGR
    img_with_traj = rendering.render_trajectory_on_camera_view(
        camera_image, left_2d, right_2d, color=test_color, outline_thickness=3
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
    
    # Convert to metric with dummy contract
    pred_traj_m = kinematics.controls_to_metric_trajectory(action_sequence_pred, current_speed, contract=get_dummy_contract())
    target_traj_m = kinematics.controls_to_metric_trajectory(action_sequence_target, current_speed, contract=get_dummy_contract())
    
    cam_img = rendering.complete_front_camera_view_with_trajectory(
        prediction_xy=target_traj_m,
        front_camera_image=front_camera_image,
        K=K, R=R, t=t,
        color=(59, 108, 255),
        is_approximate=True
    )
    cam_img = rendering.complete_front_camera_view_with_trajectory(
        prediction_xy=pred_traj_m,
        front_camera_image=cam_img,
        K=K, R=R, t=t,
        color=(140, 255, 0),
        is_approximate=True
    )
    
    grid_img = rendering.generate_grid(prediction_xy=pred_traj_m, target_xy=target_traj_m)
    
    combined_img = rendering.concatenate_grid_and_camera(grid_img, cam_img)
    
    assert combined_img is not None
    # Grid is 1080x480. Camera (400, 600) is resized to height 1080 -> width = 600 * (1080/400) = 1620
    # Expected combined width = 480 + 1620 = 2100
    assert combined_img.shape == (1080, 2100, 3)
    
    # Check that colors were drawn in the camera part
    cam_part = combined_img[:, 480:]
    # Resizing might interpolate colors, so we check for presence approximately or just check it's not empty
    assert np.any(cam_part != 0)
