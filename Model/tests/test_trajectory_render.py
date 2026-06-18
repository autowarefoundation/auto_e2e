import sys
sys.path.append('..')
from visualization.trajectory_rendering import Visualization
import torch
import pytest
from PIL import Image
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
    mock_map = Image.new("RGB", (640, 360), color="#111111")
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
    result_img.save(output_path)

    assert result_img is not None, "Visualization function returned None"
    assert isinstance(result_img, Image.Image), "Visualization function did not return an image"
    assert result_img.size == mock_map.size, "Size does not match"
    assert result_img.mode == mock_map.mode, "Mode does not match"
    assert list(map_copy.getdata()) == list(mock_map.getdata()), "Original image mutated"
    assert list(result_img.getdata()) != list(mock_map.getdata()), "Image file was not created in the target directory"
    assert os.path.isfile(output_path), "Image file was not created in the target directory"