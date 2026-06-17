import sys
sys.path.append('..')
from visualization.trajectory_rendering import Visualization
import torch
from PIL import Image

def run_visualization_test_dummy_data():
    print("Initializing mock inputs for visualization test...")

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

    print("Executing render_trajectory...")
    try:
        # Run your visualization function
        result_img = Visualization.render_trajectory(
            action_sequence=mock_actions,
            current_speed=mock_speed,
            map_image=mock_map,
            radius_m=mock_radius
        )

        # 4. Save and inspect the result
        output_path = "test_trajectory_output.png"
        result_img.save(output_path)
        print(f"Success! Test image saved to: {output_path}")

    except Exception as e:
        print(f"Test failed with an error: {e}")


if __name__ == "__main__":
    run_visualization_test_dummy_data()