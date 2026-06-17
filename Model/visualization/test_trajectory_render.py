import sys
sys.path.append('..')
from visualization.trajectory_rendering import Visualization
import torch
from PIL import Image
from model_components.auto_e2e import AutoE2E
from torch.utils.data import DataLoader
from data_parsing.l2d.camera import NUM_VIEWS
import torchvision.transforms.functional as F

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

def run_visualization_test_l2d():
    print("Initializing inputs for visualization test...")
    
    result = forward_pass_for_visualization_test(episodes=[0], batch_size=2, pretrained_backbone=False)
    if result is None:
        print("L2D dataset not available. Skipping L2D visualization test.")
        return
        
    pred_trajectory, target_trajectory, map_image, current_speed = result
    radius_m = 800.0  # Standard map metric boundary assumption
    
    print(f"Rendering ground truth trajectory (speed: {current_speed:.2f} m/s)...")
    gt_img = Visualization.render_trajectory(
        action_sequence=target_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        radius_m=radius_m
    )
    gt_img.save("test_trajectory_l2d_gt.png")
    print("Saved ground truth to test_trajectory_l2d_gt.png")
    
    print("Rendering predicted trajectory...")
    pred_img = Visualization.render_trajectory(
        action_sequence=pred_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        radius_m=radius_m
    )
    pred_img.save("test_trajectory_l2d_pred.png")
    print("Saved prediction to test_trajectory_l2d_pred.png")

def forward_pass_for_visualization_test(episodes: list[int], batch_size: int, pretrained_backbone: bool):
    """
    The function is almost a 1-to-1 copy of test_live_dataset in forward_pass_test.py for L2D dataset
    Run forward pass with real L2D data.
    """
    try:
        from data_parsing.l2d import L2DDataset
    except ImportError as e:
        print(f"[live] SKIPPED: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live] Device: {device}")

    try:
        dataset = L2DDataset(
            repo_id="yaak-ai/L2D",
            episodes=episodes,
            local_files_only=False,
        )
    except Exception as e:
        print(f"[live] SKIPPED: cannot load dataset: {e}")
        return

    print(f"[live] Valid samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    visual_tiles = batch["visual_tiles"].to(device)
    visual_history = batch["visual_history"].to(device)
    egomotion_history = batch["egomotion_history"].to(device)
    trajectory_target = batch["trajectory_target"].to(device)

    map_tensor = batch["visual_tiles"][-1, 6].cpu()

    # Reverse standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnormalized_map = (map_tensor * std) + mean

    unnormalized_map = torch.clamp(unnormalized_map, 0.0, 1.0)

    raw_map_image = F.to_pil_image(unnormalized_map)

    current_speed = egomotion_history[-1, 252].item()

    model = AutoE2E(
        num_views=NUM_VIEWS,
        is_pretrained=pretrained_backbone,
    ).to(device)

    model.eval()

    with torch.no_grad():
        trajectory, compressed, future = model(
            visual_tiles, visual_history, egomotion_history
        )

    return trajectory[-1].cpu(), trajectory_target[-1].cpu(), raw_map_image, current_speed

if __name__ == "__main__":
    run_visualization_test_dummy_data()
    run_visualization_test_l2d()