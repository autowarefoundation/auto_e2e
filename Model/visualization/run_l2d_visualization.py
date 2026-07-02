"""
Usage:
    cd Model/visualization
    python run_l2d_visualization.py

    # With real data (requires lerobot + cached dataset):
    python run_l2d_visualization.py --live --episodes 0
"""

import sys
import os
sys.path.append('..')
from visualization.trajectory_rendering import Visualization
import torch
from model_components.auto_e2e import AutoE2E
import cv2
import numpy as np
from data_parsing.l2d.camera import NUM_VIEWS
import argparse
from torch.utils.data import DataLoader

def visualization_on_l2d(episodes: list[int], zoom_in: bool = False) -> np.ndarray:
    result = forward_pass_for_visualization_test(episodes=episodes, batch_size=2, pretrained_backbone=False)
    
    pred_trajectory, target_trajectory, map_image, current_speed, current_heading = result
    radius_m = 800.0  # Standard map metric boundary assumption

    print(f"Rendering trajectories (speed: {current_speed:.2f} m/s)...")

    if zoom_in:
        h, w = map_image.shape[:2]
        cropped_w, cropped_h = w // 8, h // 8
        map_image = map_image[h//2 - cropped_h : h//2 + cropped_h, w//2 - cropped_w : w//2 + cropped_w]
        # Since we cropped the central half of the map, the radius is halved.
        radius_m = radius_m / 4.0

    target_w, target_h = 1280, 720
    map_image = cv2.resize(map_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 1. Draw extracted ground truth (actual driven path)
    combined_img = Visualization.render_trajectory_map_tile(
        action_sequence=target_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        radius_m=radius_m,
        color=(255, 108, 59),
        initial_heading=current_heading
    )

    # 2. Draw predicted path
    combined_img = Visualization.render_trajectory_map_tile(
        action_sequence=pred_trajectory,
        current_speed=current_speed,
        map_image=combined_img,
        radius_m=radius_m,
        color=(164, 217, 52), 
        initial_heading=current_heading
    )

    return combined_img

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

    camera_tiles = visual_tiles[:, :6]
    map_input = visual_tiles[:, 6]

    visual_history = batch["visual_history"].to(device)
    egomotion_history = batch["egomotion_history"].to(device)
    trajectory_target = batch["trajectory_target"].to(device)

    raw_map_tensor = batch["raw_map"][-1].cpu()
    raw_map_array = (raw_map_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    raw_map_image = cv2.cvtColor(raw_map_array, cv2.COLOR_RGB2BGR)

    current_speed = egomotion_history[-1, 252].item()
    current_heading = batch["current_heading"][-1].item() if "current_heading" in batch else 0.0

    model = AutoE2E(
        num_views=NUM_VIEWS - 1,
        is_pretrained=pretrained_backbone,
    ).to(device)

    model.eval()

    with torch.no_grad():
        out = model(
            camera_tiles=camera_tiles,
            map_input=map_input,
            visual_history=visual_history,
            egomotion_history=egomotion_history,
            mode="infer"
        )
        trajectory = out[0] if isinstance(out, tuple) else out

    return trajectory[-1].cpu(), trajectory_target[-1].cpu(), raw_map_image, current_speed, current_heading

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L2D visualization test')
    parser.add_argument('--live', action='store_true', help='Run live L2D dataset visualization')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0], help='List of episodes to load')
    parser.add_argument(
        "--zoom_in", action="store_true", help="Zoom in on the agent"
    )
    args = parser.parse_args()

    if args.live:
        combined_image = visualization_on_l2d(args.episodes, zoom_in=args.zoom_in)
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "visualization_result_l2d.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, combined_image)
    else:
        print("Skipping. Run with --live to execute L2D visualization.")