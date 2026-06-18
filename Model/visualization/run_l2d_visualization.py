"""
Usage:
    cd Model/visualization
    python run_l2d_visualization.py

    # With real data (requires lerobot + cached dataset):
    python run_l2d_visualization.py --live --episodes 0
"""

import sys
sys.path.append('..')
from visualization.trajectory_rendering import Visualization
import torch
from model_components.auto_e2e import AutoE2E
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from data_parsing.l2d.camera import NUM_VIEWS
from PIL import Image
from pathlib import Path
import os
import argparse

def visualization_on_l2d(save_path: Path, episodes: list[int]):
    result = forward_pass_for_visualization_test(episodes=episodes, batch_size=2, pretrained_backbone=False)

    pred_trajectory, target_trajectory, map_image, current_speed = result
    radius_m = 800.0  # Standard map metric boundary assumption

    print(f"Rendering ground truth trajectory (speed: {current_speed:.2f} m/s)...")
    gt_img = Visualization.render_trajectory_map_tile(
        action_sequence=target_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        radius_m=radius_m
    )
    output_path_gt = save_path / "test_trajectory_l2d_gt.png"
    gt_img.save(output_path_gt)

    pred_img = Visualization.render_trajectory_map_tile(
        action_sequence=pred_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        radius_m=radius_m
    )
    output_path_pred = save_path / "test_trajectory_l2d_pred.png"
    pred_img.save(output_path_pred)

    assert not result is None, 'L2D dataset not available. Skipping L2D visualization test'
    assert os.path.isfile(output_path_gt), "Image file was not created in the target directory"


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
            visual_tiles,
            map_tensor=batch["visual_tiles"][-1, 6],
            visual_history=visual_history,
            egomotion_history=egomotion_history,
            mode="infer"
        )

    return trajectory[-1].cpu(), trajectory_target[-1].cpu(), raw_map_image, current_speed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L2D visualization test')
    parser.add_argument('--live', action='store_true', help='Run live L2D dataset visualization')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0], help='List of episodes to load')
    args = parser.parse_args()

    if args.live:
        visualization_on_l2d(Path("test_images"), args.episodes)
    else:
        print("Skipping. Run with --live to execute L2D visualization.")