"""
Usage:
    cd Model/visualization
    python L2D_visualization/run_l2d_visualization.py

    # With real data (requires lerobot + cached dataset):
    python L2D_visualization/run_l2d_visualization.py --live --episodes 0
"""

import os

from Tools.trajectory_visualization.rendering import (
    render_trajectory_map_tile,
    render_trajectory_on_a_grid,
    complete_front_camera_view_with_trajectory,
    concatenate_grid_and_camera,
    MapGeometry
)
from Tools.trajectory_visualization.kinematics import controls_to_metric_trajectory, ModelOutputContract
import torch
from Model.model_components.auto_e2e import AutoE2E
import cv2
import numpy as np
from Model.data_parsing.l2d.camera import NUM_VIEWS
import argparse
import yaml

def visualization_on_l2d(episodes: list[int], frame_index: int = 0, zoom_in: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = forward_pass_for_visualization_test(episodes=episodes, frame_index=frame_index, pretrained_backbone=False)
    
    pred_trajectory, target_trajectory, map_image, front_image, current_speed, current_heading = result
    resolution_m_px = 1 # Actual resolution based on L2D map scale

    print(f"Rendering trajectories (speed: {current_speed:.2f} m/s)...")

    if zoom_in:
        h, w = map_image.shape[:2]
        cropped_w, cropped_h = w // 8, h // 8
        map_image = map_image[h//2 - cropped_h : h//2 + cropped_h, w//2 - cropped_w : w//2 + cropped_w]

    current_w = map_image.shape[1]
    target_w, target_h = 1280, 720
    map_image = cv2.resize(map_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Scale resolution based on resize ratio
    resolution_m_px = resolution_m_px * (current_w / target_w)
    
    # 0.5 Define color scheme
    prediction_color = (140, 255, 0)
    actual_trajectory_color = (255, 80, 120)

    contract = ModelOutputContract(
        num_timesteps=pred_trajectory.shape[0] // 2,
        num_signals=2,
        sampling_interval_dt=0.1,
        acceleration_unit="m/s^2",
        curvature_unit="rad/m",
        speed_unit="m/s",
        coordinate_handedness="right-handed"
    )

    # Precompute metric trajectories
    pred_xy = controls_to_metric_trajectory(pred_trajectory, current_speed, contract=contract)
    target_xy = controls_to_metric_trajectory(target_trajectory, current_speed, contract=contract) if target_trajectory is not None else None

    h, w = map_image.shape[:2]
    map_geometry = MapGeometry(
        meters_per_pixel_x=resolution_m_px,
        meters_per_pixel_y=resolution_m_px,
        ego_pixel_x=w / 2.0,
        ego_pixel_y=h / 2.0,
        rotation_rad=current_heading
    )

    # 1. Draw extracted ground truth (actual driven path)
    combined_img = map_image.copy()
    if target_xy is not None:
        combined_img = render_trajectory_map_tile(
            prediction_xy=target_xy,
            map_image=combined_img,
            geometry=map_geometry,
            color=actual_trajectory_color
        )

    # 2. Draw predicted path
    combined_img = render_trajectory_map_tile(
        prediction_xy=pred_xy,
        map_image=combined_img,
        geometry=map_geometry,
        color=prediction_color,
        is_approximate=True
    )

    # 3. Create Camera View with trajectory and Grid
    grid_with_trajectory = render_trajectory_on_a_grid(
        prediction_xy=pred_xy,
        target_xy=target_xy,
        prediction_color=prediction_color,
        actual_trajectory_color=actual_trajectory_color
    )

    # Load Extrinsics
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data_parsing', 'l2d', 'configs', 'extrinsic_RDF.yaml')
    R, t = load_extrinsics(yaml_path, view_name="observation.images.front_left")
    
    cam_trajectory_view = front_image.copy()
    
    if target_xy is not None:
        cam_trajectory_view = complete_front_camera_view_with_trajectory(
            prediction_xy=target_xy,
            front_camera_image=cam_trajectory_view,
            R=R,
            t=t,
            color=actual_trajectory_color,
            is_approximate=True
        )
    
    if pred_xy is not None:
        cam_trajectory_view = complete_front_camera_view_with_trajectory(
            prediction_xy=pred_xy,
            front_camera_image=cam_trajectory_view,
            R=R,
            t=t,
            color=prediction_color,
            is_approximate=True
        )
        
    cam_trajectory_view = concatenate_grid_and_camera(
        grid_img=grid_with_trajectory,
        cam_img=cam_trajectory_view
    )

    return combined_img, grid_with_trajectory, cam_trajectory_view

def forward_pass_for_visualization_test(
    episodes: list[int], frame_index: int = 0, pretrained_backbone: bool = False
    ):
    """
    Run forward pass with real L2D data at a specific frame index.
    """
    try:
        from Model.data_parsing.l2d import L2DDataset
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

    if frame_index >= len(dataset) or frame_index < 0:
        print(f"[live] Frame index {frame_index} is out of bounds (0 to {len(dataset)-1}). Defaulting to 0.")
        frame_index = 0

    from torch.utils.data.dataloader import default_collate
    sample = dataset[frame_index]
    batch = default_collate([sample])

    visual_tiles = batch["visual_tiles"].to(device)

    camera_tiles = visual_tiles[:, :6]
    map_input = visual_tiles[:, 6]

    visual_history = batch["visual_history"].to(device)
    egomotion_history = batch["egomotion_history"].to(device)
    trajectory_target = batch["trajectory_target"].to(device)

    raw_map_tensor = batch["raw_map"][-1].cpu()
    raw_map_array = (raw_map_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    raw_map_image = cv2.cvtColor(raw_map_array, cv2.COLOR_RGB2BGR)

    ep_idx, row = dataset._samples[frame_index]
    raw_front_tensor = dataset.lerobot_dataset[row]["observation.images.front_left"].cpu()
    raw_front_array = (raw_front_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    raw_front_image = cv2.cvtColor(raw_front_array, cv2.COLOR_RGB2BGR)

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

    return trajectory[-1].cpu(), trajectory_target[-1].cpu(), raw_map_image, raw_front_image, current_speed, current_heading

def load_extrinsics(path_to_yaml: str, view_name: str = "observation.images.front_left") -> tuple[np.ndarray, np.ndarray]:
        """
        Parses the calibration YAML with camera extrinsics.
        
        Args:
            path_to_yaml: Path to the calibration YAML file.
            view_name: The camera view to extract the matrix for (e.g., "observation.images.front_left").
        
        Returns:
            R and t
        """
   
        with open(path_to_yaml, 'r') as f:
            calib_data = yaml.safe_load(f)

        try:
            # Map view_name to YAML key format
            yaml_key = f"cam_{view_name.split('.')[-1]}"
            
            view_calib = calib_data[yaml_key] if yaml_key in calib_data else calib_data
            
            # Extract Extrinsic rotation (3x3) and translation (3x1)
            R = np.array(view_calib['extrinsic_rotation_ref_cam_from_cam'], dtype=np.float32).reshape(3, 3)
            t = np.array(view_calib['extrinsic_t_ref_cam_from_cam'], dtype=np.float32).reshape(3, 1)

            return R, t
            
        except KeyError as e:
            raise KeyError(f"Key {e} not found in YAML. Please check the structure of {path_to_yaml} and update the keys in this function.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='L2D visualization test')
    parser.add_argument('--live', action='store_true', help='Run live L2D dataset visualization')
    parser.add_argument('--episodes', type=int, nargs='+', default=[0], help='List of episodes to load')
    parser.add_argument('--frame', type=int, default=0, help='Which frame index of the episode to visualize')
    parser.add_argument(
        "--zoom_in", action="store_true", help="Zoom in on the agent"
    )
    args = parser.parse_args()

    if args.live:
        combined_image, grid_with_trajectory, cam_trajectory_view = visualization_on_l2d(args.episodes, frame_index=args.frame, zoom_in=args.zoom_in)
        save_path_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "visualization_result_map.png")
        save_path_grid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "visualization_result_grid.png")
        save_path_cam = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "visualization_result_cam.png")
        os.makedirs(os.path.dirname(save_path_map), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_grid), exist_ok=True)
        cv2.imwrite(save_path_map, combined_image)
        cv2.imwrite(save_path_grid, grid_with_trajectory)
        cv2.imwrite(save_path_cam, cam_trajectory_view)
    else:
        print("Skipping. Run with --live to execute L2D visualization.")