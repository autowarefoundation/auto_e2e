"""
Usage:
    cd Model/visualization
    python -m Kit_Scenes_visualization.kit_scenes_visualizer --scene_ids <uuid> --frame 0
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import cv2
import numpy as np
import argparse
from typing import Optional

from visualization.trajectory_rendering import Visualization
from model_components.auto_e2e import AutoE2E
from data_parsing.kit_scenes.camera import NUM_VIEWS
from data_parsing.kit_scenes.map import generate_bev_map_tile


def visualization_on_kit_scenes(scene_ids: Optional[list[str]] = None, frame_index: int = 0, zoom_in: bool = False, dataset_root: Optional[str] = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    result = forward_pass_for_visualization_test(scene_ids=scene_ids, frame_index=frame_index, dataset_root=dataset_root, pretrained_backbone=False)
    
    if result is None:
        print("Failed to get visualization result.")
        return None, None
        
    pred_trajectory, target_trajectory, map_image, raw_camera_image, current_speed, current_heading, P = result
    
    resolution_m_px: float = 1.0 # Actual resolution depends on the zoom scale, we'll keep it as 1 to see how it looks.
    # The default generate_bev_map_tile size is 1024, radius is 60m. So 120m / 1024px = 0.117 m/px.
    resolution_m_px = 120.0 / 1024.0

    print(f"Rendering trajectories (speed: {current_speed:.2f} m/s)...")

    if zoom_in:
        h, w = map_image.shape[:2]
        cropped_w, cropped_h = w // 4, h // 4
        map_image = map_image[h//2 - cropped_h : h//2 + cropped_h, w//2 - cropped_w : w//2 + cropped_w]
        resolution_m_px = resolution_m_px * 0.5  # Zoomed in, so resolution is finer

    current_w = map_image.shape[1]
    target_w, target_h = 1024, 1024
    if map_image.shape[:2] != (target_h, target_w):
        map_image = cv2.resize(map_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resolution_m_px = resolution_m_px * (current_w / target_w)

    # 0.5 Define color scheme
    prediction_color = (140, 255, 0)
    actual_trajectory_color = (255, 80, 120)

    # 1. Draw extracted ground truth (actual driven path) on map tile
    combined_img = Visualization.render_trajectory_map_tile(
        action_sequence=target_trajectory,
        current_speed=current_speed,
        map_image=map_image,
        resolution_m_px=resolution_m_px,
        color=actual_trajectory_color,
        initial_heading=current_heading
    )

    # 2. Draw predicted path on map tile
    combined_img = Visualization.render_trajectory_map_tile(
        action_sequence=pred_trajectory,
        current_speed=current_speed,
        map_image=combined_img,
        resolution_m_px=resolution_m_px,
        color=prediction_color, 
        initial_heading=current_heading
    )

    # 3. Create Camera View with trajectory and Grid
    grid_with_trajectory = Visualization.render_trajectory_on_a_grid(
        action_sequence=pred_trajectory,
        current_speed=current_speed,
        actual_action_sequence=target_trajectory,
        prediction_color=prediction_color,
        actual_trajectory_color=actual_trajectory_color
    )

    cam_trajectory_view = raw_camera_image.copy()
    
    if target_trajectory is not None:
        cam_trajectory_view = Visualization.complete_front_camera_view_with_trajectory(
            action_sequence=target_trajectory,
            current_speed=current_speed,
            front_camera_image=cam_trajectory_view,
            P=P,
            color=actual_trajectory_color
        )
        
    if pred_trajectory is not None:
        cam_trajectory_view = Visualization.complete_front_camera_view_with_trajectory(
            action_sequence=pred_trajectory,
            current_speed=current_speed,
            front_camera_image=cam_trajectory_view,
            P=P,
            color=prediction_color
        )
        
    camera_and_grid = Visualization.concatenate_grid_and_camera(
        grid_img=grid_with_trajectory,
        cam_img=cam_trajectory_view
    )

    return combined_img, camera_and_grid


def forward_pass_for_visualization_test(
    scene_ids: Optional[list[str]] = None, frame_index: int = 0, dataset_root: Optional[str] = None, pretrained_backbone: bool = False
    ):
    """
    Run forward pass with real KIT Scenes data at a specific frame index.
    """
    try:
        from data_parsing.kit_scenes import KitScenesDataset
    except ImportError as e:
        print(f"[live] SKIPPED: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live] Device: {device}")

    try:
        dataset = KitScenesDataset(
            data_root=dataset_root,
            scene_ids=scene_ids,
            rasterize_map_at_runtime=True,
        )
    except Exception as e:
        print(f"[live] SKIPPED: cannot load dataset: {e}")
        return None

    print(f"[live] Valid samples: {len(dataset)}")

    if frame_index >= len(dataset) or frame_index < 0:
        print(f"[live] Frame index {frame_index} is out of bounds (0 to {len(dataset)-1}). Defaulting to 0.")
        frame_index = 0

    from torch.utils.data.dataloader import default_collate
    sample = dataset[frame_index]
    batch = default_collate([sample])

    visual_tiles = batch["visual_tiles"].to(device)
    map_tile = batch["map_tile"].to(device)
    visual_history = batch["visual_history"].to(device)
    egomotion_history = batch["egomotion_history"].to(device)
    trajectory_target = batch["trajectory_target"].to(device)
    camera_params = batch["camera_params"].to(device)

    # Get raw camera frame directly from the SDK loader for visualization
    loader = dataset._sdk.get_sensor_loader(sample["scene_id"])
    raw_cam_array = loader.get_camera_image("camera_base_front_center", sample["frame_idx"])
    raw_cam_image = cv2.cvtColor(raw_cam_array, cv2.COLOR_RGB2BGR)

    # Get raw high-res BEV map tile directly
    ego_xy = dataset._scene_positions[sample["scene_id"]][sample["frame_idx"]]
    ego_yaw = float(dataset._scene_egomotion[sample["scene_id"]][sample["frame_idx"], 2])
    
    raw_map_array = generate_bev_map_tile(
        scene_path=loader.scene_path,
        ego_x=float(ego_xy[0]),
        ego_y=float(ego_xy[1]),
        ego_yaw=float(ego_yaw),
        canvas_size=1024, # Render a higher-res map for visualization
        radius_meters=60.0
    )
    if raw_map_array is not None:
        raw_map_image = cv2.cvtColor(raw_map_array, cv2.COLOR_RGB2BGR)
    else:
        raw_map_image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    current_speed = egomotion_history[-1, 252].item()
    current_heading = 0.0 # Map tile is already ego-centric (yaw=0 points up)

    model = AutoE2E(
        num_views=NUM_VIEWS,
        is_pretrained=pretrained_backbone,
    ).to(device)

    model.eval()

    with torch.no_grad():
        out = model(
            camera_tiles=visual_tiles,
            map_input=map_tile,
            visual_history=visual_history,
            egomotion_history=egomotion_history,
            camera_params=camera_params,
            mode="infer"
        )
        trajectory = out[0] if isinstance(out, tuple) else out

    # Compute unscaled projection matrix for the raw high-res camera image
    calib = loader.get_camera_calibration("camera_base_front_center")
    K_raw = calib.intrinsic
    T_ref_to_cam = np.linalg.inv(calib.extrinsic)
    P_unscaled = K_raw @ T_ref_to_cam[:3, :]
    
    # visualization/trajectory_rendering.py assumes input points to P are in RDF (Right, Down, Forward)
    # KIT Scenes reference frame is FLU (Forward, Left, Up).
    # Since Z=0 is floating above the camera, the reference frame is likely on the roof (IMU).
    # We can anchor the trajectory to the ground by finding the camera's Z height in the reference frame,
    # and assuming the camera is ~1.5m above the ground.
    camera_z_in_ref = calib.extrinsic[2, 3]
    z_ground = camera_z_in_ref - 1.5
    
    T_RDF_to_FLU = np.array([
        [ 0,  0,  1,  0],        # Forward = Z_RDF
        [-1,  0,  0,  0],        # Left = -X_RDF
        [ 0,  0,  0,  z_ground], # Up = ground level
        [ 0,  0,  0,  1]
    ], dtype=np.float32)
    
    P = (P_unscaled @ T_RDF_to_FLU).astype(np.float32)
    
    return trajectory[-1].cpu(), trajectory_target[-1].cpu(), raw_map_image, raw_cam_image, current_speed, current_heading, P


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KIT Scenes visualization test')
    parser.add_argument('--dataset_root', type=str, default=None, help='Path to KIT Scenes dataset root. Defaults to $KITSCENES_ROOT if not provided.')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None, help='List of scene IDs to load')
    parser.add_argument('--frame', type=int, default=0, help='Which frame index to visualize')
    parser.add_argument("--zoom_in", action="store_true", help="Zoom in on the agent")
    args = parser.parse_args()

    dataset_root = args.dataset_root or os.environ.get("KITSCENES_ROOT")
    if dataset_root is None:
        raise ValueError("Dataset root must be provided via --dataset_root or KITSCENES_ROOT environment variable.")

    combined_image, camera_and_grid = visualization_on_kit_scenes(args.scene_ids, frame_index=args.frame, zoom_in=args.zoom_in, dataset_root=dataset_root)

    
    if combined_image is not None and camera_and_grid is not None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path_map = os.path.join(save_dir, "visualization_result_map.png")
        save_path_grid = os.path.join(save_dir, "visualization_result_grid.png")
        
        cv2.imwrite(save_path_map, combined_image)
        cv2.imwrite(save_path_grid, camera_and_grid)
        
        print(f"Saved visualization results to:\n- {save_path_map}\n- {save_path_grid}")
    else:
        print("Failed to run KIT Scenes visualization.")
