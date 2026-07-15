import os
import cv2
import torch
import json

from .checkpoint_loader import load_checkpoint
from .dataset_reader import get_dataset_iterator, get_dataset_manifest
from .manifest import ManifestWriter
from .rendering import generate_grid, concatenate_grid_and_camera
from .kinematics import controls_to_metric_trajectory, ModelOutputContract

def run_visualization(checkpoint: str, dataset_dir: str, output_dir: str, episodes: list[str] | None = None, max_frames_per_episode: int = 300, selection_manifest: str | None = None):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {checkpoint}...")
    model, model_config = load_checkpoint(checkpoint, device)
    
    print(f"Loading dataset from {dataset_dir}...")
    dataset_manifest = get_dataset_manifest(dataset_dir)
    
    scene_selection = None
    if selection_manifest:
        with open(selection_manifest, 'r') as f:
            manifest_data = json.load(f)
        scene_selection = manifest_data.get("scenes", [])
    elif episodes is not None:
        scene_selection = [{"episode_id": str(ep), "start_frame": 0, "end_frame": max_frames_per_episode - 1} for ep in episodes]
    else:
        # If no scenes specified, we still want to bound each episode by max_frames_per_episode.
        # We can't pre-populate scene_selection if we don't know the episodes, but the dataset reader
        # can enforce max_frames_per_episode globally if we pass it, or we enforce it here.
        pass

    data_iterator = get_dataset_iterator(
        dataset_dir, 
        scene_selection=scene_selection,
        global_max_frames=max_frames_per_episode if scene_selection is None else None
    )
    
    contract = ModelOutputContract.from_config_and_manifest(model_config, dataset_manifest)
    
    # For now, hardcode or guess dataset details. 
    # In a real setup, we might read dataset metadata.json
    dataset_name = os.path.basename(dataset_dir.strip('/'))
    dataset_version = "latest"
    
    manifest = ManifestWriter(
        output_dir=output_dir,
        checkpoint_name=os.path.basename(checkpoint),
        model_config=model_config,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )
    
    # Save a run-summary.json
    with open(os.path.join(output_dir, "run-summary.json"), "w") as f:
        json.dump({"status": "in_progress"}, f)
    
    video_writer = None
    
    print("Running inference and rendering...")
    current_episode = None
    frames_in_current_episode = 0
    total_frames_processed = 0
    ep_start_frame = 0
    ep_dir_path = ""
    frames_dir = ""
    
    with torch.no_grad():
        for batch in data_iterator:
            ep_id = batch.get("episode_index", [0])[0] if isinstance(batch.get("episode_index"), list) else batch.get("episode_index", 0)
            
            if ep_id != current_episode:
                # Close previous episode
                if current_episode is not None:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    manifest.add_episode(current_episode, ep_start_frame, ep_start_frame + frames_in_current_episode - 1)
                
                # Setup new episode
                current_episode = ep_id
                frames_in_current_episode = 0
                ep_start_frame = batch.get("frame_index", [0])[0] if isinstance(batch.get("frame_index"), list) else batch.get("frame_index", 0)
                
                ep_dir_path = os.path.join(output_dir, "episodes", f"episode-{ep_id:06d}")
                frames_dir = os.path.join(ep_dir_path, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                # Write empty metrics.json
                with open(os.path.join(ep_dir_path, "metrics.json"), "w") as f:
                    json.dump({}, f)
                
            visual_tiles = batch["visual_tiles"].to(device)
            visual_history = batch["visual_history"].to(device)
            egomotion_history = batch["egomotion_history"].to(device)
            trajectory_target = batch["trajectory_target"].to(device)
            
            # Forward pass
            output = model(
                camera_tiles=visual_tiles,
                map_input=None,
                visual_history=visual_history,
                egomotion_history=egomotion_history,
                mode="infer"
            )
            
            # Handle tuple output
            pred_trajectory = output if isinstance(output, torch.Tensor) else output[0]
            
            # Convert to CPU for rendering
            pred_seq = pred_trajectory[0].cpu()
            target_seq = trajectory_target[0].cpu()
            
            # Current speed (placeholder, since it's not strictly extracted)
            current_speed = 0.0
            
            pred_xy = controls_to_metric_trajectory(pred_seq, current_speed, contract=contract)
            act_xy = controls_to_metric_trajectory(target_seq, current_speed, contract=contract)
            
            grid_img = generate_grid(prediction_xy=pred_xy, target_xy=act_xy)
            
            # Extract front camera image from unnormalized visualization representations
            viz_images = batch["visualization_image"] # (Batch, NumCams, H, W, 3)
            cam_img = viz_images[0, 0].numpy() # Batch 0, Camera 0
            
            # Combine
            final_frame = concatenate_grid_and_camera(grid_img, cam_img)
            
            # Initialize video writer on first frame
            if video_writer is None:
                h, w = final_frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(os.path.join(ep_dir_path, "video.mp4"), fourcc, 10.0, (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError("Failed to initialize video codec. The requested video codec (mp4v) may be unavailable.")
            
            video_writer.write(final_frame)
            
            # Save frame image
            cv2.imwrite(os.path.join(frames_dir, f"{frames_in_current_episode:06d}.jpg"), final_frame)
            
            # Save first frame as thumbnail
            if frames_in_current_episode == 0:
                cv2.imwrite(os.path.join(ep_dir_path, "thumbnail.jpg"), final_frame)
            
            frames_in_current_episode += 1
            total_frames_processed += 1
            print(f"\rProcessed frame {frames_in_current_episode} of episode {current_episode}", end="")
            
        if total_frames_processed == 0:
            raise ValueError(f"No frames were found or selected in the dataset {dataset_dir}. Empty output sequences.")

    # Cleanup last episode
    if current_episode is not None:
        if video_writer is not None:
            video_writer.release()
        manifest.add_episode(current_episode, ep_start_frame, ep_start_frame + frames_in_current_episode - 1)
        
    manifest.write()
    
    # Update run-summary.json
    with open(os.path.join(output_dir, "run-summary.json"), "w") as f:
        json.dump({"status": "completed", "episodes_processed": len(manifest.data["episodes"])}, f)
        
    print(f"Artifacts saved to {output_dir}")
