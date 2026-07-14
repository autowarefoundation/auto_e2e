import os
import cv2
import torch
from tqdm import tqdm

from .checkpoint_loader import load_checkpoint
from .dataset_reader import get_dataset_iterator
from .manifest import ManifestWriter
from .rendering import generate_grid, concatenate_grid_and_camera
from .kinematics import accel_and_curv_to_meters_trajectory

def run_visualization(checkpoint: str, dataset_dir: str, output_dir: str, episodes: list = None, max_frames_per_episode: int = 300):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {checkpoint}...")
    model = load_checkpoint(checkpoint, device)
    
    print(f"Loading dataset from {dataset_dir}...")
    data_iterator = get_dataset_iterator(dataset_dir)
    
    manifest = ManifestWriter(output_dir)
    
    video_path = os.path.join(output_dir, "trajectory_video.mp4")
    thumbnail_path = os.path.join(output_dir, "thumbnail.jpg")
    
    video_writer = None
    
    num_frames_to_render = max_frames_per_episode
    if episodes is not None:
        num_frames_to_render = len(episodes) * max_frames_per_episode
        print(f"Rendering {num_frames_to_render} frames based on {len(episodes)} episodes.")
    else:
        print(f"Rendering up to {num_frames_to_render} frames.")
    
    print("Running inference and rendering...")
    frames_processed = 0
    with torch.no_grad():
        for batch in tqdm(data_iterator, total=num_frames_to_render):
            if frames_processed >= num_frames_to_render:
                break
                
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
            
            pred_m = accel_and_curv_to_meters_trajectory(pred_seq, current_speed, 64)
            act_m = accel_and_curv_to_meters_trajectory(target_seq, current_speed, 64)
            
            grid_img = generate_grid(prediction_m=pred_m, actual_trajectory_m=act_m)
            
            # Extract front camera image from unnormalized visualization representations
            viz_images = batch["visualization_image"] # (Batch, NumCams, H, W, 3)
            cam_img = viz_images[0, 0].numpy() # Batch 0, Camera 0
            
            # Combine
            final_frame = concatenate_grid_and_camera(grid_img, cam_img)
            
            # Initialize video writer on first frame
            if video_writer is None:
                h, w = final_frame.shape[:2]
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
            
            video_writer.write(final_frame)
     
            # Save first frame as thumbnail
            if frames_processed == 0:
                cv2.imwrite(thumbnail_path, final_frame)
                manifest.add_thumbnail(thumbnail_path)
            
            frames_processed += 1
            
    if video_writer is not None:
        video_writer.release()
        
    manifest.add_video(video_path, frames_processed)
    
    # Optional: populate metadata
    manifest.add_metadata("checkpoint", os.path.basename(checkpoint))
    manifest.add_metadata("dataset", os.path.basename(dataset_dir))
    
    manifest.write()
    print(f"Artifacts saved to {output_dir}")
