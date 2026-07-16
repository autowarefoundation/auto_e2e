"""
Usage:
    PYTHONPATH=. python Tools/trajectory_visualization/Kit_Scenes_visualization/generate_kitscenes_video.py
        --scene_ids <uuid>
        --dataset_root Tools/trajectory_visualization/Kit_Scenes_visualization/data
        --num_frames 30

    Note: The dataset_root can also be specified via the KITSCENES_ROOT environment variable.
"""

import os
import cv2
import argparse
import numpy as np
import subprocess

from Tools.trajectory_visualization.Kit_Scenes_visualization.kit_scenes_visualizer import visualization_on_kit_scenes

def main():
    """
    Entry point for generating a KIT Scenes visualization video.
    
    Parses CLI arguments, retrieves scene datasets, runs inference frame by frame, 
    and encodes the output into a video file using ffmpeg.
    """
    parser = argparse.ArgumentParser(description='KIT Scenes video generation')
    parser.add_argument('--dataset_root', type=str, default=None, help='Path to KIT Scenes dataset root. Defaults to $KITSCENES_ROOT if not provided.')
    parser.add_argument('--scene_ids', type=str, nargs='+', default=None, help='List of scene IDs to load')
    parser.add_argument('--num_frames', type=int, default=30, help='Number of frames to visualize')
    parser.add_argument("--zoom_in", action="store_true", help="Zoom in on the agent")
    args = parser.parse_args()

    dataset_root = args.dataset_root or os.environ.get("KITSCENES_ROOT")
    if dataset_root is None:
        raise ValueError("Dataset root must be provided via --dataset_root or KITSCENES_ROOT environment variable.")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "frames")
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images", "kitscenes_video.mp4")
    
    # Clean old frames
    for f in os.listdir(save_dir):
        if f.endswith('.jpg'):
            os.remove(os.path.join(save_dir, f))
    
    for frame_idx in range(args.num_frames):
        print(f"Generating frame {frame_idx} / {args.num_frames}...")
        try:
            combined_image, camera_and_grid = visualization_on_kit_scenes(
                scene_ids=args.scene_ids, 
                frame_index=frame_idx, 
                zoom_in=args.zoom_in, 
                dataset_root=dataset_root
            )
            
            if combined_image is None or camera_and_grid is None:
                print(f"Skipping frame {frame_idx} (failed to render)")
                continue
                
            # Combine them horizontally to form one video frame
            target_h = max(combined_image.shape[0], camera_and_grid.shape[0])
            
            # Resize vertically to match if necessary
            if combined_image.shape[0] != target_h:
                w = int(combined_image.shape[1] * (target_h / combined_image.shape[0]))
                combined_image = cv2.resize(combined_image, (w, target_h))
            if camera_and_grid.shape[0] != target_h:
                w = int(camera_and_grid.shape[1] * (target_h / camera_and_grid.shape[0]))
                camera_and_grid = cv2.resize(camera_and_grid, (w, target_h))
                
            final_frame = np.concatenate([combined_image, camera_and_grid], axis=1)
            
            frame_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, final_frame)
            
        except Exception as e:
            print(f"Error on frame {frame_idx}: {e}")
            break

    print("Encoding with ffmpeg...")
    subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-pattern_type", "glob", "-i", f"{save_dir}/*.jpg", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path])
    print(f"Successfully saved video to {video_path}")

if __name__ == "__main__":
    main()
