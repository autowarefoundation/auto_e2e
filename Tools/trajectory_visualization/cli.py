import argparse
import sys
import os

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Tools.trajectory_visualization.runner import run_visualization

def main():
    parser = argparse.ArgumentParser(description="Trajectory Visualization Tool")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory containing config.yaml and .pt file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the processed evaluation dataset directory (.tar shards).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to write output videos and manifest.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to render.")
    
    args = parser.parse_args()
    
    run_visualization(
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_frames_to_render=args.num_frames
    )

if __name__ == "__main__":
    main()
