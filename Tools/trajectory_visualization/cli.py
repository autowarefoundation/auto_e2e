import argparse
import sys
import os

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Tools.trajectory_visualization.runner import run_visualization

def main():
    parser = argparse.ArgumentParser(description="Trajectory Visualization Tool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the processed evaluation dataset directory (.tar shards).")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to write output videos and manifest.")
    parser.add_argument("--episodes", type=int, nargs='+', help="List of episode indices to process.")
    parser.add_argument("--max-frames-per-episode", type=int, default=300, help="Number of frames to render per episode.")
    
    args = parser.parse_args()
    
    run_visualization(
        checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        max_frames_per_episode=args.max_frames_per_episode
    )

if __name__ == "__main__":
    main()
