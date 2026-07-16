import json
import os
from typing import Dict, Any

class ManifestWriter:
    """
    Handles the creation and updating of the visualization manifest JSON file.
    """
    def __init__(self, output_dir: str, checkpoint_name: str, model_config: dict, dataset_name: str, dataset_version: str):
        """
        Initializes the ManifestWriter with metadata.
        
        Args:
            output_dir (str): Directory where the manifest and videos will be saved.
            checkpoint_name (str): Name of the model checkpoint used.
            model_config (dict): The configuration dictionary of the model.
            dataset_name (str): Name of the dataset being visualized.
            dataset_version (str): Version of the dataset being visualized.
        """
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            "checkpoint": {
                "name": checkpoint_name,
                "model_config": model_config
            },
            "dataset": {
                "name": dataset_name,
                "version": dataset_version
            },
            "episodes": []
        }

    def add_episode(self, episode_id: int, start_frame: int, end_frame: int):
        """
        Registers a processed episode in the manifest.
        
        Args:
            episode_id (int): The integer ID of the episode.
            start_frame (int): The starting frame index of the rendered video.
            end_frame (int): The ending frame index of the rendered video.
        """
        # Format episode directory name
        ep_dir = f"episode-{episode_id:06d}"
        
        self.data["episodes"].append({
            "episode_id": episode_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "video": f"episodes/{ep_dir}/video.mp4",
            "thumbnail": f"episodes/{ep_dir}/thumbnail.jpg",
            "metrics": f"episodes/{ep_dir}/metrics.json"
        })

    def write(self):
        """
        Writes the current state of the manifest data to the manifest.json file.
        """
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=4)
