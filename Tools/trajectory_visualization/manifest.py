import json
import os
from typing import Dict, Any

class ManifestWriter:
    def __init__(self, output_dir: str, checkpoint_name: str, model_config: dict, dataset_name: str, dataset_version: str):
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
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=4)
