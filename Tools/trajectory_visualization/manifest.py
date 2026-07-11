import json
import os
from typing import Dict, Any

class ManifestWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.data: Dict[str, Any] = {
            "videos": [],
            "thumbnails": [],
            "metrics": {},
            "metadata": {}
        }

    def add_video(self, video_path: str, num_frames: int):
        self.data["videos"].append({
            "path": os.path.relpath(video_path, self.output_dir),
            "num_frames": num_frames
        })

    def add_thumbnail(self, thumbnail_path: str):
        self.data["thumbnails"].append(os.path.relpath(thumbnail_path, self.output_dir))

    def add_metric(self, key: str, value: Any):
        self.data["metrics"][key] = value
        
    def add_metadata(self, key: str, value: Any):
        self.data["metadata"][key] = value

    def write(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=4)
