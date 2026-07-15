import os
import sys
from collections import defaultdict

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Model.data_parsing.pre_extracted import make_pre_extracted_loader

import json

def get_dataset_manifest(dataset_dir: str) -> dict:
    """Reads the dataset manifest. Fails strictly if missing."""
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}. The tool cannot derive contract properties.")
    with open(manifest_path, 'r') as f:
        return json.load(f)

def get_dataset_iterator(dataset_dir: str, scene_selection: list[dict] | None = None, global_max_frames: int | None = None):
    """
    Initializes a WebDataset reader for the trajectory visualization.
    It reads the pre-extracted data sequentially, grouping by episode, and 
    explicitly sorting by frame index to guarantee the temporal sequence.
    
    Args:
        dataset_dir: Path to directory containing .tar shard files.
        scene_selection: Optional list of dicts specifying episodes and frame ranges to render.
        global_max_frames: Optional max frames per episode to fallback to if scene_selection is not provided.
        
    Returns:
        iterator: An iterator yielding batches (of size 1) from the dataset in chronological order.
    """
    loader = make_pre_extracted_loader(
        shard_dir=dataset_dir,
        batch_size=1,
        num_workers=0,  # Ensure all items can be safely gathered from single process
        split="eval",
        shuffle=0,       # Disable shuffle
        return_visualization_image=True
    )
    
    episodes = defaultdict(list)
    
    # Store geometry properties so we can attach them to the returned iterator
    projection = getattr(loader, "projection", None)
    geometry_type = getattr(loader, "geometry_type", "pseudo")
    
    for batch in loader:
        if "episode_index" not in batch or "frame_index" not in batch:
            raise ValueError("Dataset shard is missing required temporal metadata (episode_index or frame_index).")

        # episode_index could be a tensor or a list
        if isinstance(batch["episode_index"], list):
            ep_id = batch["episode_index"][0]
        else:
            ep_id = batch["episode_index"]
            if hasattr(ep_id, "item"):
                ep_id = ep_id.item()
        
        # frame_index could be a tensor or a list
        if isinstance(batch["frame_index"], list):
            frame_idx = batch["frame_index"][0]
        else:
            frame_idx = batch["frame_index"]
            if hasattr(frame_idx, "item"):
                frame_idx = frame_idx.item()
                
        # We don't filter immediately by episode here because WebDataset iteration
        # is sequential anyway. We collect all (or we could pre-filter if memory is an issue).
        # We will filter in sorted_generator instead to ensure exact bounds matching.
        episodes[ep_id].append((frame_idx, batch))
        
    def sorted_generator():
        
        selection_map = None
        if scene_selection is not None:
            selection_map = {}
            for scene in scene_selection:
                ep_id_str = str(scene["episode_id"])
                selection_map[ep_id_str] = {
                    "start_frame": scene.get("start_frame", 0),
                    "end_frame": scene.get("end_frame", float('inf'))
                }

        # Sort by episode ID to have consistent ordering
        for ep_id in sorted(episodes.keys(), key=str):
            ep_str = str(ep_id)
            
            # Apply selection map filtering at episode level
            if selection_map is not None and ep_str not in selection_map:
                continue
                
            samples = episodes[ep_id]
            samples.sort(key=lambda x: x[0])  # explicitly sort by frame_index
            
            frames_yielded = 0
            for frame_idx, batch in samples:
                # Apply selection map filtering at frame level
                if selection_map is not None:
                    limits = selection_map[ep_str]
                    if frame_idx < limits["start_frame"] or frame_idx > limits["end_frame"]:
                        continue
                else:
                    # Apply global fallback
                    if global_max_frames is not None and frames_yielded >= global_max_frames:
                        break
                        
                yield batch
                frames_yielded += 1

    class DatasetIteratorWrapper:
        def __init__(self, gen, proj, geom):
            self.gen = gen
            self.projection = proj
            self.geometry_type = geom
        def __iter__(self):
            return self.gen
        def __next__(self):
            return next(self.gen)

    return DatasetIteratorWrapper(sorted_generator(), projection, geometry_type)
