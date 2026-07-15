import os
import sys
from collections import defaultdict

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Model.data_parsing.pre_extracted import make_pre_extracted_loader

def get_dataset_iterator(dataset_dir: str, episodes_to_render: list[str] | None = None):
    """
    Initializes a WebDataset reader for the trajectory visualization.
    It reads the pre-extracted data sequentially, grouping by episode, and 
    explicitly sorting by frame index to guarantee the temporal sequence.
    
    Args:
        dataset_dir: Path to directory containing .tar shard files.
        episodes_to_render: Optional list of episode string identifiers to include.
        
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
                
        # Filter if requested episodes are provided
        if episodes_to_render is not None:
            # Check against string/int variants just in case
            if str(ep_id) not in [str(e) for e in episodes_to_render]:
                continue
                
        episodes[ep_id].append((frame_idx, batch))
        
    def sorted_generator():
        # Sort by episode ID to have consistent ordering
        for ep_id in sorted(episodes.keys(), key=str):
            samples = episodes[ep_id]
            samples.sort(key=lambda x: x[0])  # explicitly sort by frame_index
            for frame_idx, batch in samples:
                yield batch

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
