import os
import sys

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Model.data_parsing.pre_extracted import make_pre_extracted_loader

def get_dataset_iterator(dataset_dir: str):
    """
    Initializes a WebDataset reader for the trajectory visualization.
    It reads the pre-extracted data sequentially (no shuffle) with batch_size=1.
    
    Args:
        dataset_dir: Path to directory containing .tar shard files.
        
    Returns:
        iterator: An iterator yielding batches (of size 1) from the dataset.
    """
    loader = make_pre_extracted_loader(
        shard_dir=dataset_dir,
        batch_size=1,
        num_workers=1,  # Keep single-threaded to guarantee strict order if needed, or rely on wds defaults
        split="eval",
        shuffle=0,       # Disable shuffle for sequential frame processing
        return_visualization_image=True
    )
    
    return iter(loader)
