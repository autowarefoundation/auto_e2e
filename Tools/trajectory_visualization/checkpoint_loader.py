import os
import torch
import sys

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Model.model_components.auto_e2e import AutoE2E

from typing import Tuple

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[AutoE2E, dict]:
    """
    Reconstructs the AutoE2E model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file containing config and model_state_dict.
        device: torch.device to load the model to.
        
    Returns:
        Tuple[AutoE2E, dict]: The reconstructed model and its configuration dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing the required 'config' key.")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing the required 'model_state_dict' key.")
        
    config = checkpoint["config"]
    
    try:
        # Initialize the model with the exact config from the checkpoint.
        # This will fail explicitly if required configurations are missing or incompatible.
        model = AutoE2E(**config)
    except TypeError as e:
        raise ValueError(f"Failed to initialize AutoE2E with the provided config: {e}") from e
    
    state_dict = checkpoint["model_state_dict"]
    
    # Handle DataParallel/DDP module prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return model, config
