import os
import glob
import yaml
import torch
import sys

# Assume script is run from Tools/trajectory_visualization/ or similar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Model.model_components.auto_e2e import AutoE2E

def load_checkpoint(checkpoint_path: str, device: torch.device) -> AutoE2E:
    """
    Reconstructs the AutoE2E model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file. The directory must contain config.yaml.
        device: torch.device to load the model to.
        
    Returns:
        AutoE2E: The reconstructed model.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Expected config.yaml in {checkpoint_dir}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Extract params with defaults based on AutoE2E initialization
    backbone = config.get('backbone', 'swin_v2_tiny')
    planner_mode = config.get('planner_mode', 'bezier')
    embed_dim = config.get('embed_dim', 8)
    num_views = config.get('num_views', 8)
    
    # Initialize the model
    model = AutoE2E(
        backbone=backbone,
        num_views=num_views,
        embed_dim=embed_dim,
        planner_mode=planner_mode
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle if state dict is wrapped in 'state_dict' or 'model' key
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
        
    # Handle DataParallel/DDP module prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    return model
