import yaml
import torch
import pytest
import os
from unittest.mock import patch, MagicMock

from Tools.trajectory_visualization.checkpoint_loader import load_checkpoint

@pytest.fixture
def dummy_checkpoint_path(tmp_path):
    # Create a dummy config.yaml
    config_path = tmp_path / "config.yaml"
    config_data = {
        "backbone": "resnet18",
        "planner_mode": "bezier",
        "embed_dim": 16,
        "num_views": 4
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    # Create a dummy state dict file (.pt)
    state_dict_path = tmp_path / "model_weights.pt"
    
    # We simulate a nested state_dict with a DDP module prefix to test all logic branches
    dummy_state_dict = {
        "state_dict": {
            "module.layer1.weight": torch.tensor([1.0, 2.0]),
            "module.layer1.bias": torch.tensor([0.0])
        }
    }
    torch.save(dummy_state_dict, state_dict_path)
    
    return str(state_dict_path)


@patch("Tools.trajectory_visualization.checkpoint_loader.AutoE2E")
def test_load_checkpoint_success(mock_autoe2e, dummy_checkpoint_path):
    device = torch.device("cpu")
    
    # Configure the mock model
    mock_model_instance = MagicMock()
    mock_autoe2e.return_value = mock_model_instance
    
    # Run the function
    model = load_checkpoint(dummy_checkpoint_path, device)
    
    # Verify AutoE2E was instantiated with the correct params from config.yaml
    mock_autoe2e.assert_called_once_with(
        backbone="resnet18",
        num_views=4,
        embed_dim=16,
        planner_mode="bezier"
    )
    
    # Verify load_state_dict was called with cleaned keys (no 'module.' prefix)
    expected_clean_state_dict = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.0])
    }
    
    # Check that load_state_dict was called
    mock_model_instance.load_state_dict.assert_called_once()
    loaded_dict = mock_model_instance.load_state_dict.call_args[0][0]
    
    assert "layer1.weight" in loaded_dict
    assert torch.equal(loaded_dict["layer1.weight"], expected_clean_state_dict["layer1.weight"])
    assert "layer1.bias" in loaded_dict
    assert torch.equal(loaded_dict["layer1.bias"], expected_clean_state_dict["layer1.bias"])
    
    # Verify eval mode and device assignment
    mock_model_instance.to.assert_called_once_with(device)
    mock_model_instance.eval.assert_called_once()
    
    assert model == mock_model_instance


def test_load_checkpoint_missing_config(tmp_path):
    device = torch.device("cpu")
    # No config.yaml created, but trying to load a .pt from tmp_path
    pt_file = str(tmp_path / "model.pt")
    with pytest.raises(FileNotFoundError, match="Expected config.yaml in"):
        load_checkpoint(pt_file, device)


@patch("Tools.trajectory_visualization.checkpoint_loader.AutoE2E")
def test_load_checkpoint_missing_weights(mock_autoe2e, tmp_path):
    device = torch.device("cpu")
    
    # Create config.yaml but no .pt file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({}, f)
        
    pt_file = str(tmp_path / "model.pt")
    with pytest.raises(FileNotFoundError, match="No such file or directory"): # Since torch.load will fail
        load_checkpoint(pt_file, device)
