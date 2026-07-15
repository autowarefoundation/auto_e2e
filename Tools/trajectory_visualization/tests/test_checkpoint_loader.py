import torch
import pytest
from unittest.mock import patch, MagicMock

from Tools.trajectory_visualization.checkpoint_loader import load_checkpoint

@pytest.fixture
def dummy_checkpoint_path(tmp_path):
    # Create a dummy state dict file (.pt) containing config and model_state_dict
    state_dict_path = tmp_path / "model_weights.pt"
    
    config_data = {
        "backbone": "resnet18",
        "planner_mode": "bezier",
        "embed_dim": 16,
        "num_views": 4
    }
    
    # We simulate a nested state_dict with a DDP module prefix to test all logic branches
    dummy_state_dict = {
        "module.layer1.weight": torch.tensor([1.0, 2.0]),
        "module.layer1.bias": torch.tensor([0.0])
    }
    
    checkpoint_data = {
        "config": config_data,
        "model_state_dict": dummy_state_dict
    }
    torch.save(checkpoint_data, state_dict_path)
    
    return str(state_dict_path)


@patch("Tools.trajectory_visualization.checkpoint_loader.AutoE2E")
def test_load_checkpoint_success(mock_autoe2e, dummy_checkpoint_path):
    device = torch.device("cpu")
    
    # Configure the mock model
    mock_model_instance = MagicMock()
    mock_autoe2e.return_value = mock_model_instance
    
    # Run the function
    model, config = load_checkpoint(dummy_checkpoint_path, device)
    
    # Verify AutoE2E was instantiated with the exact config from checkpoint
    mock_autoe2e.assert_called_once_with(
        backbone="resnet18",
        num_views=4,
        embed_dim=16,
        planner_mode="bezier"
    )
    
    # Verify load_state_dict was called with cleaned keys (no 'module.' prefix) and strict=True
    expected_clean_state_dict = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.0])
    }
    
    # Check that load_state_dict was called
    mock_model_instance.load_state_dict.assert_called_once()
    loaded_dict = mock_model_instance.load_state_dict.call_args[0][0]
    strict_arg = mock_model_instance.load_state_dict.call_args[1].get('strict', None)
    
    assert strict_arg is True
    
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
    pt_file = tmp_path / "model.pt"
    
    checkpoint_data = {
        "model_state_dict": {}
    }
    torch.save(checkpoint_data, pt_file)
    
    with pytest.raises(ValueError, match="missing the required 'config' key"):
        load_checkpoint(str(pt_file), device)


def test_load_checkpoint_missing_model_state_dict(tmp_path):
    device = torch.device("cpu")
    pt_file = tmp_path / "model.pt"
    
    checkpoint_data = {
        "config": {}
    }
    torch.save(checkpoint_data, pt_file)
    
    with pytest.raises(ValueError, match="missing the required 'model_state_dict' key"):
        load_checkpoint(str(pt_file), device)


def test_load_checkpoint_file_not_found(tmp_path):
    device = torch.device("cpu")
    pt_file = tmp_path / "missing.pt"
    
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint(str(pt_file), device)
