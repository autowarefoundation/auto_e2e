from Tools.trajectory_visualization.cli import main  # noqa: E402
from unittest.mock import patch, MagicMock
import sys

sys.modules['webdataset'] = MagicMock()

@patch("Tools.trajectory_visualization.cli.run_visualization")
def test_cli_main(mock_run_visualization):
    test_args = [
        "cli.py",
        "--checkpoint_dir", "/path/to/ckpt",
        "--dataset_dir", "/path/to/data",
        "--output_dir", "/path/to/out",
        "--num_frames", "50"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_run_visualization.assert_called_once_with(
        checkpoint_dir="/path/to/ckpt",
        dataset_dir="/path/to/data",
        output_dir="/path/to/out",
        num_frames_to_render=50
    )

@patch("Tools.trajectory_visualization.cli.run_visualization")
def test_cli_main_defaults(mock_run_visualization):
    test_args = [
        "cli.py",
        "--checkpoint_dir", "/path/to/ckpt",
        "--dataset_dir", "/path/to/data",
        "--output_dir", "/path/to/out"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_run_visualization.assert_called_once_with(
        checkpoint_dir="/path/to/ckpt",
        dataset_dir="/path/to/data",
        output_dir="/path/to/out",
        num_frames_to_render=100
    )
