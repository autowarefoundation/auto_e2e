import sys
from unittest.mock import patch, MagicMock

sys.modules['webdataset'] = MagicMock()

from Tools.trajectory_visualization.cli import main  # noqa: E402
@patch("Tools.trajectory_visualization.cli.run_visualization")
def test_cli_main(mock_run_visualization):
    test_args = [
        "cli.py",
        "--checkpoint", "/path/to/ckpt/best.pt",
        "--dataset-dir", "/path/to/data",
        "--output-dir", "/path/to/out",
        "--episodes", "0", "1", "2",
        "--max-frames-per-episode", "50"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_run_visualization.assert_called_once_with(
        checkpoint="/path/to/ckpt/best.pt",
        dataset_dir="/path/to/data",
        output_dir="/path/to/out",
        episodes=[0, 1, 2],
        max_frames_per_episode=50
    )

@patch("Tools.trajectory_visualization.cli.run_visualization")
def test_cli_main_defaults(mock_run_visualization):
    test_args = [
        "cli.py",
        "--checkpoint", "/path/to/ckpt/best.pt",
        "--dataset-dir", "/path/to/data",
        "--output-dir", "/path/to/out"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()
        
    mock_run_visualization.assert_called_once_with(
        checkpoint="/path/to/ckpt/best.pt",
        dataset_dir="/path/to/data",
        output_dir="/path/to/out",
        episodes=None,
        max_frames_per_episode=300
    )
