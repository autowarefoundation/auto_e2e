import sys
from unittest.mock import patch, MagicMock

# Mock webdataset before it gets imported by dataset_reader -> pre_extracted
sys.modules['webdataset'] = MagicMock()

from Tools.trajectory_visualization.dataset_reader import get_dataset_iterator  # noqa: E402

@patch("Tools.trajectory_visualization.dataset_reader.make_pre_extracted_loader")
def test_get_dataset_iterator(mock_make_loader):
    # Setup mock iterator
    mock_loader_instance = MagicMock()
    mock_make_loader.return_value = mock_loader_instance
    mock_loader_instance.projection = "dummy_projection"
    mock_loader_instance.geometry_type = "dummy_geometry"
    
    mock_iter = iter([
        {"episode_index": [0], "frame_index": [2], "data": "c"},
        {"episode_index": [0], "frame_index": [1], "data": "b"},
        {"episode_index": [0], "frame_index": [0], "data": "a"},
        {"episode_index": [1], "frame_index": [0], "data": "d"}
    ])
    mock_loader_instance.__iter__.return_value = mock_iter

    dataset_dir = "/dummy/path"
    iterator = get_dataset_iterator(dataset_dir)
    
    batches = list(iterator)
    assert len(batches) == 4
    # Check sorting within episode 0
    assert batches[0]["data"] == "a"
    assert batches[1]["data"] == "b"
    assert batches[2]["data"] == "c"
    
    # Check that projection properties are attached
    assert getattr(iterator, "projection", None) == "dummy_projection"
    assert getattr(iterator, "geometry_type", None) == "dummy_geometry"
    
    # Check that it called make_loader correctly
    mock_make_loader.assert_called_once_with(
        shard_dir="/dummy/path",
        batch_size=1,
        num_workers=0,
        split="eval",
        shuffle=0,
        return_visualization_image=True
    )

@patch("Tools.trajectory_visualization.dataset_reader.make_pre_extracted_loader")
def test_get_dataset_iterator_scene_selection(mock_make_loader):
    mock_loader_instance = MagicMock()
    mock_make_loader.return_value = mock_loader_instance
    
    mock_iter = iter([
        {"episode_index": [0], "frame_index": [0], "data": "e0f0"},
        {"episode_index": [0], "frame_index": [1], "data": "e0f1"},
        {"episode_index": [0], "frame_index": [2], "data": "e0f2"},
        {"episode_index": [1], "frame_index": [0], "data": "e1f0"},
        {"episode_index": [2], "frame_index": [0], "data": "e2f0"}
    ])
    mock_loader_instance.__iter__.return_value = mock_iter

    scene_selection = [
        {"episode_id": "0", "start_frame": 1, "end_frame": 2},
        {"episode_id": "2"}
    ]

    iterator = get_dataset_iterator("/dummy", scene_selection=scene_selection)
    batches = list(iterator)
    
    assert len(batches) == 3
    assert batches[0]["data"] == "e0f1"
    assert batches[1]["data"] == "e0f2"
    assert batches[2]["data"] == "e2f0"
