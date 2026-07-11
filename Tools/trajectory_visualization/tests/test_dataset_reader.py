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
    mock_iter = iter([1, 2, 3])
    mock_loader_instance.__iter__.return_value = mock_iter
    
    dataset_dir = "/dummy/path"
    iterator = get_dataset_iterator(dataset_dir)
    
    # Verify make_pre_extracted_loader was called with correct arguments
    mock_make_loader.assert_called_once_with(
        shard_dir=dataset_dir,
        batch_size=1,
        num_workers=1,
        split="eval",
        shuffle=0,
        return_visualization_image=True
    )
    
    # Verify iterator works as expected
    assert next(iterator) == 1
    assert next(iterator) == 2
    assert next(iterator) == 3
