import os
import json
import tempfile

from Tools.trajectory_visualization.manifest import ManifestWriter

def test_manifest_writer():
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ManifestWriter(temp_dir)
        
        # Test adding items
        writer.add_video("/test/path/video.mp4", 100)
        writer.add_thumbnail("/test/path/thumb.jpg")
        writer.add_metric("accuracy", 0.95)
        writer.add_metadata("model", "test_model")
        
        # Write to file
        writer.write()
        
        manifest_path = os.path.join(temp_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        
        # Verify contents
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            
        assert "videos" in data
        assert len(data["videos"]) == 1
        assert data["videos"][0]["num_frames"] == 100
        
        assert "thumbnails" in data
        assert len(data["thumbnails"]) == 1
        
        assert "metrics" in data
        assert data["metrics"]["accuracy"] == 0.95
        
        assert "metadata" in data
        assert data["metadata"]["model"] == "test_model"
