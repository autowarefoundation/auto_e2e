import os
import json
import tempfile

from Tools.trajectory_visualization.manifest import ManifestWriter

def test_manifest_writer():
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = ManifestWriter(
            output_dir=temp_dir,
            checkpoint_name="best.pt",
            model_config={"layers": 4},
            dataset_name="yaak-ai/L2D",
            dataset_version="v1"
        )
        
        # Test adding items
        writer.add_episode(episode_id=42, start_frame=100, end_frame=399)
        writer.add_episode(episode_id=108, start_frame=0, end_frame=150)
        
        # Write to file
        writer.write()
        
        manifest_path = os.path.join(temp_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        
        # Verify contents
        with open(manifest_path, 'r') as f:
            data = json.load(f)
            
        assert data["schema_version"] == 1
        assert data["checkpoint"]["name"] == "best.pt"
        assert data["checkpoint"]["model_config"] == {"layers": 4}
        assert data["dataset"]["name"] == "yaak-ai/L2D"
        
        assert "episodes" in data
        assert len(data["episodes"]) == 2
        
        ep42 = data["episodes"][0]
        assert ep42["episode_id"] == 42
        assert ep42["start_frame"] == 100
        assert ep42["end_frame"] == 399
        assert ep42["video"] == "episodes/episode-000042/video.mp4"
        assert ep42["thumbnail"] == "episodes/episode-000042/thumbnail.jpg"
        assert ep42["metrics"] == "episodes/episode-000042/metrics.json"

