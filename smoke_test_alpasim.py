import torch
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Model/plugins')))
from alpasim_driver.plugin import AutoE2EDriver, PredictionInput

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from Tools.trajectory_visualization.rendering import render_frame, trajectory_extent
from Tools.trajectory_visualization.artifacts import ShardSample
import io

class DummyAutoE2EModel(torch.nn.Module):
    def forward(self, tensors):
        # Generate some realistic-looking dummy trajectory points (e.g. a curve)
        t = torch.linspace(0, 20, 64)
        x = t
        y = 0.5 * t ** 2
        points = torch.stack([x, y], dim=1).unsqueeze(0)  # shape (1, 64, 2)
        headings = torch.atan2(t, torch.ones_like(t)).unsqueeze(0)  # shape (1, 64)
        return {
            "trajectory_points": points,
            "headings": headings
        }

torch.serialization.add_safe_globals([DummyAutoE2EModel])

def create_dummy_checkpoint(ckpt_path):
    model = DummyAutoE2EModel()
    torch.save(model, ckpt_path)

def generate_mock_prediction_input():
    camera_names = [
        "camera_base_front_center",
        "camera_ring_front",
        "camera_ring_front_left",
        "camera_ring_front_right",
        "camera_ring_rear",
        "camera_ring_rear_left",
        "camera_ring_rear_right",
    ]
    cameras = {}
    for name in camera_names:
        cameras[name] = Image.new("RGB", (256, 256), color="gray")
    
    return PredictionInput(
        cameras=cameras,
        speed=10.0,
        acceleration=0.5,
        command=1
    )

def main():
    ckpt_path = "dummy_random.ckpt"
    create_dummy_checkpoint(ckpt_path)
    print(f"Created dummy checkpoint at {ckpt_path}")
    
    driver = AutoE2EDriver(model_checkpoint=ckpt_path)
    print("Initialized AutoE2EDriver")
    
    mock_input = generate_mock_prediction_input()
    prediction = driver.predict(mock_input)
    print("Executed predict()")
    
    points = prediction.trajectory_points
    headings = prediction.headings
    print(f"Trajectory points shape: {points.shape}")
    print(f"Headings shape: {headings.shape}")
    
    extent = trajectory_extent([points])
    empty_target = np.zeros((0, 2), dtype=np.float32)

    blank = Image.new("RGB", (1280, 720), color="black")
    buf = io.BytesIO()
    blank.save(buf, format="JPEG")
    camera_jpeg = buf.getvalue()

    calibration = {
        "projection": {
            "type": "pinhole",
            "matrix": [
                [
                    [1000.0, 0.0, 640.0, 0.0],
                    [0.0, 1000.0, 360.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            ]
        },
        "dataset": "kitscenes"
    }

    sample = ShardSample(
        sample_uid="smoke_test_sample",
        scene_uid="smoke_test_scene",
        frame_idx=0,
        dataset="kitscenes",
        camera_jpeg=camera_jpeg,
        initial_speed=10.0,
        target_controls=empty_target,
        calibration=calibration
    )

    frame_image = render_frame(
        sample,
        prediction=points,
        target=empty_target,
        v0=10.0,
        base_seed=0,
        extent=extent,
        camera_index=0
    )
    
    out_img = "smoke_test_evidence.png"
    frame_image.save(out_img)

    print(f"Saved visual evidence to {out_img}")

if __name__ == "__main__":
    main()
