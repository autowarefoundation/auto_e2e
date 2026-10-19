import os
import sys
import torch
import numpy as np
from PIL import Image

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_DRIVER_DIR = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
_PLUGINS_DIR = os.path.abspath(os.path.join(_DRIVER_DIR, ".."))
_MODEL_DIR = os.path.abspath(os.path.join(_PLUGINS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, ".."))

for path in [_REPO_ROOT, _MODEL_DIR, _PLUGINS_DIR, _DRIVER_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from alpasim_driver.plugin import AutoE2EDriver, PredictionInput  # noqa: E402
from Tools.trajectory_visualization.rendering import render_frame, trajectory_extent  # noqa: E402
from Tools.trajectory_visualization.artifacts import ShardSample  # noqa: E402
import io  # noqa: E402

from model_components.auto_e2e import AutoE2E  # noqa: E402

def create_model_checkpoint(ckpt_path: str) -> None:
    model = AutoE2E(num_views=7, is_pretrained=False)
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
    camera_images = {}
    for name in camera_names:
        camera_images[name] = Image.new("RGB", (256, 256), color="gray")
    
    return PredictionInput(
        camera_images=camera_images,
        speed=10.0,
        acceleration=0.5,
        command=1
    )

def main():
    ckpt_path = "dummy_random.ckpt"
    create_model_checkpoint(ckpt_path)
    print(f"Created model checkpoint at {ckpt_path}")
    
    driver = AutoE2EDriver(model_checkpoint=ckpt_path, allow_mock=False)
    print("Initialized AutoE2EDriver")
    
    mock_input = generate_mock_prediction_input()
    prediction = driver.predict(mock_input)
    print("Executed predict()")
    
    points = prediction.trajectory_xy
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
