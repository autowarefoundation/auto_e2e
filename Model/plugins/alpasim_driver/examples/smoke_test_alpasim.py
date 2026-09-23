import io
import os
import sys

import numpy as np
import torch
from PIL import Image

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_DRIVER_DIR = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
_PLUGINS_DIR = os.path.abspath(os.path.join(_DRIVER_DIR, ".."))
_MODEL_DIR = os.path.abspath(os.path.join(_PLUGINS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, ".."))

for path in [_REPO_ROOT, _MODEL_DIR, _PLUGINS_DIR, _DRIVER_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


def create_model_checkpoint(ckpt_path: str) -> None:
    from model_components.auto_e2e import AutoE2E

    model = AutoE2E(num_views=6, map_context_channels=14, is_pretrained=False)
    torch.save(
        {
            "config": {
                "num_views": 6,
                "map_context_channels": 14,
                "is_pretrained": False,
            },
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )


def generate_mock_prediction_input():
    from alpasim_driver.models.base import DriveCommand, PredictionInput

    camera_names = [
        "camera_base_front_center",
        "camera_ring_front_left",
        "camera_ring_front_right",
        "camera_ring_rear",
        "camera_ring_rear_left",
        "camera_ring_rear_right",
    ]
    camera_images = {}
    for name in camera_names:
        camera_images[name] = Image.new("RGB", (256, 256), color="gray")

    class MockQuat:
        w: float = 1.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

    class MockPose:
        quat = MockQuat()
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

    class MockPoseAtTime:
        timestamp_us: int = 0
        pose = MockPose()

    return PredictionInput(
        camera_images=camera_images,
        command=DriveCommand.STRAIGHT,
        speed=10.0,
        acceleration=0.5,
        ego_pose_history=[MockPoseAtTime()],
        inference_seed=0,
    )


def main():
    from alpasim_autoe2e.plugin import AutoE2EDriver
    from Tools.trajectory_visualization.artifacts import ShardSample
    from Tools.trajectory_visualization.rendering import (
        render_frame,
        trajectory_extent,
    )

    ckpt_path = "dummy_random.ckpt"
    create_model_checkpoint(ckpt_path)
    print(f"Created model checkpoint at {ckpt_path}")

    driver = AutoE2EDriver(model_checkpoint=ckpt_path, allow_mock=False)
    print("Initialized AutoE2EDriver")

    if driver.parser.rasterizer is None:

        class MockRaster:
            route_mask = np.zeros((2, 256, 256), dtype=np.float32)
            map_context = np.zeros((14, 256, 256), dtype=np.float32)
            route_valid = True

        class MockRasterizer:
            def render(self, nav_map, route, live_pose):
                return MockRaster()

        driver.parser.rasterizer = MockRasterizer()
        driver.parser.route = True

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
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ],
        },
        "dataset": "kitscenes",
    }

    sample = ShardSample(
        sample_uid="smoke_test_sample",
        scene_uid="smoke_test_scene",
        frame_idx=0,
        dataset="kitscenes",
        camera_jpeg=camera_jpeg,
        initial_speed=10.0,
        target_controls=empty_target,
        calibration=calibration,
    )

    frame_image = render_frame(
        sample,
        prediction=points,
        target=empty_target,
        v0=10.0,
        base_seed=0,
        extent=extent,
        camera_index=0,
    )

    out_img = "smoke_test_evidence.png"
    frame_image.save(out_img)

    print(f"Saved visual evidence to {out_img}")


if __name__ == "__main__":
    main()
