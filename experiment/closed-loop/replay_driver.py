"""Replay a rollout's logged driver inputs through AutoE2EDriver, with ablations.

For each selected decision, rebuild the ``PredictionInput`` the driver saw from
the camera images and ego poses in ``rollout.asl``, check that the replayed
plan matches the logged one, and rerun it with the egomotion history's yaw
rate and curvature zeroed. A plan that only turns when those columns are
present is driven by the ego's own motion, not by the cameras.

Run inside the AlpaSim image with a GPU, e.g. on the GPU host:

    docker run --rm --gpus all \
      -v ~/closed-loop-runs/<run>:/rundir:ro \
      -v ~/alpasim/data/drivers/autoe2e:/drivers:ro \
      -v <repo>:/src:ro \
      -e AUTOE2E_MODEL_ROOT=/drivers/Model \
      -e PYTHONPATH=/src/Model/plugins/alpasim_driver:/src/Model/plugins/alpasim_driver/tools \
      alpasim-base:autoe2e-<commit> bash -lc "cd /repo && uv run python \
        /src/experiment/closed-loop/replay_driver.py /rundir <scene prefix> 0,5,10,20"
"""

from __future__ import annotations

import argparse
import asyncio
import io
import math
from pathlib import Path

import numpy as np
from alpasim_driver.models.base import CameraFrame, DriveCommand, ModelPrediction, PredictionInput
from alpasim_utils.logs import async_read_pb_log
from PIL import Image
from score_rollout import se3, to_trajectory

import alpasim_autoe2e.parser as parser_module
from alpasim_autoe2e.parser import CONTEXT_FRAMES
from alpasim_autoe2e.plugin import AutoE2EDriver

NUPLAN_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_front_tele_30fov",
    "camera_rear_right_70fov",
]
HISTORY_STEP_US = 500_000  # inference.subsample_factor 5 on 10 Hz frames


async def read_log(path: Path):
    images: dict[str, dict[int, bytes]] = {}
    poses: dict[int, object] = {}
    requests: list[int] = []
    returns: list[list] = []
    async for message in async_read_pb_log(str(path)):
        kind = message.WhichOneof("log_entry")
        if kind == "driver_camera_image":
            image = message.driver_camera_image.camera_image
            images.setdefault(image.logical_id, {})[int(image.frame_end_us)] = bytes(image.image_bytes)
        elif kind == "driver_ego_trajectory":
            for pose in message.driver_ego_trajectory.trajectory.poses:
                poses[int(pose.timestamp_us)] = pose
        elif kind == "driver_request":
            requests.append(int(message.driver_request.time_now_us))
        elif kind == "driver_return":
            returns.append(list(message.driver_return.trajectory.poses))
    return images, [poses[t] for t in sorted(poses)], requests, returns


def prediction_input(images, poses, camera_ids, now_us: int) -> PredictionInput:
    """The frames and poses the driver held when the request at ``now_us`` arrived."""
    cameras = {}
    for camera in camera_ids:
        newest = max(t for t in images[camera] if t <= now_us)
        times = [newest - HISTORY_STEP_US * i for i in range(CONTEXT_FRAMES - 1, -1, -1)]
        cameras[camera] = [
            CameraFrame(t, np.array(Image.open(io.BytesIO(images[camera][t])).convert("RGB")))
            for t in times
        ]
    return PredictionInput(
        camera_images=cameras,
        command=DriveCommand.STRAIGHT,
        speed=0.0,
        acceleration=0.0,
        ego_pose_history=[p for p in poses if int(p.timestamp_us) <= now_us],
        inference_seed=0,
        previous_plan=None,
        route=None,
    )


def heading_deg(prediction: ModelPrediction, seconds: float) -> float:
    rotation = prediction.selected_rotations[int(round(seconds * 10)) - 1]
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def logged_heading_deg(poses, seconds: float) -> float:
    plan = to_trajectory(poses)
    t0 = int(poses[0].timestamp_us)
    relative = np.linalg.inv(se3(plan, t0)) @ se3(plan, t0 + int(seconds * 1e6))
    return math.degrees(math.atan2(relative[1, 0], relative[0, 0]))


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("run_dir", type=Path)
    args.add_argument("scene", help="clipgt ID prefix, e.g. 01d503d4")
    args.add_argument("decisions", help="zero-based decision indices, e.g. 0,5,10,20")
    args.add_argument("--checkpoint", default="/drivers/checkpoint.pt")
    opts = args.parse_args()

    asl = next(opts.run_dir.glob(f"rollouts/clipgt-{opts.scene}*/*/rollout.asl"))
    images, poses, requests, returns = asyncio.run(read_log(asl))
    driver = AutoE2EDriver(opts.checkpoint, NUPLAN_CAMERAS, device="cuda")

    original = parser_module.egomotion_from_poses

    def without_yaw(*a, **kw):
        egomotion = original(*a, **kw).copy()
        egomotion[:, 2:] = 0.0  # yaw rate, curvature
        return egomotion

    for index in (int(i) for i in opts.decisions.split(",")):
        inputs = prediction_input(images, poses, driver.camera_ids, requests[index])
        parser_module.egomotion_from_poses = original
        replay = driver.predict(inputs)
        parser_module.egomotion_from_poses = without_yaw
        ablated = driver.predict(inputs)
        parser_module.egomotion_from_poses = original
        newest = original(
            parser_module.pose_trajectory(inputs.ego_pose_history),
            int(replay.model_t0_us),
            driver.profile.egomotion,
        )[-1]
        print(
            f"decision {index + 1}: heading@3s logged {logged_heading_deg(returns[index], 3.0):+.2f}"
            f" replay {heading_deg(replay, 3.0):+.2f} yaw/curvature=0 {heading_deg(ablated, 3.0):+.2f} deg;"
            f" newest yaw rate {newest[2]:+.4f} rad/s, curvature {newest[3]:+.4f} 1/m"
        )


if __name__ == "__main__":
    main()
