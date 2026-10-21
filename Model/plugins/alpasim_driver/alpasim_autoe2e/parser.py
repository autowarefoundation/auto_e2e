"""Turns AlpaSim ``PredictionInput`` into AutoE2E model inputs.

The tensors follow the packed training contract of the reactive BEVFormer T8
model (``data_parsing/pre_extracted.py`` and ``data_parsing/kit_scenes``):

* ``camera_tiles``: the current frame of every camera, squashed to a square
  tile and ImageNet-normalised.
* ``camera_history_tiles`` / ``history_projections``: the seven previous frames,
  500 ms apart, with projections that map current-ego points into each
  historical image (``reference_frame: current_ego``).
* ``egomotion_history``: 64 rows of ``[speed, acceleration, yaw_rate,
  curvature]`` derived from the ego poses by finite differences, the way
  ``data_parsing/kit_scenes/egomotion.py`` derives them, with the curvature
  rule of the rig profile's dataset.

Everything is anchored at ``t0``, the timestamp of the newest camera frame; the
driver reports the same ``t0`` back to AlpaSim with the prediction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from alpasim_driver.models.base import ModelInputValidationError, PredictionInput
from alpasim_utils.geometry import Pose, Trajectory
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)
from PIL import Image
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
)
from torchvision import transforms
from training.dataset_policy import (
    AUTO_E2E_TIMESTEPS,
    EGOMOTION_SIGNALS,
    adapt_egomotion_history,
    training_policy_for_dataset,
)

from .config import IMAGE_MEAN, IMAGE_STD, CameraCalibration, EgomotionRule, RigProfile

CONTEXT_FRAMES = REACTIVE_BEVFORMER_HISTORY_FRAMES + 1
"""Camera frames per view the model consumes: seven history frames plus t0."""

EGOMOTION_DT_S = 0.1
"""Row spacing of the egomotion history (the 10 Hz KITScenes timeline)."""

EGOMOTION_SPAN_US = int((AUTO_E2E_TIMESTEPS + 1) * EGOMOTION_DT_S * 1_000_000)
"""Pose history needed before t0: 64 rows plus one more for centred differences."""

# Same tolerance as the KITScenes T8 history check in data_parsing/kit_scenes.
_FRAME_TIMESTAMP_TOLERANCE_US = 50_000


@dataclass
class ParsedObservation:
    """Model inputs for one prediction plus the frame they are expressed in."""

    model_inputs: dict[str, Any]
    initial_speed_mps: float
    t0_us: int
    pose_local_to_rig_t0: Pose


def pose_trajectory(ego_pose_history: list[Any]) -> Trajectory:
    """Convert AlpaSim ``PoseAtTime`` messages (local frame) into a trajectory."""
    poses = sorted(ego_pose_history, key=lambda p: p.timestamp_us)
    quaternions = np.array(
        [[p.pose.quat.x, p.pose.quat.y, p.pose.quat.z, p.pose.quat.w] for p in poses],
        dtype=np.float64,
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return Trajectory(
        np.array([p.timestamp_us for p in poses], dtype=np.uint64),
        np.array(
            [[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in poses],
            dtype=np.float32,
        ),
        quaternions.astype(np.float32),
    )


def egomotion_from_poses(
    trajectory: Trajectory, t0_us: int, rule: EgomotionRule = EgomotionRule()
) -> np.ndarray:
    """Return the ``(64, 4)`` egomotion history ending one step before ``t0``.

    Mirrors ``poses_to_arrays`` / ``load_egomotion`` in
    ``data_parsing/kit_scenes/egomotion.py``: speed is the ground-plane speed,
    acceleration its time derivative, yaw rate the derivative of the unwrapped
    yaw and curvature from the dataset's ``rule`` (``yaw_rate / max(speed,
    0.1)`` for KITScenes), all by ``np.gradient`` on a 10 Hz grid. The rows are
    the 64 samples before ``t0``, as in training.
    """
    steps = np.arange(-(AUTO_E2E_TIMESTEPS + 1), 1)
    times_us = t0_us + (steps * EGOMOTION_DT_S * 1_000_000).astype(np.int64)
    positions = np.empty((len(times_us), 2), dtype=np.float64)
    yaws = np.empty(len(times_us), dtype=np.float64)
    for index, time_us in enumerate(times_us):
        pose = trajectory.interpolate_pose(int(time_us))
        positions[index] = np.asarray(pose.vec3, dtype=np.float64)[:2]
        yaws[index] = float(pose.yaw())

    velocity = np.gradient(positions, EGOMOTION_DT_S, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.gradient(speed, EGOMOTION_DT_S)
    yaw_rate = np.gradient(np.unwrap(yaws), EGOMOTION_DT_S)
    curvature = rule.curvature(yaw_rate, speed)
    egomotion = np.stack([speed, acceleration, yaw_rate, curvature], axis=1)
    # Drop the extra leading row and t0 itself: training rows are [t0-64, t0).
    return egomotion[1:-1].astype(np.float32)


def history_projections(
    cameras: tuple[CameraCalibration, ...],
    local_to_rig_t0: np.ndarray,
    local_to_rig_history: list[np.ndarray],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """``(H, V, 3, 4)`` projections of current-ego points into history frames.

    Mirrors ``compute_temporal_camera_projection_matrices`` in
    ``data_parsing/kit_scenes/camera.py``.
    """
    frames = []
    for local_to_rig_past in local_to_rig_history:
        past_from_current = np.linalg.inv(local_to_rig_past) @ local_to_rig_t0
        frames.append(
            np.stack(
                [
                    camera.scaled_intrinsic(target_hw)
                    @ (camera.reference_to_camera() @ past_from_current)[:3, :]
                    for camera in cameras
                ]
            )
        )
    return np.stack(frames).astype(np.float32)


class AlpasimStreamParser:
    """Builds AutoE2E inputs from AlpaSim camera frames and ego poses."""

    def __init__(
        self,
        profile: RigProfile,
        camera_slots: list[str],
        image_size: int,
        front_image_size: int,
    ) -> None:
        self.profile = profile
        self.cameras = profile.ordered_by_slots(camera_slots)
        self.image_size = image_size
        self.front_image_size = front_image_size
        self.policy = training_policy_for_dataset(profile.dataset_name)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)]
        )
        tile_hw = (image_size, image_size)
        self.projection = torch.from_numpy(
            np.stack([camera.projection(tile_hw) for camera in self.cameras])
        ).float()
        self.front_projection = torch.from_numpy(
            self.cameras[0].projection((front_image_size, front_image_size))[None]
        ).float()

    @property
    def camera_ids(self) -> list[str]:
        return [camera.logical_id for camera in self.cameras]

    def _tile(self, image: Any, size: int) -> torch.Tensor:
        """Squash a frame to ``size x size`` and normalise it, as in training."""
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.cpu().numpy())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
        return self.transform(image)

    def _frames_by_row(
        self, camera_images: dict[str, list[Any]]
    ) -> tuple[list[int], list[list[Any]]]:
        """Return row timestamps and ``[row][view]`` images, oldest row first."""
        rows: list[list[Any]] = [[] for _ in range(CONTEXT_FRAMES)]
        row_times: list[int] = []
        for view, camera in enumerate(self.cameras):
            frames = camera_images.get(camera.logical_id)
            if frames is None:
                raise ModelInputValidationError(
                    f"AutoE2E: no frames for camera {camera.logical_id}"
                )
            frames = sorted(frames, key=lambda frame: frame[0])
            if len(frames) != CONTEXT_FRAMES:
                raise ModelInputValidationError(
                    f"AutoE2E needs {CONTEXT_FRAMES} frames per camera "
                    f"(inference.context_length={CONTEXT_FRAMES}), got "
                    f"{len(frames)} for {camera.logical_id}"
                )
            for row, (timestamp_us, image) in enumerate(frames):
                if view == 0:
                    row_times.append(int(timestamp_us))
                elif abs(int(timestamp_us) - row_times[row]) > _FRAME_TIMESTAMP_TOLERANCE_US:
                    raise ModelInputValidationError(
                        f"AutoE2E: {camera.logical_id} frame {row} is "
                        f"{int(timestamp_us) - row_times[row]} us away from the "
                        f"{self.cameras[0].logical_id} frame"
                    )
                rows[row].append(image)
        spacings = np.diff(row_times)
        if np.any(
            np.abs(spacings - REACTIVE_BEVFORMER_FRAME_INTERVAL_US)
            > _FRAME_TIMESTAMP_TOLERANCE_US
        ):
            raise ModelInputValidationError(
                "AutoE2E history frames must be "
                f"{REACTIVE_BEVFORMER_FRAME_INTERVAL_US} us apart, got spacings "
                f"{spacings.tolist()}; check frame_interval_us and "
                "inference.subsample_factor"
            )
        return row_times, rows

    def parse(self, prediction_input: PredictionInput) -> ParsedObservation:
        row_times, rows = self._frames_by_row(prediction_input.camera_images)
        t0_us = row_times[-1]

        if not prediction_input.ego_pose_history:
            raise ModelInputValidationError("AutoE2E: ego pose history is empty")
        trajectory = pose_trajectory(prediction_input.ego_pose_history)
        time_range_us = trajectory.time_range_us
        first_us, last_us = time_range_us.start, time_range_us.stop - 1
        earliest_needed_us = min(row_times[0], t0_us - EGOMOTION_SPAN_US)
        if first_us > earliest_needed_us or last_us < t0_us:
            raise ModelInputValidationError(
                f"AutoE2E needs ego poses from {earliest_needed_us} to {t0_us} us, "
                f"got {first_us} to {last_us} us. Let the ego follow the recording "
                "for at least 7 s (runtime.simulation_config.force_gt_duration_us)."
            )

        pose_t0 = trajectory.interpolate_pose(t0_us)
        local_to_rig_t0 = np.asarray(pose_t0.as_se3(), dtype=np.float64)
        local_to_rig_history = [
            np.asarray(trajectory.interpolate_pose(t).as_se3(), dtype=np.float64)
            for t in row_times[:-1]
        ]
        tile_hw = (self.image_size, self.image_size)

        visual_tiles = torch.stack(
            [self._tile(image, self.image_size) for image in rows[-1]]
        )
        front_camera_tile = None
        if self.profile.native_front_camera_tile:
            # pre_extracted.py keeps the native front tile and derives the base
            # tile from it, so the two views of the front camera stay identical.
            front_camera_tile = self._tile(rows[-1][0], self.front_image_size)
            visual_tiles[0] = F.interpolate(
                front_camera_tile.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        camera_history_tiles = torch.stack(
            [
                torch.stack([self._tile(image, self.image_size) for image in row])
                for row in rows[:-1]
            ]
        )
        camera_history_projection = torch.from_numpy(
            history_projections(
                self.cameras, local_to_rig_t0, local_to_rig_history, tile_hw
            )
        )

        egomotion = torch.from_numpy(egomotion_from_poses(trajectory, t0_us, self.profile.egomotion))
        egomotion_history = adapt_egomotion_history(
            egomotion.reshape(1, AUTO_E2E_TIMESTEPS * EGOMOTION_SIGNALS), self.policy
        )

        height, width = AUTOE2E_NAVIGATION_GEOMETRY.height_px, AUTOE2E_NAVIGATION_GEOMETRY.width_px
        model_inputs: dict[str, Any] = {
            "camera_tiles": visual_tiles.unsqueeze(0),
            "camera_projection_matrix": self.projection.unsqueeze(0),
            "camera_history_tiles": camera_history_tiles.unsqueeze(0),
            "camera_history_projection_matrix": camera_history_projection.unsqueeze(0),
            "egomotion_history": egomotion_history,
            "visual_history": torch.zeros(1, 896),
            # No map or route reaches the driver through AlpaSim's model API, so
            # the model runs on its camera-only track behind explicit validity
            # gates (evaluated as "camera_only_missing_map_route").
            "map_context": torch.zeros(1, MAP_CHANNEL_COUNT, height, width),
            "route_mask": torch.zeros(1, ROUTE_CHANNEL_COUNT, height, width),
            "map_valid": torch.zeros(1, dtype=torch.bool),
            "route_valid": torch.zeros(1, dtype=torch.bool),
        }
        if front_camera_tile is not None:
            model_inputs["front_camera_tile"] = front_camera_tile.unsqueeze(0)
            model_inputs["front_camera_projection_matrix"] = self.front_projection.unsqueeze(0)

        return ParsedObservation(
            model_inputs=model_inputs,
            initial_speed_mps=float(egomotion[-1, 0]),
            t0_us=t0_us,
            pose_local_to_rig_t0=pose_t0,
        )
