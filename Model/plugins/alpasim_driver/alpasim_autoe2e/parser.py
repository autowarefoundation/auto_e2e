"""Turns AlpaSim ``PredictionInput`` into the named inputs of ``AutoE2E.forward``.

The tensors follow the AutoBEV sample contract
(``data_processing/autobev_members.py``):

* ``images`` [1, 6, 3, 768, 1344]: the newest frame of every camera, scaled to
  cover 1344x768 with the aspect ratio kept, centre-cropped and
  ImageNet-normalised (``crop_params``);
* ``history_images`` / ``history_projection``: the seven previous frames,
  500 ms apart, oldest first, with projections that map current-ego points
  into each past crop;
* ``egomotion_history`` [1, 256]: 64 rows of ``[speed, acceleration,
  yaw_rate, curvature]`` derived from the ego poses by finite differences, with
  the curvature rule of the rig profile's dataset. ``AutoE2E`` scales it.

AlpaSim passes no map or route, so the navigation rasters are zero behind
``map_valid = route_valid = False``. The map would be cropped at the
simulator's exact ego pose, so ``map_pose`` is zero.

Everything is anchored at ``t0``, the timestamp of the newest camera frame; the
driver reports the same ``t0`` back to AlpaSim with the prediction.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from alpasim_driver.models.base import ModelInputValidationError, PredictionInput
from alpasim_utils.geometry import Pose, Trajectory
from data_processing.autobev_members import crop_params
from model_components.autobev.geometry import FRAME_INTERVAL_US, HISTORY_FRAMES, IMAGE_H, IMAGE_W
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)
from PIL import Image
from torchvision import transforms

from .config import IMAGE_MEAN, IMAGE_STD, CameraCalibration, EgomotionRule, RigProfile

CONTEXT_FRAMES: Final = HISTORY_FRAMES + 1
"""Camera frames per view the model consumes: seven history frames plus t0."""

EGOMOTION_ROWS: Final = 64
"""Rows of the egomotion history: 6.4 s at 10 Hz."""

EGOMOTION_DT_S: Final = 0.1
"""Row spacing of the egomotion history."""

EGOMOTION_SPAN_US: Final = int((EGOMOTION_ROWS + 1) * EGOMOTION_DT_S * 1_000_000)
"""Pose history needed before t0: 64 rows plus one more for centred differences."""

MAP_POSE_SIGMA: Final = (1.0, 1.0, math.radians(1.0))
"""Standard deviation (x, y in m, yaw in rad) given as the crop pose's covariance.

The simulator pose is exact, so there is no estimate to describe. These are
the nominal accuracies the training EKF assumes (``EKF_GNSS_SIGMA_M`` = 1 m,
``EKF_INITIAL_YAW_SIGMA_DEG`` = 1 deg in ``training.localization_noise``), so
map matching searches the same window as for a well-localized training sample.
"""

# Tolerance between the frames of one row and on the 500 ms spacing.
_FRAME_TIMESTAMP_TOLERANCE_US: Final = 50_000


@dataclass
class ParsedObservation:
    """Model inputs for one prediction plus the frame they are expressed in."""

    model_inputs: dict[str, torch.Tensor]
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

    Speed is the ground-plane speed, acceleration its time derivative, yaw
    rate the derivative of the unwrapped yaw and curvature from the dataset's
    ``rule``, all by ``np.gradient`` on a 10 Hz grid. The rows are the 64
    samples before ``t0``, as in training.
    """
    steps = np.arange(-(EGOMOTION_ROWS + 1), 1)
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
    frame_wh: list[tuple[int, int]],
    local_to_rig_t0: np.ndarray,
    local_to_rig_history: list[np.ndarray],
) -> np.ndarray:
    """``(H, V, 3, 4)`` projections of current-ego points into the past crops."""
    return np.stack([
        np.stack([
            camera.projection(size, np.linalg.inv(local_to_rig_past) @ local_to_rig_t0)
            for camera, size in zip(cameras, frame_wh)
        ])
        for local_to_rig_past in local_to_rig_history
    ]).astype(np.float32)


def _rgb(image: Any) -> Image.Image:
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.cpu().numpy())
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


class AlpasimStreamParser:
    """Builds AutoE2E inputs from AlpaSim camera frames and ego poses."""

    def __init__(self, profile: RigProfile, camera_slots: tuple[str, ...]) -> None:
        self.profile = profile
        self.cameras = profile.ordered_by_slots(list(camera_slots))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGE_MEAN, IMAGE_STD)]
        )

    @property
    def camera_ids(self) -> list[str]:
        return [camera.logical_id for camera in self.cameras]

    def _crop(self, image: Image.Image) -> torch.Tensor:
        """Scale to cover 1344x768, centre-crop and normalise, as the packers do."""
        _, _, scaled_w, scaled_h, left, top = crop_params(*image.size)
        image = image.resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR)
        return self.transform(image.crop((left, top, left + IMAGE_W, top + IMAGE_H)))

    def _frames_by_row(
        self, camera_images: dict[str, list[Any]]
    ) -> tuple[list[int], list[list[Image.Image]]]:
        """Return row timestamps and ``[row][view]`` images, oldest row first."""
        rows: list[list[Image.Image]] = [[] for _ in range(CONTEXT_FRAMES)]
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
                rows[row].append(_rgb(image))
        spacings = np.diff(row_times)
        if np.any(np.abs(spacings - FRAME_INTERVAL_US) > _FRAME_TIMESTAMP_TOLERANCE_US):
            raise ModelInputValidationError(
                f"AutoE2E history frames must be {FRAME_INTERVAL_US} us apart, got "
                f"spacings {spacings.tolist()}; check frame_interval_us and "
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
        frame_wh = [image.size for image in rows[-1]]
        projection = np.stack([
            camera.projection(size, np.eye(4)) for camera, size in zip(self.cameras, frame_wh)
        ]).astype(np.float32)
        history_projection = history_projections(
            self.cameras, frame_wh, local_to_rig_t0, local_to_rig_history
        )
        egomotion = torch.from_numpy(
            egomotion_from_poses(trajectory, t0_us, self.profile.egomotion)
        )

        height, width = AUTOE2E_NAVIGATION_GEOMETRY.height_px, AUTOE2E_NAVIGATION_GEOMETRY.width_px
        model_inputs = {
            "images": torch.stack([self._crop(image) for image in rows[-1]]).unsqueeze(0),
            "history_images": torch.stack([
                torch.stack([self._crop(image) for image in row]) for row in rows[:-1]
            ]).unsqueeze(0),
            "projection": torch.from_numpy(projection).unsqueeze(0),
            "history_projection": torch.from_numpy(history_projection).unsqueeze(0),
            "egomotion_history": egomotion.reshape(1, -1),
            "map_context": torch.zeros(1, MAP_CHANNEL_COUNT, height, width),
            "route_mask": torch.zeros(1, ROUTE_CHANNEL_COUNT, height, width),
            "map_valid": torch.zeros(1, dtype=torch.bool),
            "route_valid": torch.zeros(1, dtype=torch.bool),
            "map_pose": torch.zeros(1, 3),
            "map_pose_covariance": torch.diag(torch.tensor(MAP_POSE_SIGMA) ** 2).unsqueeze(0),
            "map_matching": torch.ones(1, dtype=torch.bool),
        }
        return ParsedObservation(
            model_inputs=model_inputs,
            initial_speed_mps=float(egomotion[-1, 0]),
            t0_us=t0_us,
            pose_local_to_rig_t0=pose_t0,
        )
