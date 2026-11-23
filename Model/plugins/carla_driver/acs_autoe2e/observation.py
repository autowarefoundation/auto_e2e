"""AutoE2E model inputs from an ACS session.

The tensors are those of the AlpaSim plugin (``alpasim_autoe2e.parser``), and
so of the packed training data of the reactive BEVFormer T8 model:

* ``camera_tiles``: the newest frame of every camera, squashed to a square tile
  and ImageNet-normalised;
* ``camera_history_tiles`` / ``camera_history_projection_matrix``: the seven
  frames 500 ms, 1 s, ... 3.5 s before it, with projections of current-ego
  points into each of them;
* ``egomotion_history``: 64 rows of ``[speed, acceleration, yaw_rate,
  curvature]`` on a 10 Hz grid before ``t0``, by finite differences of the ego
  poses, with the dataset's curvature rule;
* ``map_context`` / ``route_mask``: all zeros behind closed validity gates, as
  under AlpaSim: the model runs on its camera-only track.

The difference is where they come from. ACS streams camera frames as they are
rendered and the ego pose of every simulation tick, and the policy keeps them
(``SessionState.frame_history`` and ``ego_trajectory``); the frames 500 ms
apart are picked from that stream here. Everything is anchored at ``t0``, the
time of the newest camera frame.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from alpasim_autoe2e.config import IMAGE_MEAN, IMAGE_STD, EgomotionRule, RigProfile
from carla_driver_interface.driver import SessionState
from carla_driver_interface.geometry import Pose, Trajectory
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)
from PIL import Image
from scipy.spatial.transform import Rotation
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

from .lens import VirtualView

CONTEXT_FRAMES = REACTIVE_BEVFORMER_HISTORY_FRAMES + 1
"""Camera frames per view the model consumes: seven history frames plus t0."""

EGOMOTION_DT_S = 0.1
"""Row spacing of the egomotion history."""

EGOMOTION_SPAN_US = int((AUTO_E2E_TIMESTEPS + 1) * EGOMOTION_DT_S * 1_000_000)
"""Pose history needed before t0: 64 rows plus one more for centred differences."""

HISTORY_SPAN_US = REACTIVE_BEVFORMER_HISTORY_FRAMES * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
"""Camera history needed before t0 (3.5 s)."""

FRAME_TOLERANCE_US = 50_000
"""How far a picked frame may be from its 500 ms grid point (as in training)."""


class HistoryNotReady(RuntimeError):
    """The session does not hold enough frames or poses yet (still warming up)."""


@dataclass
class Observation:
    """Model inputs for one prediction, and the frame they are expressed in."""

    model_inputs: dict[str, Any]
    initial_speed_mps: float
    t0_us: int
    local_from_rig_t0: Pose


def pose_matrix(trajectory: Trajectory, timestamp_us: int) -> np.ndarray:
    """``local <- rig`` at ``timestamp_us``, interpolated from the ego poses."""
    pose = trajectory.interpolate(int(timestamp_us))
    if pose is None:
        raise HistoryNotReady(f"no ego pose at {timestamp_us} us")
    return pose.as_matrix()


def egomotion_from_trajectory(
    trajectory: Trajectory, t0_us: int, rule: EgomotionRule
) -> np.ndarray:
    """The ``(64, 4)`` egomotion history ending one step before ``t0``.

    The same finite differences as ``egomotion_from_poses`` in the AlpaSim
    plugin, which mirrors ``data_parsing/kit_scenes/egomotion.py``.
    """
    steps = np.arange(-(AUTO_E2E_TIMESTEPS + 1), 1)
    times_us = t0_us + (steps * EGOMOTION_DT_S * 1_000_000).astype(np.int64)
    positions = np.empty((len(times_us), 2), dtype=np.float64)
    yaws = np.empty(len(times_us), dtype=np.float64)
    for index, time_us in enumerate(times_us):
        matrix = pose_matrix(trajectory, int(time_us))
        positions[index] = matrix[:2, 3]
        yaws[index] = np.arctan2(matrix[1, 0], matrix[0, 0])
    velocity = np.gradient(positions, EGOMOTION_DT_S, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.gradient(speed, EGOMOTION_DT_S)
    yaw_rate = np.gradient(np.unwrap(yaws), EGOMOTION_DT_S)
    curvature = rule.curvature(yaw_rate, speed)
    egomotion = np.stack([speed, acceleration, yaw_rate, curvature], axis=1)
    return egomotion[1:-1].astype(np.float32)


def pick_history_frames(
    frames: list[Any], t0_us: int
) -> list[Any]:
    """The frames at ``t0 - 3.5 s, ..., t0 - 0.5 s, t0``, oldest first.

    Raises :class:`HistoryNotReady` when one of them has not been rendered yet.
    """
    if not frames:
        raise HistoryNotReady("no frames yet")
    times = np.array([frame.timestamp_us for frame in frames], dtype=np.int64)
    picked = []
    for k in range(REACTIVE_BEVFORMER_HISTORY_FRAMES, -1, -1):
        target = t0_us - k * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        index = int(np.argmin(np.abs(times - target)))
        if abs(int(times[index]) - target) > FRAME_TOLERANCE_US:
            raise HistoryNotReady(
                f"{frames[0].logical_id}: no frame within {FRAME_TOLERANCE_US} us of "
                f"{target} us (have {int(times[0])} to {int(times[-1])} us)"
            )
        picked.append(frames[index])
    return picked


class ObservationBuilder:
    """Builds AutoE2E inputs from the frames and poses an ACS session holds."""

    def __init__(self, profile: RigProfile, camera_slots: list[str], image_size: int,
                 front_image_size: int, views: dict[str, VirtualView] | None = None) -> None:
        self.profile = profile
        self.views = dict(views or {})
        self.cameras = profile.ordered_by_slots(camera_slots)
        self.image_size = image_size
        self.front_image_size = front_image_size
        self.policy = training_policy_for_dataset(profile.dataset_name)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        self.transform = transforms.Compose([self.to_tensor, self.normalize])
        tile_hw = (image_size, image_size)
        self.projection = torch.from_numpy(
            np.stack([camera.projection(tile_hw) for camera in self.cameras])
        ).float()
        self.front_projection = torch.from_numpy(
            self.cameras[0].projection((front_image_size, front_image_size))[None]
        ).float()
        # Tiles already decoded, by (camera, frame time, size): a frame is used by up
        # to eight consecutive predictions, so it is decoded once.
        self._tiles: dict[tuple[str, int, int], torch.Tensor] = {}

    def sources(self, camera_id: str) -> list[str]:
        """The declared cameras whose frames make ``camera_id``'s view."""
        view = self.views.get(camera_id)
        return view.source_ids if view is not None else [camera_id]

    def _image(self, frame: Any) -> Image.Image:
        with Image.open(io.BytesIO(frame.image_bytes)) as image:
            return image.convert("RGB")

    def _tile(self, camera_id: str, row: tuple[Any, ...], size: int) -> torch.Tensor:
        """One view's tile from a row of its sources' frames (one each)."""
        key = (camera_id, int(row[0].timestamp_us), size)
        tile = self._tiles.get(key)
        if tile is None:
            view = self.views.get(camera_id)
            if view is not None:
                images = [self.to_tensor(self._image(frame)) for frame in row]
                tile = self.normalize(view.render(images, (size, size)))
            else:
                rgb = self._image(row[0])
                if rgb.size != (size, size):
                    rgb = rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
                tile = self.transform(rgb)
            self._tiles[key] = tile
        return tile

    def forget_before(self, timestamp_us: int) -> None:
        """Drop cached tiles of frames older than ``timestamp_us``."""
        for key in [k for k in self._tiles if k[1] < timestamp_us]:
            del self._tiles[key]

    def newest_common_time(self, session: SessionState) -> int:
        """``t0``: the newest instant every camera has a frame for."""
        newest = []
        for camera in self.cameras:
            for source in self.sources(camera.logical_id):
                frames = session.frame_history.get(source) or []
                if not frames:
                    raise HistoryNotReady(f"{source}: no frame yet")
                newest.append(frames[-1].timestamp_us)
        return int(min(newest))

    def build(self, session: SessionState) -> Observation:
        """The model inputs at the newest common camera time."""
        t0_us = self.newest_common_time(session)
        with session.lock:
            frames = {
                source: list(session.frame_history[source])
                for camera in self.cameras
                for source in self.sources(camera.logical_id)
            }
            trajectory = Trajectory(
                list(session.ego_trajectory.timestamps_us), list(session.ego_trajectory.poses)
            )
        # rows_by_view[view][row]: the frames of the view's sources at one history time
        rows_by_view = [
            list(zip(*(pick_history_frames(frames[source], t0_us)
                       for source in self.sources(c.logical_id))))
            for c in self.cameras
        ]
        ids = [c.logical_id for c in self.cameras]
        self.forget_before(t0_us - HISTORY_SPAN_US - FRAME_TOLERANCE_US)
        # interpolate() clamps outside the poses' span, so check the span first
        times = trajectory.timestamps_us
        needed_from = t0_us - max(EGOMOTION_SPAN_US, HISTORY_SPAN_US)
        if not times or int(times[0]) > needed_from or int(times[-1]) < t0_us:
            raise HistoryNotReady(
                f"ego poses cover {times[0] if times else None} to "
                f"{times[-1] if times else None} us; need {needed_from} to {t0_us} us"
            )
        row_times = [row[0].timestamp_us for row in rows_by_view[0]]

        local_from_rig_t0 = pose_matrix(trajectory, t0_us)
        local_from_rig_history = [pose_matrix(trajectory, t) for t in row_times[:-1]]
        tile_hw = (self.image_size, self.image_size)

        visual_tiles = torch.stack(
            [self._tile(cid, rows[-1], self.image_size) for cid, rows in zip(ids, rows_by_view)]
        )
        front_camera_tile = None
        if self.profile.native_front_camera_tile:
            front_camera_tile = self._tile(ids[0], rows_by_view[0][-1], self.front_image_size)
            visual_tiles[0] = F.interpolate(
                front_camera_tile.unsqueeze(0), size=tile_hw, mode="bilinear",
                align_corners=False, antialias=True,
            ).squeeze(0)
        camera_history_tiles = torch.stack(
            [
                torch.stack([self._tile(cid, rows[row], self.image_size)
                             for cid, rows in zip(ids, rows_by_view)])
                for row in range(REACTIVE_BEVFORMER_HISTORY_FRAMES)
            ]
        )
        history_projection = []
        for local_from_rig_past in local_from_rig_history:
            past_from_current = np.linalg.inv(local_from_rig_past) @ local_from_rig_t0
            history_projection.append(
                np.stack(
                    [
                        camera.scaled_intrinsic(tile_hw)
                        @ (camera.reference_to_camera() @ past_from_current)[:3, :]
                        for camera in self.cameras
                    ]
                )
            )
        camera_history_projection = torch.from_numpy(np.stack(history_projection).astype(np.float32))

        egomotion = torch.from_numpy(
            egomotion_from_trajectory(trajectory, t0_us, self.profile.egomotion)
        )
        egomotion_history = adapt_egomotion_history(
            egomotion.reshape(1, AUTO_E2E_TIMESTEPS * EGOMOTION_SIGNALS), self.policy
        )
        height = AUTOE2E_NAVIGATION_GEOMETRY.height_px
        width = AUTOE2E_NAVIGATION_GEOMETRY.width_px
        model_inputs: dict[str, Any] = {
            "camera_tiles": visual_tiles.unsqueeze(0),
            "camera_projection_matrix": self.projection.unsqueeze(0),
            "camera_history_tiles": camera_history_tiles.unsqueeze(0),
            "camera_history_projection_matrix": camera_history_projection.unsqueeze(0),
            "egomotion_history": egomotion_history,
            "visual_history": torch.zeros(1, 896),
            # The map and route are not rasterised yet: the model runs on its
            # camera-only track behind explicit validity gates, as under AlpaSim.
            "map_context": torch.zeros(1, MAP_CHANNEL_COUNT, height, width),
            "route_mask": torch.zeros(1, ROUTE_CHANNEL_COUNT, height, width),
            "map_valid": torch.zeros(1, dtype=torch.bool),
            "route_valid": torch.zeros(1, dtype=torch.bool),
        }
        if front_camera_tile is not None:
            model_inputs["front_camera_tile"] = front_camera_tile.unsqueeze(0)
            model_inputs["front_camera_projection_matrix"] = self.front_projection.unsqueeze(0)
        return Observation(
            model_inputs=model_inputs,
            initial_speed_mps=float(egomotion[-1, 0]),
            t0_us=t0_us,
            local_from_rig_t0=pose_from_matrix(local_from_rig_t0),
        )


def pose_from_matrix(matrix: np.ndarray) -> Pose:
    """A :class:`Pose` from a ``(4, 4)`` rigid transform."""
    return Pose.from_rotation(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]))
