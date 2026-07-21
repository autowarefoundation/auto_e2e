"""Raw pre-extraction dataset for KITScenes Multimodal.

The Flyte pipeline partitions KITScenes by scene UUID. This wrapper keeps that
identity stable across partitions and exposes the same contract as the L2D and
NVIDIA parsers: raw camera/map tensors, egomotion targets, deterministic sample
and split IDs, reasoning front clips, and world-model frame windows.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterator, TypedDict

import numpy as np
import torch
from kitscenes.dataset import KITScenesDataset as _KITScenesSDK
from kitscenes.poses import load_ego_poses
from torch.utils.data import Dataset

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from typing_extensions import NotRequired

from data_processing.contract_versions import UID_SCHEMA_VERSION
from data_parsing.l2d.world_model_windows import (
    build_windows,
    stride_for_hz,
    window_offsets,
)

from .camera import (
    CAMERA_NAMES,
    compute_camera_projection_matrices,
    load_camera_frame,
)
from .egomotion import (
    MIN_ROWS,
    _FUTURE_TIMESTEPS,
    _HISTORY_TIMESTEPS,
    load_egomotion,
    pose_yaws,
    poses_to_arrays,
)
from .map import _cached_scene_map, generate_bev_map_tile

logger = logging.getLogger(__name__)

_VISUAL_HISTORY_DIM = 896
_TRAJECTORY_GROUND_Z_M = -2.1
_UID_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


class KitScenesSample(TypedDict):
    visual_tiles: torch.Tensor
    map_tile: torch.Tensor
    egomotion_history: torch.Tensor
    visual_history: torch.Tensor
    trajectory_target: torch.Tensor
    scene_id: str
    frame_idx: int
    pose_current: dict[str, float | int]
    gps_future: np.ndarray
    camera_params: torch.Tensor
    history_frames: NotRequired[torch.Tensor]
    future_frames: NotRequired[torch.Tensor]


def _utm32_to_wgs84(xy: np.ndarray) -> np.ndarray:
    """Convert absolute UTM zone 32N coordinates to [latitude, longitude]."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "EPSG:32632", "EPSG:4326", always_xy=True
    )
    longitude, latitude = transformer.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([latitude, longitude]).astype(np.float64)


def _local_xy_to_absolute_utm(
    scene_path: Path, positions_local: np.ndarray
) -> np.ndarray:
    """Shift scene-local pose XY into absolute UTM zone 32N coordinates."""
    scene_map = _cached_scene_map(scene_path)
    if scene_map is None:
        raise ValueError(
            f"KITScenes scene {scene_path.name!r} has no loadable map origin"
        )
    origin = np.asarray(scene_map.utm_origin, dtype=np.float64)
    if origin.shape != (2,):
        raise ValueError(
            f"KITScenes map origin must have shape (2,), got {origin.shape}"
        )
    return np.asarray(positions_local, dtype=np.float64) + origin


def _heading_cw_from_north(yaw_rad: float) -> float:
    """Convert CCW radians from UTM east to clockwise degrees from north."""
    return float((90.0 - np.degrees(yaw_rad)) % 360.0)


def _contiguous_prefix_length(indices: list[int]) -> int:
    """Return the number of contiguous camera frame IDs starting at zero."""
    expected = 0
    for frame_idx in indices:
        if frame_idx != expected:
            break
        expected += 1
    return expected


class KitScenesDataset(Dataset):
    """Dataset where each item is one valid ``(scene_id, frame_idx)`` pair.

    Camera tensors are resized to ``image_size`` but remain unnormalized uint8.
    Packing JPEG-encodes them, and the pre-extracted loader performs the sole
    tensor normalization used by training.
    """

    def __init__(
        self,
        data_root: Path | str | None = None,
        backbone_name: str | None = None,
        split: str | None = None,
        camera_names: list[str] | None = None,
        scene_ids: list[str] | None = None,
        rasterize_map_at_runtime: bool = True,
        image_size: int = 256,
        include_world_model_windows: bool = False,
        wm_num_frames: int = 4,
        wm_hz: float = 1.0,
        source_hz: float = 10.0,
        reasoning_clip_only: bool = False,
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if backbone_name is not None:
            logger.debug(
                "backbone_name=%s is ignored; pipeline samples are unnormalized",
                backbone_name,
            )

        self.camera_names = list(camera_names or CAMERA_NAMES)
        self.rasterize_map_at_runtime = rasterize_map_at_runtime
        self.image_size = image_size
        self._wm_enabled = include_world_model_windows
        self._wm_num_frames = wm_num_frames
        self._wm_stride = stride_for_hz(source_hz, wm_hz)
        self._source_hz = source_hz
        self._reasoning_clip_only = reasoning_clip_only
        self._front_cam = self.camera_names[0]

        self._sdk = _KITScenesSDK(root=data_root, split=split)
        available = set(self._sdk.scene_ids)
        scenes = list(scene_ids) if scene_ids is not None else self._sdk.scene_ids
        if not scenes:
            raise ValueError(f"No scenes found under: {self._sdk.root}")
        if len(set(scenes)) != len(scenes):
            raise ValueError("scene_ids contains duplicates")
        missing = sorted(set(scenes) - available)
        if missing:
            raise ValueError(
                f"Requested scene IDs are not materialized in split {split!r}: "
                f"{missing}"
            )

        self._scene_egomotion: dict[str, np.ndarray] = {}
        self._scene_positions_local: dict[str, np.ndarray] = {}
        self._scene_latlon: dict[str, np.ndarray] = {}
        self._scene_yaws: dict[str, np.ndarray] = {}
        self._scene_timestamps_ns: dict[str, np.ndarray] = {}
        self._scene_camera_params: dict[str, torch.Tensor] = {}
        self._scene_ids: list[str] = []
        self._samples: list[tuple[str, int]] = []

        for scene_id in scenes:
            samples = self._valid_samples_for_scene(scene_id)
            if samples:
                self._scene_ids.append(scene_id)
                self._samples.extend(samples)

        if not self._samples:
            raise ValueError("No valid samples found across all scenes.")

        logger.info(
            "KitScenesDataset: %d samples from %d scenes",
            len(self._samples),
            len(self._scene_ids),
        )

    def _valid_samples_for_scene(self, scene_id: str) -> list[tuple[str, int]]:
        loader = self._sdk.get_sensor_loader(scene_id)
        present = set(loader.get_camera_names())
        missing = [name for name in self.camera_names if name not in present]
        if missing:
            logger.warning("Scene %s: missing cameras %s. Skipping.", scene_id, missing)
            return []

        poses = tuple(load_ego_poses(loader.scene_path))
        if len(poses) < MIN_ROWS:
            logger.warning(
                "Scene %s has only %d ego poses (need %d). Skipping.",
                scene_id,
                len(poses),
                MIN_ROWS,
            )
            return []

        camera_lengths = [
            _contiguous_prefix_length(loader.get_frame_indices(name))
            for name in self.camera_names
        ]
        usable = min(
            len(poses),
            len(loader.get_reference_timestamps()),
            *camera_lengths,
        )
        min_idx = _HISTORY_TIMESTEPS
        max_idx = usable - _FUTURE_TIMESTEPS - 1
        if max_idx < min_idx:
            logger.warning(
                "Scene %s: usable span %d too short for a sample. Skipping.",
                scene_id,
                usable,
            )
            return []

        egomotion, positions_local = poses_to_arrays(poses[:usable])
        positions_utm = _local_xy_to_absolute_utm(
            loader.scene_path, positions_local
        )
        yaws = pose_yaws(poses[:usable])
        timestamps_ns = np.asarray(
            [pose.timestamp_ns for pose in poses[:usable]], dtype=np.int64
        )

        self._scene_egomotion[scene_id] = egomotion
        self._scene_positions_local[scene_id] = positions_local
        self._scene_latlon[scene_id] = _utm32_to_wgs84(positions_utm)
        self._scene_yaws[scene_id] = yaws
        self._scene_timestamps_ns[scene_id] = timestamps_ns
        self._scene_camera_params[scene_id] = compute_camera_projection_matrices(
            loader,
            camera_names=self.camera_names,
            image_size=self.image_size,
        )

        return [
            (scene_id, frame_idx)
            for frame_idx in range(min_idx, max_idx + 1)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def sample_uid(self, idx: int) -> str:
        scene_id, frame_idx = self._samples[idx]
        uid = (
            f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-"
            f"f{frame_idx:06d}"
        )
        if not _UID_SAFE.fullmatch(uid):
            raise ValueError(f"unsafe KITScenes sample uid: {uid!r}")
        return uid

    def split_group_uid(self, idx: int) -> str:
        scene_id, _ = self._samples[idx]
        return f"kitscenes-{scene_id}"

    def frame_index(self, idx: int) -> int:
        _, frame_idx = self._samples[idx]
        return frame_idx

    def row_identity(self, idx: int) -> tuple[str, int]:
        """Return the scene-qualified current row used by dedup packing."""
        return self._samples[idx]

    def window_frame_ids(self, idx: int) -> dict[str, list[list[str]]]:
        scene_id, frame_idx = self._samples[idx]
        history_offsets, future_offsets = window_offsets(
            self._wm_num_frames, self._wm_stride
        )
        scene_len = len(self._scene_egomotion[scene_id])

        def identifiers(offsets: list[int]) -> list[list[str]]:
            output: list[list[str]] = []
            for offset in offsets:
                row = frame_idx + offset
                if row < 0 or row >= scene_len:
                    raise IndexError(
                        f"WM row {row} leaves KITScenes scene {scene_id} "
                        f"[0,{scene_len})"
                    )
                output.append([
                    f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-"
                    f"r{row:06d}-c{view}"
                    for view in range(len(self.camera_names))
                ])
            return output

        return {
            "history": identifiers(history_offsets),
            "future": identifiers(future_offsets),
        }

    def window_rows(self, idx: int) -> list[tuple[str, int]]:
        scene_id, frame_idx = self._samples[idx]
        history_offsets, future_offsets = window_offsets(
            self._wm_num_frames, self._wm_stride
        )
        scene_len = len(self._scene_egomotion[scene_id])
        rows = []
        for offset in history_offsets + future_offsets:
            row = frame_idx + offset
            if row < 0 or row >= scene_len:
                raise IndexError(
                    f"WM row {row} leaves KITScenes scene {scene_id} "
                    f"[0,{scene_len})"
                )
            rows.append((scene_id, row))
        return rows

    def egomotion_for(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ego_history, trajectory, _, _ = self.numeric_for(idx)
        return ego_history, trajectory

    def geospatial_for(
        self, idx: int
    ) -> tuple[dict[str, float | int], np.ndarray]:
        _, _, pose_current, gps_future = self.numeric_for(idx)
        return pose_current, gps_future

    def episode_indices(self) -> list[str]:
        """Return loaded scene ids in stable source order."""
        return list(self._scene_ids)

    def episode_path(self, scene_id: str) -> np.ndarray:
        """Return the full scene path as ``[lat, lon, heading, timestamp]``."""
        if scene_id not in self._scene_latlon:
            raise KeyError(f"unknown KITScenes scene {scene_id!r}")
        latlon = self._scene_latlon[scene_id]
        yaws = self._scene_yaws[scene_id]
        timestamps = self._scene_timestamps_ns[scene_id]
        path = np.empty((len(latlon), 4), dtype=np.float64)
        path[:, :2] = latlon
        path[:, 2] = (90.0 - np.degrees(yaws)) % 360.0
        path[:, 3] = timestamps
        return path

    def sample_pose_records(self) -> Iterator[dict[str, Any]]:
        """Yield exact pose rows for every packed sample in this partition."""
        for idx, (scene_id, frame_idx) in enumerate(self._samples):
            latlon = self._scene_latlon[scene_id]
            yield {
                "sample_uid": self.sample_uid(idx),
                "episode_id": scene_id,
                "frame_index": frame_idx,
                "latitude_deg": float(latlon[frame_idx, 0]),
                "longitude_deg": float(latlon[frame_idx, 1]),
                "heading_deg_cw_from_north": _heading_cw_from_north(
                    self._scene_yaws[scene_id][frame_idx]
                ),
                "timestamp_ns": int(
                    self._scene_timestamps_ns[scene_id][frame_idx]
                ),
                "gps_accuracy_m": None,
            }

    def numeric_for(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, float | int],
        np.ndarray,
    ]:
        scene_id, frame_idx = self._samples[idx]
        ego_history, trajectory = load_egomotion(
            self._scene_egomotion[scene_id], frame_idx=frame_idx
        )
        latlon = self._scene_latlon[scene_id]
        gps_future = np.asarray(
            latlon[frame_idx:frame_idx + _FUTURE_TIMESTEPS + 1],
            dtype=np.float64,
        )
        expected = (_FUTURE_TIMESTEPS + 1, 2)
        if gps_future.shape != expected:
            raise IndexError(
                f"KITScenes GPS window must be {expected}, got {gps_future.shape}"
            )
        pose_current: dict[str, float | int] = {
            "latitude_deg": float(latlon[frame_idx, 0]),
            "longitude_deg": float(latlon[frame_idx, 1]),
            "heading_deg_cw_from_north": _heading_cw_from_north(
                self._scene_yaws[scene_id][frame_idx]
            ),
            "timestamp_ns": int(
                self._scene_timestamps_ns[scene_id][frame_idx]
            ),
            "gps_accuracy_m": float("nan"),
        }
        return ego_history, trajectory, pose_current, gps_future

    def projection_spec(self, image_size: int = 256) -> dict:
        """Return a pinhole projection spec in packed-image coordinates."""
        if image_size == self.image_size:
            matrices = [
                self._scene_camera_params[scene_id]
                for scene_id in self._scene_ids
            ]
        else:
            matrices = [
                compute_camera_projection_matrices(
                    self._sdk.get_sensor_loader(scene_id),
                    camera_names=self.camera_names,
                    image_size=image_size,
                )
                for scene_id in self._scene_ids
            ]
        reference = matrices[0]
        if any(not torch.allclose(reference, matrix) for matrix in matrices[1:]):
            raise ValueError(
                "KITScenes partition contains scenes with different calibration; "
                "pack one scene per partition"
            )
        return {
            "type": "pinhole",
            "matrix": reference.tolist(),
            "reference_frame": "top_lidar_flu",
            "ground_z_m": _TRAJECTORY_GROUND_Z_M,
        }

    def _load_multiview_frame(
        self, scene_id: str, frame_idx: int
    ) -> torch.Tensor:
        return load_camera_frame(
            self._sdk.get_sensor_loader(scene_id),
            frame_idx,
            camera_names=self.camera_names,
            image_size=self.image_size,
        )

    def map_for_row(self, scene_id: str, frame_idx: int) -> torch.Tensor:
        """Rasterize one raw uint8 map tile without loading camera images."""
        if not self.rasterize_map_at_runtime:
            return torch.zeros(
                (3, self.image_size, self.image_size), dtype=torch.uint8
            )
        position = self._scene_positions_local[scene_id][frame_idx]
        bev_map = generate_bev_map_tile(
            scene_path=self._sdk.get_sensor_loader(scene_id).scene_path,
            ego_x=float(position[0]),
            ego_y=float(position[1]),
            ego_yaw=float(self._scene_yaws[scene_id][frame_idx]),
            canvas_size=self.image_size,
        )
        if bev_map is None:
            return torch.zeros(
                (3, self.image_size, self.image_size), dtype=torch.uint8
            )
        return torch.from_numpy(bev_map.copy()).permute(2, 0, 1)

    def get_front_clip(self, idx: int) -> list[torch.Tensor]:
        """Return front frames at the fixed 0/1/2/3/4 second horizons."""
        if not self._reasoning_clip_only:
            raise RuntimeError(
                "get_front_clip requires "
                "KitScenesDataset(reasoning_clip_only=True)"
            )
        from data_processing.reasoning_label_generation.schema import (
            HORIZON_SECONDS,
        )

        scene_id, frame_idx = self._samples[idx]
        loader = self._sdk.get_sensor_loader(scene_id)
        rows = [
            frame_idx + round(seconds * self._source_hz)
            for seconds in HORIZON_SECONDS
        ]
        return [
            load_camera_frame(
                loader,
                row,
                camera_names=[self._front_cam],
                image_size=self.image_size,
            )[0]
            for row in rows
        ]

    def __getitem__(self, idx: int) -> KitScenesSample:
        scene_id, frame_idx = self._samples[idx]
        visual_tiles = self._load_multiview_frame(scene_id, frame_idx)
        map_tile = self.map_for_row(scene_id, frame_idx)

        (
            egomotion_history,
            trajectory_target,
            pose_current,
            gps_future,
        ) = self.numeric_for(idx)
        sample = KitScenesSample(
            visual_tiles=visual_tiles,
            map_tile=map_tile,
            egomotion_history=egomotion_history,
            visual_history=torch.zeros(
                _VISUAL_HISTORY_DIM, dtype=torch.float32
            ),
            trajectory_target=trajectory_target,
            scene_id=scene_id,
            frame_idx=frame_idx,
            pose_current=pose_current,
            gps_future=gps_future,
            camera_params=self._scene_camera_params[scene_id],
        )

        if self._wm_enabled:
            scene_len = len(self._scene_egomotion[scene_id])
            history_frames, future_frames = build_windows(
                lambda row: self._load_multiview_frame(scene_id, row),
                frame_idx,
                0,
                scene_len,
                num_frames=self._wm_num_frames,
                stride=self._wm_stride,
            )
            sample["history_frames"] = history_frames
            sample["future_frames"] = future_frames
        return sample
