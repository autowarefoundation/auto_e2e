"""Raw nuPlan scenario packing for Reactive multi-task training."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import multiprocessing
import os
import pickle
import shutil
import sqlite3
import tarfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from data_parsing.camera_slots import CANONICAL_SIX_CAMERA_SLOTS
from data_processing.contract_versions import contract_versions
from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_TAXONOMY_VERSION,
)
from navigation.contracts import canonical_json_bytes
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
    REACTIVE_BEVFORMER_FRAME_OFFSETS,
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_INDEX,
)

from .targets import (
    NuPlanReactiveTargets,
    build_nuplan_reactive_targets,
    nuplan_reactive_target_members,
)

NUPLAN_CAMERA_CHANNELS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L2",
    "CAM_B0",
    "CAM_R2",
)
NUPLAN_CAMERA_SLOTS = CANONICAL_SIX_CAMERA_SLOTS
NUPLAN_CAMERA_SYNC_TOLERANCE_US = 50_000
NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US = (
    REACTIVE_BEVFORMER_FRAME_INTERVAL_US // 2 - 1
)
NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US = (
    2 * NUPLAN_CAMERA_SYNC_TOLERANCE_US
)
NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M = (-4.0, -2.0, 0.0, 2.0)
NUPLAN_RECTIFICATION_POLICY_VERSION = "nuplan_rectified_pinhole_v1"
NUPLAN_PACK_MANIFEST_VERSION = "nuplan_reactive_manifest_v11"
_NUPLAN_MANIFEST_INVARIANT_KEYS = (
    "bev_taxonomy_version",
    "camera_order",
    "camera_slots",
    "camera_visibility_heights_m",
    "contracts",
    "dataset",
    "dataset_version",
    "geometry_type",
    "has_bev_segmentation",
    "has_reactive_navigation",
    "has_route_reconstruction",
    "has_trajectory_xy",
    "front_camera_fpn_image_size",
    "front_camera_image_size",
    "front_camera_index",
    "history_fallback_max_offset_us",
    "history_camera_spread_max_us",
    "temporal_frame_interval_us",
    "temporal_frame_offsets",
    "image_size",
    "map_context_channels",
    "map_version",
    "navigation_geometry",
    "num_views",
    "projection_scope",
    "route_channels",
    "schema_version",
    "source_revision",
    "split_policy",
)


@dataclasses.dataclass(frozen=True)
class NuPlanCameraBundle:
    """Rectified camera pixels and reference-pose projection matrices."""

    jpeg_by_channel: Mapping[str, bytes]
    projection_matrices: np.ndarray
    front_projection_matrix: np.ndarray
    history_jpeg_by_frame: tuple[Mapping[str, bytes], ...]
    history_projection_matrices: np.ndarray
    camera_visibility: np.ndarray
    metadata: Mapping[str, object]
    front_camera_fpn_jpeg: bytes = b""


@dataclasses.dataclass(frozen=True)
class _NuPlanPackPartition:
    index: int
    db_files: tuple[str, ...]
    scenario_estimate: int
    positive_db_file_count: int = 0
    scenario_limit: int = 0


@dataclasses.dataclass(frozen=True)
class _NuPlanPackWorkerConfig:
    partition: _NuPlanPackPartition
    data_root: str
    map_root: str
    sensor_root: str
    output_directory: str
    source_revision: str
    map_version: str
    image_size: int
    samples_per_shard: int


class _NuPlanNoScenariosError(ValueError):
    """Signal that filtering removed every scenario in one DB partition."""

    def __init__(
        self,
        message: str,
        *,
        prefiltered_samples: Sequence[Mapping[str, str]] = (),
    ) -> None:
        super().__init__(message)
        self.prefiltered_samples = tuple(
            dict(sample) for sample in prefiltered_samples
        )


def _pin_nuplan_pack_thread_environment() -> None:
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["NUM_NODES"] = "1"


def _initialize_nuplan_pack_worker() -> None:
    _pin_nuplan_pack_thread_environment()
    try:
        import cv2
    except ModuleNotFoundError:
        return
    cv2.setNumThreads(1)


def _quaternion_transform(
    translation_xyz: Any,
    quaternion_wxyz: Any,
) -> np.ndarray:
    translation = np.asarray(translation_xyz, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("SE3 translation/quaternion shape is invalid")
    norm = float(np.linalg.norm(quaternion))
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
        or norm <= 1e-12
    ):
        raise ValueError("SE3 translation/quaternion is invalid")
    w, x, y, z = quaternion / norm
    rotation = np.asarray([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def camera_visibility_from_projection_matrices(
    projection_matrices: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> np.ndarray:
    """Return cells whose ground centers project into at least one camera."""
    matrices = np.asarray(projection_matrices, dtype=np.float64)
    if (
        matrices.ndim != 3
        or matrices.shape[1:] != (3, 4)
        or not np.isfinite(matrices).all()
    ):
        raise ValueError("projection_matrices must be finite [V,3,4]")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("camera image dimensions must be positive")
    x_grid, y_grid = geometry.pixel_center_grids()
    cell_count = x_grid.size
    height_count = len(NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M)
    points = np.stack(
        [
            np.tile(x_grid.reshape(-1), height_count),
            np.tile(y_grid.reshape(-1), height_count),
            np.repeat(
                np.asarray(
                    NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M,
                    dtype=np.float64,
                ),
                cell_count,
            ),
            np.ones(cell_count * height_count, dtype=np.float64),
        ],
        axis=0,
    )
    visible = np.zeros(cell_count, dtype=np.bool_)
    for matrix in matrices:
        projected = (matrix @ points).reshape(3, height_count, cell_count)
        depth = projected[2]
        valid_depth = depth > 1e-6
        safe_depth = np.where(valid_depth, depth, 1.0)
        column = projected[0] / safe_depth
        row = projected[1] / safe_depth
        visible |= (
            valid_depth
            & (column >= 0.0)
            & (column < image_width)
            & (row >= 0.0)
            & (row < image_height)
        ).any(axis=0)
    return visible.reshape(geometry.height_px, geometry.width_px)


def lidar_observability_from_points(
    points_ego_xyz: np.ndarray,
    *,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
    angular_bins: int = 1440,
) -> np.ndarray:
    """Approximate current LiDAR ray coverage on the common BEV grid."""
    points = np.asarray(points_ego_xyz, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] < 2
        or angular_bins <= 0
    ):
        raise ValueError("LiDAR points must have shape [N,>=2]")
    finite = np.isfinite(points[:, :2]).all(axis=1)
    points = points[finite]
    if not len(points):
        return np.zeros(
            (geometry.height_px, geometry.width_px),
            dtype=np.bool_,
        )
    point_ranges = np.linalg.norm(points[:, :2], axis=1)
    point_angles = np.arctan2(points[:, 1], points[:, 0])
    bins = np.floor(
        (point_angles + math.pi) / (2.0 * math.pi) * angular_bins
    ).astype(np.int64)
    bins = np.clip(bins, 0, angular_bins - 1)
    maximum_range: np.ndarray = np.zeros(
        angular_bins,
        dtype=np.float64,
    )
    np.maximum.at(maximum_range, bins, point_ranges)
    expanded_range = maximum_range.copy()
    bin_width = 2.0 * math.pi / angular_bins
    for bin_index in np.flatnonzero(maximum_range > 0.0):
        ray_range = maximum_range[bin_index]
        angular_margin = math.ceil(
            math.atan2(geometry.meters_per_pixel, max(
                ray_range,
                geometry.meters_per_pixel,
            ))
            / bin_width
        )
        angular_margin = min(angular_margin, angular_bins // 4)
        for offset in range(-angular_margin, angular_margin + 1):
            target_bin = (int(bin_index) + offset) % angular_bins
            expanded_range[target_bin] = max(
                expanded_range[target_bin],
                ray_range,
            )

    x_grid, y_grid = geometry.pixel_center_grids()
    cell_ranges = np.hypot(x_grid, y_grid)
    cell_angles = np.arctan2(y_grid, x_grid)
    cell_bins = np.floor(
        (cell_angles + math.pi) / (2.0 * math.pi) * angular_bins
    ).astype(np.int64)
    cell_bins = np.clip(cell_bins, 0, angular_bins - 1)
    return (
        expanded_range[cell_bins] > 0.0
    ) & (
        cell_ranges <= expanded_range[cell_bins] + geometry.meters_per_pixel
    )


def _decode_pickle_vector(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    decoded = pickle.loads(value) if isinstance(value, bytes) else value
    array = np.asarray(decoded, dtype=np.float64)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError(f"nuPlan {name} has an invalid shape or value")
    return array


def _camera_rows(
    log_file: str,
    lidar_token: str,
) -> tuple[
    sqlite3.Row,
    dict[str, sqlite3.Row],
    tuple[dict[str, sqlite3.Row], ...],
]:
    connection = sqlite3.connect(log_file)
    connection.row_factory = sqlite3.Row
    try:
        reference = connection.execute(
            """
            SELECT lp.timestamp, ep.x, ep.y, ep.z,
                   ep.qw, ep.qx, ep.qy, ep.qz
            FROM lidar_pc AS lp
            INNER JOIN ego_pose AS ep ON ep.token = lp.ego_pose_token
            WHERE lp.token = ?
            """,
            (bytearray.fromhex(lidar_token),),
        ).fetchone()
        if reference is None:
            raise ValueError("nuPlan lidar reference pose is missing")
        placeholders = ",".join("?" for _ in NUPLAN_CAMERA_CHANNELS)

        def rows_near(
            target_timestamp: int,
            *,
            fallback_max_offset_us: int | None = None,
        ) -> dict[str, sqlite3.Row]:
            rows = connection.execute(
                f"""
                SELECT img.filename_jpg, img.timestamp,
                       cam.channel, cam.model, cam.translation, cam.rotation,
                       cam.intrinsic, cam.distortion, cam.width, cam.height,
                       ep.x, ep.y, ep.z, ep.qw, ep.qx, ep.qy, ep.qz
                FROM image AS img
                INNER JOIN camera AS cam ON cam.token = img.camera_token
                INNER JOIN ego_pose AS ep ON ep.token = img.ego_pose_token
                WHERE cam.channel IN ({placeholders})
                  AND img.timestamp BETWEEN ? AND ?
                ORDER BY ABS(img.timestamp - ?), img.timestamp, cam.channel
                """,
                (
                    *NUPLAN_CAMERA_CHANNELS,
                    target_timestamp - NUPLAN_CAMERA_SYNC_TOLERANCE_US,
                    target_timestamp + NUPLAN_CAMERA_SYNC_TOLERANCE_US,
                    target_timestamp,
                ),
            ).fetchall()
            by_channel: dict[str, sqlite3.Row] = {}
            for row in rows:
                by_channel.setdefault(str(row["channel"]), row)
            missing = set(NUPLAN_CAMERA_CHANNELS) - set(by_channel)
            if missing and fallback_max_offset_us is not None:
                for channel in sorted(missing):
                    nearest = connection.execute(
                        """
                        SELECT img.filename_jpg, img.timestamp,
                               cam.channel, cam.model, cam.translation,
                               cam.rotation, cam.intrinsic, cam.distortion,
                               cam.width, cam.height,
                               ep.x, ep.y, ep.z,
                               ep.qw, ep.qx, ep.qy, ep.qz
                        FROM image AS img
                        INNER JOIN camera AS cam
                            ON cam.token = img.camera_token
                        INNER JOIN ego_pose AS ep
                            ON ep.token = img.ego_pose_token
                        WHERE cam.channel = ?
                          AND img.timestamp BETWEEN ? AND ?
                        ORDER BY ABS(img.timestamp - ?), img.timestamp
                        LIMIT 1
                        """,
                        (
                            channel,
                            target_timestamp - fallback_max_offset_us,
                            target_timestamp + fallback_max_offset_us,
                            target_timestamp,
                        ),
                    ).fetchone()
                    if nearest is not None:
                        by_channel[channel] = nearest
                missing = set(NUPLAN_CAMERA_CHANNELS) - set(by_channel)
            if missing:
                raise ValueError(
                    "nuPlan sample is missing required cameras near "
                    f"{target_timestamp}: {sorted(missing)}"
                )
            if fallback_max_offset_us is not None:
                timestamps = [
                    int(row["timestamp"]) for row in by_channel.values()
                ]
                if (
                    max(timestamps) - min(timestamps)
                    > NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US
                ):
                    raise ValueError(
                        "nuPlan history camera timestamp spread exceeds "
                        f"{NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US} us"
                    )
            return by_channel

        reference_timestamp = int(reference["timestamp"])
        current_rows = rows_near(reference_timestamp)
        history_rows = tuple(
            rows_near(
                reference_timestamp
                + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US,
                fallback_max_offset_us=(
                    NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US
                ),
            )
            for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]
        )
    finally:
        connection.close()
    return reference, current_rows, history_rows


def _rectify_camera_rows(
    rows: Mapping[str, sqlite3.Row],
    *,
    sensor_root: str,
    reference_pose: np.ndarray,
    reference_timestamp: int,
    image_size: int,
    native_front: bool,
) -> tuple[
    dict[str, bytes],
    bytes | None,
    np.ndarray,
    np.ndarray | None,
    list[dict[str, object]],
]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuPlan offline camera rectification requires OpenCV"
        ) from exc

    jpegs: dict[str, bytes] = {}
    front_camera_fpn_jpeg = None
    matrices = []
    front_projection_matrix = None
    camera_metadata = []
    for camera_index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
        row = rows[channel]
        sensor_model = str(row["model"]).strip()
        if not sensor_model:
            raise ValueError(
                f"nuPlan {channel} camera sensor model is missing"
            )
        native_width = int(row["width"])
        native_height = int(row["height"])
        if native_width <= 0 or native_height <= 0:
            raise ValueError("nuPlan camera dimensions are invalid")
        image_path = Path(sensor_root) / str(row["filename_jpg"])
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (native_height, native_width):
            raise ValueError(
                f"nuPlan camera image dimensions differ for {channel}"
            )
        intrinsic = _decode_pickle_vector(
            row["intrinsic"],
            expected_shape=(3, 3),
            name=f"{channel} intrinsic",
        )
        distortion_raw = (
            pickle.loads(row["distortion"])
            if isinstance(row["distortion"], bytes)
            else row["distortion"]
        )
        distortion = np.asarray(
            distortion_raw if distortion_raw is not None else [],
            dtype=np.float64,
        ).reshape(-1)
        if distortion.shape != (5,) or not np.isfinite(distortion).all():
            raise ValueError(f"nuPlan {channel} distortion is invalid")
        rectified_intrinsic, _ = cv2.getOptimalNewCameraMatrix(
            intrinsic,
            distortion,
            (native_width, native_height),
            0.0,
            (native_width, native_height),
        )
        rectified = cv2.undistort(
            rgb,
            intrinsic,
            distortion,
            None,
            rectified_intrinsic,
        )
        packed_image_size = (
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            if native_front
            and camera_index == REACTIVE_FRONT_CAMERA_INDEX
            else image_size
        )
        resized = Image.fromarray(rectified).resize(
            (packed_image_size, packed_image_size),
            resample=Image.Resampling.BILINEAR,
        )
        output = io.BytesIO()
        resized.save(
            output,
            format="JPEG",
            quality=90,
            optimize=False,
            progressive=False,
        )
        jpegs[channel] = output.getvalue()
        if (
            native_front
            and camera_index == REACTIVE_FRONT_CAMERA_INDEX
        ):
            base_output = io.BytesIO()
            Image.fromarray(rectified).resize(
                (image_size, image_size),
                resample=Image.Resampling.BILINEAR,
            ).save(
                base_output,
                format="JPEG",
                quality=90,
                optimize=False,
                progressive=False,
            )
            front_camera_fpn_jpeg = base_output.getvalue()

        base_scaled_intrinsic = rectified_intrinsic.copy()
        base_scaled_intrinsic[0] *= image_size / native_width
        base_scaled_intrinsic[1] *= image_size / native_height
        packed_scaled_intrinsic = rectified_intrinsic.copy()
        packed_scaled_intrinsic[0] *= packed_image_size / native_width
        packed_scaled_intrinsic[1] *= packed_image_size / native_height
        ego_from_camera = _quaternion_transform(
            _decode_pickle_vector(
                row["translation"],
                expected_shape=(3,),
                name=f"{channel} translation",
            ),
            _decode_pickle_vector(
                row["rotation"],
                expected_shape=(4,),
                name=f"{channel} rotation",
            ),
        )
        global_from_image_ego = _quaternion_transform(
            [row["x"], row["y"], row["z"]],
            [row["qw"], row["qx"], row["qy"], row["qz"]],
        )
        camera_from_reference = np.linalg.inv(
            global_from_image_ego @ ego_from_camera
        ) @ reference_pose
        matrices.append(
            base_scaled_intrinsic @ camera_from_reference[:3]
        )
        if (
            native_front
            and camera_index == REACTIVE_FRONT_CAMERA_INDEX
        ):
            front_projection_matrix = (
                packed_scaled_intrinsic @ camera_from_reference[:3]
            )
        camera_metadata.append({
            "base_scaled_rectified_intrinsic": (
                base_scaled_intrinsic.tolist()
            ),
            "channel": channel,
            "distortion": distortion.tolist(),
            "image_time_offset_us": (
                int(row["timestamp"]) - reference_timestamp
            ),
            "native_intrinsic": intrinsic.tolist(),
            "native_size_wh": [native_width, native_height],
            "packed_size_wh": [packed_image_size, packed_image_size],
            "rectified_intrinsic": rectified_intrinsic.tolist(),
            "scaled_rectified_intrinsic": (
                packed_scaled_intrinsic.tolist()
            ),
            "sensor_model": sensor_model,
            "sensor_to_ego": ego_from_camera.tolist(),
        })
    return (
        jpegs,
        front_camera_fpn_jpeg,
        np.stack(matrices).astype(np.float32),
        (
            np.asarray(front_projection_matrix, dtype=np.float32)[None]
            if front_projection_matrix is not None
            else None
        ),
        camera_metadata,
    )


def load_nuplan_camera_bundle(
    scenario: Any,
    *,
    iteration: int = 0,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> NuPlanCameraBundle:
    """Load and pose-compensate the canonical six nuPlan cameras."""
    if image_size != REACTIVE_CAMERA_IMAGE_SIZE:
        raise ValueError(
            "nuPlan base camera image size differs from model contract"
        )
    log_file = getattr(scenario, "_log_file", None)
    sensor_root = getattr(scenario, "_sensor_root", None)
    lidar_tokens = getattr(scenario, "_lidarpc_tokens", None)
    if (
        not isinstance(log_file, str)
        or not isinstance(sensor_root, str)
        or lidar_tokens is None
    ):
        raise ValueError(
            "nuPlan scenario does not expose local DB/sensor roots"
        )
    lidar_token = str(lidar_tokens[iteration])
    reference, rows, history_rows = _camera_rows(log_file, lidar_token)
    reference_pose = _quaternion_transform(
        [reference["x"], reference["y"], reference["z"]],
        [
            reference["qw"],
            reference["qx"],
            reference["qy"],
            reference["qz"],
        ],
    )

    reference_timestamp = int(reference["timestamp"])
    (
        jpegs,
        front_camera_fpn_jpeg,
        projection_matrices,
        front_projection_matrix,
        camera_metadata,
    ) = _rectify_camera_rows(
        rows,
        sensor_root=sensor_root,
        reference_pose=reference_pose,
        reference_timestamp=reference_timestamp,
        image_size=image_size,
        native_front=True,
    )
    if front_projection_matrix is None:
        raise RuntimeError("nuPlan front camera projection was not produced")
    if front_camera_fpn_jpeg is None:
        raise RuntimeError("nuPlan base-resolution front image was not produced")
    history_jpegs = []
    history_matrices = []
    history_timestamps = []
    for frame_rows in history_rows:
        (
            frame_jpegs,
            _frame_front_camera_fpn_jpeg,
            frame_matrices,
            _front_matrix,
            frame_metadata,
        ) = _rectify_camera_rows(
            frame_rows,
            sensor_root=sensor_root,
            reference_pose=reference_pose,
            reference_timestamp=reference_timestamp,
            image_size=image_size,
            native_front=False,
        )
        history_jpegs.append(frame_jpegs)
        history_matrices.append(frame_matrices)
        history_timestamps.append([
            reference_timestamp + cast(
                int,
                camera["image_time_offset_us"],
            )
            for camera in frame_metadata
        ])
    visibility = camera_visibility_from_projection_matrices(
        projection_matrices,
        image_width=image_size,
        image_height=image_size,
        geometry=geometry,
    )
    return NuPlanCameraBundle(
        jpeg_by_channel=jpegs,
        front_camera_fpn_jpeg=front_camera_fpn_jpeg,
        projection_matrices=projection_matrices,
        front_projection_matrix=front_projection_matrix,
        history_jpeg_by_frame=tuple(history_jpegs),
        history_projection_matrices=np.stack(
            history_matrices
        ).astype(np.float32),
        camera_visibility=visibility,
        metadata={
            "camera_order": list(NUPLAN_CAMERA_CHANNELS),
            "camera_slots": list(NUPLAN_CAMERA_SLOTS),
            "cameras": camera_metadata,
            "front_camera_image_size": (
                REACTIVE_FRONT_CAMERA_IMAGE_SIZE
            ),
            "front_camera_fpn_image_size": image_size,
            "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
            "image_size": image_size,
            "rectification_policy": NUPLAN_RECTIFICATION_POLICY_VERSION,
            "reference_lidar_timestamp_us": reference_timestamp,
            "temporal_camera_timestamps_us": history_timestamps,
            "temporal_frame_interval_us": (
                REACTIVE_BEVFORMER_FRAME_INTERVAL_US
            ),
            "temporal_frame_offsets": list(
                REACTIVE_BEVFORMER_FRAME_OFFSETS
            ),
        },
    )


def load_nuplan_lidar_observability(
    scenario: Any,
    *,
    iteration: int = 0,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> np.ndarray:
    """Load the current merged point cloud and rasterize ray coverage."""
    try:
        from nuplan.planning.simulation.observation.observation_type import (
            LidarChannel,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuplan-devkit is required to load merged point clouds"
        ) from exc
    sensors = scenario.get_sensors_at_iteration(
        iteration,
        channels=[LidarChannel.MERGED_PC],
    )
    if (
        sensors.pointcloud is None
        or LidarChannel.MERGED_PC not in sensors.pointcloud
    ):
        raise ValueError("nuPlan merged point cloud is missing")
    point_cloud = sensors.pointcloud[LidarChannel.MERGED_PC]
    points_xyz = _nuplan_point_cloud_xyz(point_cloud.points)
    lidar_from_ego = np.asarray(
        scenario.get_lidar_to_ego_transform(),
        dtype=np.float64,
    )
    if lidar_from_ego.shape != (4, 4):
        raise ValueError("nuPlan lidar-to-ego transform is invalid")
    homogeneous = np.vstack([
        points_xyz.T,
        np.ones(points_xyz.shape[0], dtype=np.float64),
    ])
    points_ego = (lidar_from_ego @ homogeneous)[:3].T
    return lidar_observability_from_points(
        points_ego,
        geometry=geometry,
    )


def _nuplan_point_cloud_xyz(points: Any) -> np.ndarray:
    """Validate the nuPlan channels-first point-cloud contract."""
    array = np.asarray(points, dtype=np.float64)
    if (
        array.ndim != 2
        or not 3 <= array.shape[0] <= 16
        or array.shape[1] <= array.shape[0]
    ):
        raise ValueError(
            "nuPlan merged point cloud must have channels-first shape [C,N]"
        )
    return array[:3].T


def _nuplan_local_sensor_asset_issue(scenario: Any) -> str | None:
    log_file = getattr(scenario, "_log_file", None)
    sensor_root = getattr(scenario, "_sensor_root", None)
    lidar_tokens = getattr(scenario, "_lidarpc_tokens", None)
    if (
        not isinstance(log_file, str)
        or not isinstance(sensor_root, str)
        or lidar_tokens is None
        or not lidar_tokens
    ):
        return "scenario does not expose local sensor paths"
    lidar_token = str(lidar_tokens[0])
    try:
        token_bytes = bytearray.fromhex(lidar_token)
    except ValueError:
        return "scenario lidar token is invalid"
    connection = sqlite3.connect(log_file)
    try:
        row = connection.execute(
            "SELECT filename FROM lidar_pc WHERE token = ?",
            (token_bytes,),
        ).fetchone()
    finally:
        connection.close()
    if row is None or not row[0]:
        return "scenario lidar filename is missing"
    lidar_filename = str(row[0])
    if not (Path(sensor_root) / lidar_filename).is_file():
        return f"local lidar asset is missing: {lidar_filename}"
    try:
        _, current_rows, history_rows = _camera_rows(
            log_file,
            lidar_token,
        )
    except ValueError as error:
        return f"camera metadata is incomplete: {error}"
    for rows in (current_rows, *history_rows):
        for channel in NUPLAN_CAMERA_CHANNELS:
            filename = str(rows[channel]["filename_jpg"])
            if not (Path(sensor_root) / filename).is_file():
                return f"local camera asset is missing: {filename}"
    return None


def _state_signals(states: Sequence[Any], *, dt: float = 0.1) -> np.ndarray:
    if len(states) != 64:
        raise ValueError("nuPlan history/future must contain 64 states")
    speed = np.asarray([
        math.hypot(
            float(state.dynamic_car_state.rear_axle_velocity_2d.x),
            float(state.dynamic_car_state.rear_axle_velocity_2d.y),
        )
        for state in states
    ])
    heading = np.unwrap(np.asarray([
        float(state.rear_axle.heading)
        for state in states
    ]))
    acceleration = np.gradient(speed, dt)
    yaw_rate = np.gradient(heading, dt)
    curvature = np.where(
        speed > 0.5,
        yaw_rate / np.maximum(speed, 0.5),
        0.0,
    )
    curvature = np.clip(curvature, -0.5, 0.5)
    signals = np.stack(
        [speed, acceleration, yaw_rate, curvature],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(signals).all():
        raise ValueError("nuPlan ego-motion signals are non-finite")
    return signals


def _nuplan_ego_member(scenario: Any, *, iteration: int) -> bytes:
    past = list(scenario.get_ego_past_trajectory(
        iteration,
        time_horizon=6.4,
        num_samples=64,
    ))
    future = list(scenario.get_ego_future_trajectory(
        iteration,
        time_horizon=6.4,
        num_samples=64,
    ))
    history_signals = _state_signals(past)
    future_signals = _state_signals(future)
    return np.concatenate([
        history_signals.reshape(-1),
        future_signals[:, [1, 3]].reshape(-1),
    ]).astype(np.float32).tobytes()


def _nuplan_split_group_uid(log_name: str) -> str:
    if not log_name:
        raise ValueError("nuPlan log name must not be empty")
    log_digest = hashlib.sha256(
        log_name.encode("utf-8")
    ).hexdigest()[:20]
    return f"nuplan-log-{log_digest}"


def _sample_identity(scenario: Any) -> tuple[str, str]:
    log_name = str(getattr(scenario, "log_name", ""))
    token = str(getattr(scenario, "token", ""))
    if not log_name or not token:
        raise ValueError("nuPlan scenario lacks log name or token")
    sample_digest = hashlib.sha256(
        f"{log_name}:{token}".encode("utf-8")
    ).hexdigest()[:24]
    return f"nuplan-{sample_digest}", _nuplan_split_group_uid(log_name)


def nuplan_reactive_sample_members(
    scenario: Any,
    *,
    iteration: int = 0,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
    source_revision: str,
    camera_bundle: NuPlanCameraBundle | None = None,
    lidar_observability: np.ndarray | None = None,
    target_builder: Callable[..., NuPlanReactiveTargets] = (
        build_nuplan_reactive_targets
    ),
) -> tuple[str, str, dict[str, bytes]]:
    """Convert one raw nuPlan scenario iteration to packed sample members."""
    if not source_revision:
        raise ValueError("nuPlan source revision must not be empty")
    bundle = camera_bundle or load_nuplan_camera_bundle(
        scenario,
        iteration=iteration,
        image_size=image_size,
    )
    if (
        image_size != REACTIVE_CAMERA_IMAGE_SIZE
        or bundle.metadata.get("image_size")
        != REACTIVE_CAMERA_IMAGE_SIZE
        or bundle.metadata.get("front_camera_index")
        != REACTIVE_FRONT_CAMERA_INDEX
        or bundle.metadata.get("front_camera_image_size")
        != REACTIVE_FRONT_CAMERA_IMAGE_SIZE
        or bundle.metadata.get("front_camera_fpn_image_size")
        != REACTIVE_CAMERA_IMAGE_SIZE
        or bundle.metadata.get("camera_order")
        != list(NUPLAN_CAMERA_CHANNELS)
        or bundle.metadata.get("camera_slots")
        != list(NUPLAN_CAMERA_SLOTS)
    ):
        raise ValueError(
            "nuPlan camera bundle dimensions differ from pack contract"
        )
    history_projection_matrices = np.asarray(
        bundle.history_projection_matrices,
        dtype=np.float32,
    )
    if (
        len(bundle.history_jpeg_by_frame)
        != REACTIVE_BEVFORMER_HISTORY_FRAMES
        or history_projection_matrices.shape != (
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            len(NUPLAN_CAMERA_CHANNELS),
            3,
            4,
        )
        or not np.isfinite(history_projection_matrices).all()
        or bundle.metadata.get("temporal_frame_offsets")
        != list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
        or bundle.metadata.get("temporal_frame_interval_us")
        != REACTIVE_BEVFORMER_FRAME_INTERVAL_US
    ):
        raise ValueError(
            "nuPlan camera bundle temporal history differs from T8 contract"
        )
    expected_front_projection = np.asarray(
        bundle.projection_matrices[REACTIVE_FRONT_CAMERA_INDEX:
                                   REACTIVE_FRONT_CAMERA_INDEX + 1],
        dtype=np.float32,
    ).copy()
    expected_front_projection[:, :2] *= (
        REACTIVE_FRONT_CAMERA_IMAGE_SIZE / REACTIVE_CAMERA_IMAGE_SIZE
    )
    actual_front_projection = np.asarray(
        bundle.front_projection_matrix,
        dtype=np.float32,
    )
    if (
        actual_front_projection.shape != (1, 3, 4)
        or not np.isfinite(actual_front_projection).all()
        or not np.allclose(
            actual_front_projection,
            expected_front_projection,
            rtol=1e-5,
            atol=1e-5,
        )
    ):
        raise ValueError(
            "nuPlan front projection does not match the base camera frame"
        )
    lidar_mask = (
        np.asarray(lidar_observability, dtype=np.bool_)
        if lidar_observability is not None
        else load_nuplan_lidar_observability(
            scenario,
            iteration=iteration,
        )
    )
    expected_shape = (
        AUTOE2E_NAVIGATION_GEOMETRY.height_px,
        AUTOE2E_NAVIGATION_GEOMETRY.width_px,
    )
    if (
        bundle.camera_visibility.shape != expected_shape
        or lidar_mask.shape != expected_shape
    ):
        raise ValueError("nuPlan observability mask geometry mismatch")
    targets = target_builder(
        scenario,
        iteration=iteration,
        camera_visibility=bundle.camera_visibility,
        lidar_observability=lidar_mask,
    )
    sample_uid, split_group_uid = _sample_identity(scenario)
    members = nuplan_reactive_target_members(
        targets,
        metadata={
            "log_name": str(scenario.log_name),
            "map_version": str(getattr(scenario, "map_version", "")),
            "scenario_token": str(scenario.token),
            "source_revision": source_revision,
        },
    )
    for index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
        try:
            members[f"cam_{index}.jpg"] = bundle.jpeg_by_channel[channel]
        except KeyError as exc:
            raise ValueError(
                f"nuPlan camera bundle lacks {channel}"
            ) from exc
    if not isinstance(bundle.front_camera_fpn_jpeg, bytes) or not (
        bundle.front_camera_fpn_jpeg
    ):
        raise ValueError(
            "nuPlan camera bundle lacks the base-resolution front image"
        )
    members["front_camera_fpn.jpg"] = bundle.front_camera_fpn_jpeg
    for history_index, history_jpegs in enumerate(
        bundle.history_jpeg_by_frame
    ):
        for camera_index, channel in enumerate(NUPLAN_CAMERA_CHANNELS):
            try:
                members[
                    f"bev_hist_{history_index}_cam_{camera_index}.jpg"
                ] = history_jpegs[channel]
            except KeyError as exc:
                raise ValueError(
                    "nuPlan camera history lacks "
                    f"frame {history_index} channel {channel}"
                ) from exc
    members["ego.npy"] = _nuplan_ego_member(
        scenario,
        iteration=iteration,
    )
    members["calib.json"] = canonical_json_bytes({
        "dataset": "nuplan/nuplan-v1.1",
        "geometry_type": "rectified_pinhole",
        "projection": {
            "matrix": bundle.projection_matrices.tolist(),
            "type": "rectified_pinhole",
        },
        "front_projection": {
            "matrix": bundle.front_projection_matrix.tolist(),
            "type": "rectified_pinhole",
        },
        "history_projection": {
            "matrix": history_projection_matrices.tolist(),
            "reference_frame": "current_ego",
            "type": "rectified_pinhole",
        },
        **dict(bundle.metadata),
    })
    members["meta.json"] = canonical_json_bytes({
        "dataset": "nuplan/nuplan-v1.1",
        "frame_idx": iteration,
        "log_name": str(scenario.log_name),
        "sample_uid": sample_uid,
        "scenario_token": str(scenario.token),
        "source_revision": source_revision,
        "split_group_uid": split_group_uid,
    })
    return sample_uid, split_group_uid, members


def _add_tar_member(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def _sample_camera_timing_metrics(
    members: Mapping[str, bytes],
) -> tuple[int, int, int]:
    payload = members.get("calib.json")
    if payload is None:
        raise ValueError("nuPlan sample is missing calib.json")
    try:
        calibration = json.loads(payload)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("nuPlan sample calibration is invalid") from exc
    cameras = calibration.get("cameras")
    if (
        not isinstance(cameras, list)
        or len(cameras) != len(NUPLAN_CAMERA_CHANNELS)
    ):
        raise ValueError("nuPlan sample camera timing metadata is invalid")
    offsets = [
        camera.get("image_time_offset_us")
        if isinstance(camera, Mapping)
        else None
        for camera in cameras
    ]
    if any(
        not isinstance(offset, int) or isinstance(offset, bool)
        for offset in offsets
    ):
        raise ValueError("nuPlan sample camera timing metadata is invalid")
    max_offset = max(abs(cast(int, offset)) for offset in offsets)
    if max_offset > NUPLAN_CAMERA_SYNC_TOLERANCE_US:
        raise ValueError(
            "nuPlan current camera time offset exceeds "
            f"{NUPLAN_CAMERA_SYNC_TOLERANCE_US} us"
        )
    reference_timestamp = calibration.get("reference_lidar_timestamp_us")
    history_timestamps = calibration.get(
        "temporal_camera_timestamps_us"
    )
    if (
        not isinstance(reference_timestamp, int)
        or isinstance(reference_timestamp, bool)
        or calibration.get("temporal_frame_offsets")
        != list(REACTIVE_BEVFORMER_FRAME_OFFSETS)
        or calibration.get("temporal_frame_interval_us")
        != REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        or not isinstance(history_timestamps, list)
        or len(history_timestamps) != REACTIVE_BEVFORMER_HISTORY_FRAMES
        or any(
            not isinstance(frame, list)
            or len(frame) != len(NUPLAN_CAMERA_CHANNELS)
            or any(
                not isinstance(timestamp, int)
                or isinstance(timestamp, bool)
                for timestamp in frame
            )
            for frame in history_timestamps
        )
    ):
        raise ValueError("nuPlan sample history timing metadata is invalid")
    typed_history = cast(list[list[int]], history_timestamps)
    expected_timestamps = [
        reference_timestamp
        + offset * REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        for offset in REACTIVE_BEVFORMER_FRAME_OFFSETS[:-1]
    ]
    history_offsets = [
        abs(timestamp - expected_timestamps[frame_index])
        for frame_index, frame in enumerate(typed_history)
        for timestamp in frame
    ]
    if any(
        any(
            typed_history[index][camera_index]
            >= typed_history[index + 1][camera_index]
            for index in range(REACTIVE_BEVFORMER_HISTORY_FRAMES - 1)
        )
        for camera_index in range(len(NUPLAN_CAMERA_CHANNELS))
    ):
        raise ValueError(
            "nuPlan history camera timestamps are not strictly increasing"
        )
    max_history_offset = max(history_offsets)
    if max_history_offset > NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US:
        raise ValueError(
            "nuPlan history camera time offset exceeds "
            f"{NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US} us"
        )
    if any(
        max(frame) - min(frame) > NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US
        for frame in typed_history
    ):
        raise ValueError(
            "nuPlan history camera timestamp spread exceeds "
            f"{NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US} us"
        )
    distinct_history_frame_count = len({
        tuple(frame) for frame in typed_history
    })
    if distinct_history_frame_count != REACTIVE_BEVFORMER_HISTORY_FRAMES:
        raise ValueError("nuPlan history camera frames are not distinct")
    return max_offset, max_history_offset, distinct_history_frame_count


def _nuplan_db_scenario_count(db_path: str | Path) -> int:
    """Count scenarios returned by the configured nuPlan builder query."""
    resolved = Path(db_path).resolve()
    connection = sqlite3.connect(
        f"{resolved.as_uri()}?mode=ro",
        uri=True,
    )
    try:
        row = connection.execute(
            """
            WITH ordered_scenes AS
            (
                SELECT token,
                       ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
                FROM scene
            ),
            num_scenes AS
            (
                SELECT COUNT(*) AS cnt
                FROM scene
            ),
            valid_scenes AS
            (
                SELECT ordered.token
                FROM ordered_scenes AS ordered
                CROSS JOIN num_scenes AS total
                WHERE ordered.row_num >= 3
                  AND ordered.row_num < total.cnt - 1
            )
            SELECT COUNT(*)
            FROM
            (
                SELECT lidar_pc.token
                FROM lidar_pc
                LEFT OUTER JOIN scenario_tag
                    ON lidar_pc.token = scenario_tag.lidar_pc_token
                INNER JOIN lidar
                    ON lidar.token = lidar_pc.lidar_token
                INNER JOIN log
                    ON lidar.log_token = log.token
                INNER JOIN valid_scenes
                    ON lidar_pc.scene_token = valid_scenes.token
                INNER JOIN image
                    ON image.ego_pose_token = lidar_pc.ego_pose_token
                INNER JOIN scene AS goal_scene
                    ON goal_scene.token = lidar_pc.scene_token
                INNER JOIN ego_pose AS goal_ego_pose
                    ON goal_scene.goal_ego_pose_token = goal_ego_pose.token
                GROUP BY lidar_pc.token,
                         lidar_pc.timestamp,
                         log.map_version
            )
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"nuPlan DB has no candidate count: {resolved}")
    count = int(row[0])
    if count < 0:
        raise ValueError(f"nuPlan DB candidate count is invalid: {resolved}")
    return count


def _partition_weighted_nuplan_db_files(
    weighted_db_files: Sequence[tuple[str | Path, int]],
    worker_count: int,
    scenario_limit: int = 0,
) -> list[_NuPlanPackPartition]:
    if worker_count <= 0:
        raise ValueError("nuPlan pack worker_count must be positive")
    if scenario_limit < 0:
        raise ValueError("nuPlan partition scenario_limit must be non-negative")
    normalized = [
        (str(Path(path).resolve()), int(weight))
        for path, weight in weighted_db_files
    ]
    if not normalized:
        raise ValueError("nuPlan DB partition input must not be empty")
    paths = [path for path, _ in normalized]
    if len(set(paths)) != len(paths):
        raise ValueError("nuPlan DB partition input contains duplicates")
    if any(weight < 0 for _, weight in normalized):
        raise ValueError("nuPlan DB scenario weights must be non-negative")
    total_weight = sum(weight for _, weight in normalized)
    if scenario_limit and scenario_limit > total_weight:
        raise ValueError(
            "nuPlan scenario limit exceeds available candidates: "
            f"limit={scenario_limit} candidates={total_weight}"
        )
    positive_count = sum(weight > 0 for _, weight in normalized)
    if positive_count == 0:
        raise _NuPlanNoScenariosError(
            "nuPlan DB files contain no tagged scenarios"
        )

    partition_count = min(
        worker_count,
        positive_count,
        scenario_limit or positive_count,
    )
    assignments: list[list[tuple[str, int]]] = [
        [] for _ in range(partition_count)
    ]
    loads = [0 for _ in range(partition_count)]
    for path, weight in sorted(
        normalized,
        key=lambda item: (-item[1], item[0]),
    ):
        target = min(
            range(partition_count),
            key=lambda index: (loads[index], index),
        )
        assignments[target].append((path, weight))
        loads[target] += weight
    partitions = [
        _NuPlanPackPartition(
            index=index,
            db_files=tuple(sorted(path for path, _ in entries)),
            scenario_estimate=loads[index],
            positive_db_file_count=sum(
                weight > 0 for _, weight in entries
            ),
        )
        for index, entries in enumerate(assignments)
        if entries
    ]
    if not scenario_limit:
        return partitions

    minimums = (
        [partition.positive_db_file_count for partition in partitions]
        if scenario_limit >= positive_count
        else [1 for _ in partitions]
    )
    quotas = list(minimums)
    remaining = scenario_limit - sum(minimums)
    remaining_capacities = [
        partition.scenario_estimate - minimum
        for partition, minimum in zip(partitions, minimums)
    ]
    total_remaining_capacity = sum(remaining_capacities)
    weighted_additions = [
        (
            remaining * capacity / total_remaining_capacity
            if total_remaining_capacity
            else 0.0
        )
        for capacity in remaining_capacities
    ]
    for index, addition in enumerate(weighted_additions):
        quotas[index] += int(addition)
    unassigned = scenario_limit - sum(quotas)
    if unassigned < 0:
        raise RuntimeError("nuPlan scenario quota allocation is negative")
    remainder_order = sorted(
        range(len(partitions)),
        key=lambda index: (
            -(
                weighted_additions[index]
                - int(weighted_additions[index])
            ),
            partitions[index].index,
        ),
    )
    eligible_remainders = [
        index
        for index in remainder_order
        if quotas[index] < partitions[index].scenario_estimate
    ]
    for index in eligible_remainders[:unassigned]:
        quotas[index] += 1
    if sum(quotas) != scenario_limit:
        raise RuntimeError("nuPlan scenario quota allocation is incomplete")
    if any(
        quota > partition.scenario_estimate
        for partition, quota in zip(partitions, quotas)
    ):
        raise RuntimeError("nuPlan scenario quota exceeds candidates")
    return [
        dataclasses.replace(partition, scenario_limit=quota)
        for partition, quota in zip(partitions, quotas)
    ]


def _pack_nuplan_partition(
    config: _NuPlanPackWorkerConfig,
) -> dict[str, object] | None:
    try:
        return pack_nuplan_local_dataset(
            data_root=config.data_root,
            map_root=config.map_root,
            sensor_root=config.sensor_root,
            db_files=config.partition.db_files,
            output_directory=config.output_directory,
            source_revision=config.source_revision,
            map_version=config.map_version,
            limit_total_scenarios=config.partition.scenario_limit,
            image_size=config.image_size,
            samples_per_shard=config.samples_per_shard,
            max_rejection_fraction=1.0,
            pack_workers=1,
            require_accepted=False,
        )
    except _NuPlanNoScenariosError as error:
        return {
            "empty_partition": True,
            "empty_reason": str(error),
            "prefiltered_count": len(error.prefiltered_samples),
            "prefiltered_samples": list(error.prefiltered_samples),
            "rejected_count": 0,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_nuplan_manifest_atomic(
    output: Path,
    manifest: Mapping[str, object],
) -> None:
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_bytes(canonical_json_bytes(manifest))
    temporary_manifest.replace(output / "manifest.json")


def _merge_nuplan_pack_partitions(
    *,
    output: Path,
    partition_root: Path,
    partitions: Sequence[_NuPlanPackPartition],
    manifests: Sequence[Mapping[str, object] | None],
    max_rejection_fraction: float,
    expected_split_group_uids: set[str] | None = None,
    require_full_log_coverage: bool = False,
) -> dict[str, object]:
    if len(partitions) != len(manifests) or not manifests:
        raise ValueError("nuPlan partition results are incomplete")
    def is_empty_manifest(
        manifest: Mapping[str, object] | None,
    ) -> bool:
        return manifest is None or manifest.get("empty_partition") is True

    nonempty_manifests: list[Mapping[str, object]] = [
        cast(Mapping[str, object], manifest)
        for manifest in manifests
        if not is_empty_manifest(manifest)
    ]
    if not nonempty_manifests:
        raise ValueError("nuPlan parallel packing produced no scenarios")
    reference_manifest = nonempty_manifests[0]
    for candidate_manifest in nonempty_manifests:
        for key in _NUPLAN_MANIFEST_INVARIANT_KEYS:
            if (
                key not in candidate_manifest
                or candidate_manifest[key] != reference_manifest.get(key)
            ):
                raise ValueError(
                    "nuPlan worker manifest invariant mismatch: "
                    f"{key}"
                )

    merged_shard_names: list[str] = []
    merged_shard_counts: dict[str, int] = {}
    merged_shard_hashes: dict[str, str] = {}
    merged_sample_uids: set[str] = set()
    merged_prefiltered: list[object] = []
    merged_rejections: list[object] = []
    merged_split_group_uids: set[str] = set()
    merged_max_camera_time_offset_us = 0
    merged_max_history_camera_time_offset_us = 0
    merged_distinct_history_frame_count = 0
    partition_statistics: list[dict[str, object]] = []
    staged_shards: list[tuple[Path, str]] = []

    for partition, manifest in zip(partitions, manifests):
        worker_directory = partition_root / f"worker-{partition.index:03d}"
        actual_tar_paths = {
            path.resolve()
            for path in worker_directory.rglob("*.tar")
        } if worker_directory.is_dir() else set()
        if is_empty_manifest(manifest):
            if actual_tar_paths:
                raise ValueError(
                    "nuPlan empty worker produced unreported shards"
                )
            empty_reason = "nuPlan partition returned no manifest"
            empty_prefiltered_count = 0
            empty_prefiltered_samples: list[object] = []
            empty_rejected_count = 0
            if manifest is not None:
                raw_empty_reason = manifest.get("empty_reason")
                raw_prefiltered_count = manifest.get("prefiltered_count")
                raw_prefiltered_samples = manifest.get(
                    "prefiltered_samples"
                )
                raw_rejected_count = manifest.get("rejected_count")
                if (
                    not isinstance(raw_empty_reason, str)
                    or not raw_empty_reason
                    or not isinstance(raw_prefiltered_count, int)
                    or isinstance(raw_prefiltered_count, bool)
                    or raw_prefiltered_count < 0
                    or not isinstance(raw_prefiltered_samples, list)
                    or len(raw_prefiltered_samples)
                    != raw_prefiltered_count
                    or not isinstance(raw_rejected_count, int)
                    or isinstance(raw_rejected_count, bool)
                    or raw_rejected_count < 0
                ):
                    raise ValueError(
                        "nuPlan empty worker diagnostics are invalid"
                    )
                empty_reason = raw_empty_reason
                empty_prefiltered_count = raw_prefiltered_count
                empty_prefiltered_samples = raw_prefiltered_samples
                empty_rejected_count = raw_rejected_count
                merged_prefiltered.extend(empty_prefiltered_samples)
            partition_statistics.append({
                "accepted_count": 0,
                "db_file_count": len(partition.db_files),
                "empty_reason": empty_reason,
                "is_empty": True,
                "prefiltered_count": empty_prefiltered_count,
                "rejected_count": empty_rejected_count,
                "scenario_estimate": partition.scenario_estimate,
                "scenario_limit": partition.scenario_limit,
            })
            continue
        assert manifest is not None
        shard_names = manifest.get("shard_names")
        shard_counts = manifest.get("shard_sample_counts")
        shard_hashes = manifest.get("shard_sha256")
        if (
            not isinstance(shard_names, list)
            or not isinstance(shard_counts, Mapping)
            or not isinstance(shard_hashes, Mapping)
        ):
            raise ValueError("nuPlan worker manifest shard contract is invalid")
        if (
            any(
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".tar")
                for name in shard_names
            )
            or len(set(shard_names)) != len(shard_names)
            or set(shard_counts) != set(shard_names)
            or set(shard_hashes) != set(shard_names)
        ):
            raise ValueError("nuPlan worker manifest shard contract is invalid")
        worker_total_samples = manifest.get("total_samples")
        if (
            not isinstance(worker_total_samples, int)
            or isinstance(worker_total_samples, bool)
            or worker_total_samples < 0
        ):
            raise ValueError(
                "nuPlan worker manifest sample count is invalid"
            )
        expected_tar_paths = {
            (worker_directory / name).resolve()
            for name in shard_names
        }
        if actual_tar_paths != expected_tar_paths:
            raise ValueError(
                "nuPlan worker manifest does not account for every shard"
            )
        worker_shard_total = 0
        for worker_name in shard_names:
            source_path = worker_directory / worker_name
            expected_count = shard_counts[worker_name]
            expected_hash = shard_hashes[worker_name]
            if (
                not isinstance(expected_count, int)
                or isinstance(expected_count, bool)
                or expected_count <= 0
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
            ):
                raise ValueError(
                    "nuPlan worker manifest shard contract is invalid"
                )
            actual_hash = _sha256_file(source_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"nuPlan worker shard checksum mismatch: {worker_name}"
                )
            with tarfile.open(source_path) as archive:
                shard_sample_uids = {
                    member.name.partition(".")[0]
                    for member in archive
                    if member.isfile()
                }
            if len(shard_sample_uids) != expected_count:
                raise ValueError(
                    "nuPlan worker shard sample count mismatch: "
                    f"{worker_name}"
                )
            duplicate_uids = merged_sample_uids.intersection(
                shard_sample_uids
            )
            if duplicate_uids:
                raise ValueError(
                    "nuPlan parallel packing produced duplicate sample UIDs"
                )
            merged_sample_uids.update(shard_sample_uids)
            merged_name = (
                f"nuplan-{len(merged_shard_names):06d}.tar"
            )
            merged_shard_names.append(merged_name)
            merged_shard_counts[merged_name] = expected_count
            merged_shard_hashes[merged_name] = actual_hash
            staged_shards.append((source_path, merged_name))
            worker_shard_total += expected_count
        if worker_shard_total != worker_total_samples:
            raise ValueError(
                "nuPlan worker manifest total sample count mismatch"
            )
        rejections = manifest.get("rejected_samples")
        if not isinstance(rejections, list):
            raise ValueError("nuPlan worker rejection contract is invalid")
        worker_rejection_count = manifest.get("rejection_count")
        if (
            not isinstance(worker_rejection_count, int)
            or isinstance(worker_rejection_count, bool)
            or worker_rejection_count != len(rejections)
        ):
            raise ValueError("nuPlan worker rejection contract is invalid")
        merged_rejections.extend(rejections)
        prefiltered_samples = manifest.get("prefiltered_samples")
        prefiltered_count = manifest.get("prefiltered_count")
        if (
            not isinstance(prefiltered_samples, list)
            or not isinstance(prefiltered_count, int)
            or isinstance(prefiltered_count, bool)
            or prefiltered_count != len(prefiltered_samples)
        ):
            raise ValueError(
                "nuPlan worker prefilter contract is invalid"
            )
        merged_prefiltered.extend(prefiltered_samples)
        worker_max_camera_time_offset_us = manifest.get(
            "max_camera_time_offset_us"
        )
        if (
            not isinstance(worker_max_camera_time_offset_us, int)
            or isinstance(worker_max_camera_time_offset_us, bool)
            or not 0
            <= worker_max_camera_time_offset_us
            <= NUPLAN_CAMERA_SYNC_TOLERANCE_US
        ):
            raise ValueError(
                "nuPlan worker camera timing contract is invalid"
            )
        merged_max_camera_time_offset_us = max(
            merged_max_camera_time_offset_us,
            worker_max_camera_time_offset_us,
        )
        worker_max_history_offset = manifest.get(
            "max_history_camera_time_offset_us"
        )
        worker_distinct_history_count = manifest.get(
            "distinct_history_frame_count"
        )
        expected_distinct_count = (
            REACTIVE_BEVFORMER_HISTORY_FRAMES
            if worker_total_samples > 0
            else 0
        )
        if (
            not isinstance(worker_max_history_offset, int)
            or isinstance(worker_max_history_offset, bool)
            or not 0
            <= worker_max_history_offset
            <= NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US
            or worker_distinct_history_count != expected_distinct_count
        ):
            raise ValueError(
                "nuPlan worker history timing contract is invalid"
            )
        if worker_total_samples > 0:
            merged_max_history_camera_time_offset_us = max(
                merged_max_history_camera_time_offset_us,
                worker_max_history_offset,
            )
            merged_distinct_history_frame_count = (
                worker_distinct_history_count
                if merged_distinct_history_frame_count == 0
                else min(
                    merged_distinct_history_frame_count,
                    worker_distinct_history_count,
                )
            )
        worker_split_group_count = manifest.get("split_group_count")
        worker_split_group_uids = manifest.get("split_group_uids")
        if (
            not isinstance(worker_split_group_count, int)
            or isinstance(worker_split_group_count, bool)
            or worker_split_group_count < 0
            or not isinstance(worker_split_group_uids, list)
            or any(
                not isinstance(uid, str) or not uid
                for uid in worker_split_group_uids
            )
            or len(set(worker_split_group_uids))
            != worker_split_group_count
        ):
            raise ValueError(
                "nuPlan worker split group contract is invalid"
            )
        merged_split_group_uids.update(worker_split_group_uids)
        partition_statistics.append({
            "accepted_count": worker_total_samples,
            "db_file_count": len(partition.db_files),
            "is_empty": False,
            "prefiltered_count": prefiltered_count,
            "rejected_count": worker_rejection_count,
            "scenario_estimate": partition.scenario_estimate,
            "scenario_limit": partition.scenario_limit,
        })

    accepted_count = len(merged_sample_uids)
    rejected_count = len(merged_rejections)
    total_count = accepted_count + rejected_count
    if total_count == 0:
        raise ValueError("nuPlan parallel packing produced no scenarios")
    packing_scenario_limit = sum(
        partition.scenario_limit for partition in partitions
    )
    if (
        packing_scenario_limit
        and accepted_count != packing_scenario_limit
    ):
        empty_details = [
            {
                "partition": partition.index,
                "prefiltered": statistics["prefiltered_count"],
                "reason": statistics.get("empty_reason", ""),
            }
            for partition, statistics in zip(
                partitions,
                partition_statistics,
            )
            if statistics["is_empty"]
        ]
        raise ValueError(
            "nuPlan parallel packing did not fill its scenario limit: "
            f"accepted={accepted_count} rejected={rejected_count} "
            f"limit={packing_scenario_limit} "
            f"empty_partitions={empty_details}"
        )
    rejection_fraction = rejected_count / total_count
    if (
        accepted_count == 0
        or rejection_fraction > max_rejection_fraction
    ):
        raise ValueError(
            "nuPlan parallel packing rejection policy failed: "
            f"accepted={accepted_count} rejected={rejected_count} "
            f"fraction={rejection_fraction:.6f}"
        )
    if expected_split_group_uids is not None:
        missing_logs = sorted(
            expected_split_group_uids - merged_split_group_uids
        )
        unexpected_logs = sorted(
            merged_split_group_uids - expected_split_group_uids
        )
        if unexpected_logs or (require_full_log_coverage and missing_logs):
            raise ValueError(
                "nuPlan parallel packing source-log coverage mismatch: "
                f"missing={missing_logs[:3]} "
                f"unexpected={unexpected_logs[:3]} "
                f"expected_count={len(expected_split_group_uids)} "
                f"actual_count={len(merged_split_group_uids)}"
            )

    for source_path, merged_name in staged_shards:
        source_path.replace(output / merged_name)
    if any(partition_root.rglob("*.tar")):
        raise ValueError("nuPlan parallel merge left unconsumed shards")

    merged = dict(reference_manifest)
    merged.update({
        "bev_segmentation_count": accepted_count,
        "bev_statistics_count": accepted_count,
        "packing_partitions": partition_statistics,
        "packing_nonempty_workers": len(nonempty_manifests),
        "packing_scenario_limit": packing_scenario_limit,
        "packing_workers": len(partitions),
        "max_camera_time_offset_us": (
            merged_max_camera_time_offset_us
        ),
        "max_history_camera_time_offset_us": (
            merged_max_history_camera_time_offset_us
        ),
        "distinct_history_frame_count": (
            merged_distinct_history_frame_count
        ),
        "prefiltered_count": len(merged_prefiltered),
        "prefiltered_samples": merged_prefiltered,
        "rejected_samples": merged_rejections,
        "rejection_count": rejected_count,
        "rejection_fraction": rejection_fraction,
        "sample_uid_digest": hashlib.sha256(
            "\n".join(sorted(merged_sample_uids)).encode("ascii")
        ).hexdigest(),
        "shard_names": merged_shard_names,
        "shard_sample_counts": merged_shard_counts,
        "shard_sha256": merged_shard_hashes,
        "split_group_count": len(merged_split_group_uids),
        "split_group_uids": sorted(merged_split_group_uids),
        "total_samples": accepted_count,
        "trajectory_xy_count": accepted_count,
    })
    if expected_split_group_uids is not None:
        merged.update({
            "packing_covered_log_count": len(merged_split_group_uids),
            "packing_source_log_count": len(expected_split_group_uids),
        })
    _write_nuplan_manifest_atomic(output, merged)
    shutil.rmtree(partition_root)
    return merged


def pack_nuplan_local_dataset(
    *,
    data_root: str | Path,
    map_root: str | Path,
    sensor_root: str | Path,
    db_files: Sequence[str | Path],
    output_directory: str | Path,
    source_revision: str,
    map_version: str,
    limit_total_scenarios: int = 0,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
    samples_per_shard: int = 100,
    max_rejection_fraction: float = 0.0,
    pack_workers: int = 1,
    require_accepted: bool = True,
) -> dict[str, object]:
    """Build and pack scenarios from one materialized nuPlan dataset."""
    if not source_revision or not map_version:
        raise ValueError("nuPlan source_revision and map_version are required")
    if limit_total_scenarios < 0:
        raise ValueError("limit_total_scenarios must be non-negative")
    if (
        image_size != REACTIVE_CAMERA_IMAGE_SIZE
        or samples_per_shard <= 0
        or not 0.0 <= max_rejection_fraction <= 1.0
    ):
        raise ValueError("nuPlan packing limits are invalid")
    if pack_workers <= 0:
        raise ValueError("nuPlan pack_workers must be positive")
    local_data = Path(data_root).resolve()
    local_map = Path(map_root).resolve()
    local_sensor = Path(sensor_root).resolve()
    for name, path in (
        ("data_root", local_data),
        ("map_root", local_map),
        ("sensor_root", local_sensor),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"nuPlan {name} is not a directory: {path}")
    resolved_db_files = [Path(path).resolve() for path in db_files]
    if not resolved_db_files:
        raise ValueError("nuPlan db_files must not be empty")
    if len(set(resolved_db_files)) != len(resolved_db_files):
        raise ValueError("nuPlan db_files contains duplicate paths")
    db_stems = [path.stem for path in resolved_db_files]
    if len(set(db_stems)) != len(db_stems):
        raise ValueError("nuPlan db_files contains duplicate log names")
    for db_path in resolved_db_files:
        if not db_path.is_file() or db_path.suffix != ".db":
            raise FileNotFoundError(f"nuPlan DB is missing: {db_path}")
    _pin_nuplan_pack_thread_environment()

    if pack_workers > 1:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "nuPlan output directory must be empty"
            )
        weighted_db_files = [
            (db_path, _nuplan_db_scenario_count(db_path))
            for db_path in resolved_db_files
        ]
        partitions = _partition_weighted_nuplan_db_files(
            weighted_db_files,
            pack_workers,
            limit_total_scenarios,
        )
        partition_root = output / ".partitions"
        partition_root.mkdir()
        configs = [
            _NuPlanPackWorkerConfig(
                partition=partition,
                data_root=str(local_data),
                map_root=str(local_map),
                sensor_root=str(local_sensor),
                output_directory=str(
                    partition_root
                    / f"worker-{partition.index:03d}"
                ),
                source_revision=source_revision,
                map_version=map_version,
                image_size=image_size,
                samples_per_shard=samples_per_shard,
            )
            for partition in partitions
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(configs),
            mp_context=context,
            initializer=_initialize_nuplan_pack_worker,
        ) as executor:
            manifests = list(executor.map(
                _pack_nuplan_partition,
                configs,
            ))
        expected_split_groups = {
            _nuplan_split_group_uid(Path(db_path).stem)
            for db_path, count in weighted_db_files
            if count > 0
        }
        require_full_log_coverage = (
            not limit_total_scenarios
            or limit_total_scenarios >= len(expected_split_groups)
        )
        return _merge_nuplan_pack_partitions(
            output=output,
            partition_root=partition_root,
            partitions=partitions,
            manifests=manifests,
            max_rejection_fraction=max_rejection_fraction,
            expected_split_group_uids=expected_split_groups,
            require_full_log_coverage=require_full_log_coverage,
        )

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import (
        ScenarioFilter,
    )
    from nuplan.planning.utils.multithreading.worker_sequential import (
        Sequential,
    )

    os.environ["NUPLAN_DATA_STORE"] = "local"
    builder = NuPlanScenarioBuilder(
        data_root=str(local_data),
        map_root=str(local_map),
        sensor_root=str(local_sensor),
        db_files=[str(path) for path in resolved_db_files],
        map_version=map_version,
        include_cameras=True,
        max_workers=1,
        verbose=False,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )
    scenarios = builder.get_scenarios(
        scenario_filter,
        Sequential(),
    )
    return pack_nuplan_reactive_scenarios(
        scenarios,
        output_directory,
        source_revision=source_revision,
        map_version=map_version,
        image_size=image_size,
        samples_per_shard=samples_per_shard,
        max_rejection_fraction=max_rejection_fraction,
        require_accepted=require_accepted,
        prefilter_local_sensor_assets=bool(limit_total_scenarios),
        target_accepted_count=limit_total_scenarios,
    )


def pack_nuplan_reactive_scenarios(
    scenarios: Iterable[Any],
    output_directory: str | Path,
    *,
    source_revision: str,
    map_version: str,
    image_size: int = REACTIVE_CAMERA_IMAGE_SIZE,
    samples_per_shard: int = 100,
    max_rejection_fraction: float = 0.0,
    require_accepted: bool = True,
    prefilter_local_sensor_assets: bool = False,
    target_accepted_count: int = 0,
    sample_builder: Callable[..., tuple[str, str, dict[str, bytes]]] = (
        nuplan_reactive_sample_members
    ),
) -> dict[str, object]:
    """Pack raw scenarios into immutable Reactive training shards."""
    if not source_revision or not map_version:
        raise ValueError("nuPlan source and map revisions must be pinned")
    if (
        image_size != REACTIVE_CAMERA_IMAGE_SIZE
        or samples_per_shard <= 0
        or not 0.0 <= max_rejection_fraction <= 1.0
        or target_accepted_count < 0
    ):
        raise ValueError("nuPlan packing limits are invalid")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("nuPlan output directory must be empty")

    accepted: list[tuple[str, str]] = []
    accepted_camera_time_offsets_us: list[int] = []
    accepted_history_time_offsets_us: list[int] = []
    accepted_distinct_history_frame_counts: list[int] = []
    prefiltered: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    shard_names: list[str] = []
    archive: tarfile.TarFile | None = None
    covered_source_logs: set[str] = set()

    def coverage_first_scenarios() -> Iterator[Any]:
        indexed_scenarios = list(enumerate(scenarios))
        by_log: dict[str, list[tuple[int, Any]]] = {}
        for index, scenario in indexed_scenarios:
            log_name = str(getattr(scenario, "log_name", ""))
            by_log.setdefault(log_name, []).append((index, scenario))
        if (
            not target_accepted_count
            or target_accepted_count < len(by_log)
        ):
            yield from (scenario for _, scenario in indexed_scenarios)
            return

        positions = {log_name: 0 for log_name in by_log}
        consumed: set[int] = set()
        while True:
            uncovered = [
                log_name
                for log_name in sorted(by_log)
                if (
                    log_name not in covered_source_logs
                    and positions[log_name] < len(by_log[log_name])
                )
            ]
            if not uncovered:
                break
            for log_name in uncovered:
                position = positions[log_name]
                index, scenario = by_log[log_name][position]
                positions[log_name] = position + 1
                consumed.add(index)
                yield scenario
        yield from (
            scenario
            for index, scenario in indexed_scenarios
            if index not in consumed
        )

    try:
        for scenario in coverage_first_scenarios():
            if prefilter_local_sensor_assets:
                issue = _nuplan_local_sensor_asset_issue(scenario)
                if issue is not None:
                    prefiltered.append({
                        "log_name": str(
                            getattr(scenario, "log_name", "")
                        ),
                        "reason": issue,
                        "scenario_token": str(
                            getattr(scenario, "token", "")
                        ),
                    })
                    continue
            try:
                sample_uid, split_group_uid, members = sample_builder(
                    scenario,
                    iteration=0,
                    image_size=image_size,
                    source_revision=source_revision,
                )
                (
                    max_camera_time_offset_us,
                    max_history_time_offset_us,
                    distinct_history_frame_count,
                ) = (
                    _sample_camera_timing_metrics(members)
                )
            except Exception as error:
                rejected.append({
                    "error": f"{type(error).__name__}: {error}",
                    "log_name": str(
                        getattr(scenario, "log_name", "")
                    ),
                    "scenario_token": str(
                        getattr(scenario, "token", "")
                    ),
                })
                continue
            if len(accepted) % samples_per_shard == 0:
                if archive is not None:
                    archive.close()
                shard_name = f"nuplan-{len(shard_names):06d}.tar"
                archive = tarfile.open(output / shard_name, mode="w")
                shard_names.append(shard_name)
            assert archive is not None
            for suffix, payload in sorted(members.items()):
                _add_tar_member(
                    archive,
                    f"{sample_uid}.{suffix}",
                    payload,
                )
            accepted.append((sample_uid, split_group_uid))
            covered_source_logs.add(
                str(getattr(scenario, "log_name", ""))
            )
            accepted_camera_time_offsets_us.append(
                max_camera_time_offset_us
            )
            accepted_history_time_offsets_us.append(
                max_history_time_offset_us
            )
            accepted_distinct_history_frame_counts.append(
                distinct_history_frame_count
            )
            if (
                target_accepted_count
                and len(accepted) >= target_accepted_count
            ):
                break
    finally:
        if archive is not None:
            archive.close()

    if not accepted and not rejected:
        raise _NuPlanNoScenariosError(
            "nuPlan scenario builder returned no packable scenarios: "
            f"prefiltered={len(prefiltered)} rejected={len(rejected)}",
            prefiltered_samples=prefiltered,
        )
    if target_accepted_count and len(accepted) != target_accepted_count:
        rejection_reasons = sorted(
            Counter(
                str(sample.get("error", "unknown"))
                for sample in rejected
            ).items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        prefilter_reasons = sorted(
            Counter(
                str(sample.get("reason", "unknown"))
                for sample in prefiltered
            ).items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        raise ValueError(
            "nuPlan local sensor candidates did not fill the target: "
            f"accepted={len(accepted)} "
            f"prefiltered={len(prefiltered)} "
            f"rejected={len(rejected)} "
            f"target={target_accepted_count} "
            f"top_prefilter_reasons={prefilter_reasons} "
            f"top_rejection_reasons={rejection_reasons}"
        )
    total = len(accepted) + len(rejected)
    rejection_fraction = len(rejected) / total
    if (
        (require_accepted and not accepted)
        or rejection_fraction > max_rejection_fraction
    ):
        raise ValueError(
            "nuPlan packing rejection policy failed: "
            f"accepted={len(accepted)} rejected={len(rejected)} "
            f"fraction={rejection_fraction:.6f}"
        )
    shard_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in shard_names
    }
    shard_sample_counts = {
        name: min(
            samples_per_shard,
            len(accepted) - index * samples_per_shard,
        )
        for index, name in enumerate(shard_names)
    }
    split_group_uids = sorted({
        group_uid for _, group_uid in accepted
    })
    manifest: dict[str, object] = {
        "bev_segmentation_count": len(accepted),
        "bev_statistics_count": len(accepted),
        "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        "camera_order": list(NUPLAN_CAMERA_CHANNELS),
        "camera_slots": list(NUPLAN_CAMERA_SLOTS),
        "camera_visibility_heights_m": list(
            NUPLAN_CAMERA_VISIBILITY_HEIGHTS_M
        ),
        "contracts": contract_versions(),
        "dataset": "nuplan/nuplan-v1.1",
        "dataset_version": source_revision,
        "geometry_type": "rectified_pinhole",
        "has_bev_segmentation": True,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "front_camera_fpn_image_size": REACTIVE_CAMERA_IMAGE_SIZE,
        "front_camera_image_size": REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
        "front_camera_index": REACTIVE_FRONT_CAMERA_INDEX,
        "history_fallback_max_offset_us": (
            NUPLAN_HISTORY_FALLBACK_MAX_OFFSET_US
        ),
        "history_camera_spread_max_us": (
            NUPLAN_HISTORY_CAMERA_SPREAD_MAX_US
        ),
        "image_size": image_size,
        "map_context_channels": 14,
        "map_version": map_version,
        "max_camera_time_offset_us": max(
            accepted_camera_time_offsets_us,
            default=0,
        ),
        "max_history_camera_time_offset_us": max(
            accepted_history_time_offsets_us,
            default=0,
        ),
        "distinct_history_frame_count": min(
            accepted_distinct_history_frame_counts,
            default=0,
        ),
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": len(NUPLAN_CAMERA_CHANNELS),
        "projection_scope": "per_sample",
        "prefiltered_count": len(prefiltered),
        "prefiltered_samples": prefiltered,
        "rejected_samples": rejected,
        "rejection_count": len(rejected),
        "rejection_fraction": rejection_fraction,
        "route_channels": 2,
        "temporal_frame_interval_us": (
            REACTIVE_BEVFORMER_FRAME_INTERVAL_US
        ),
        "temporal_frame_offsets": list(
            REACTIVE_BEVFORMER_FRAME_OFFSETS
        ),
        "sample_uid_digest": hashlib.sha256(
            "\n".join(
                sorted(sample_uid for sample_uid, _ in accepted)
            ).encode("ascii")
        ).hexdigest(),
        "schema_version": NUPLAN_PACK_MANIFEST_VERSION,
        "shard_names": shard_names,
        "shard_sample_counts": shard_sample_counts,
        "shard_sha256": shard_hashes,
        "source_revision": source_revision,
        "split_group_count": len(split_group_uids),
        "split_group_uids": split_group_uids,
        "split_policy": "log_level_hash_bucket",
        "total_samples": len(accepted),
        "trajectory_xy_count": len(accepted),
    }
    _write_nuplan_manifest_atomic(output, manifest)
    return manifest
