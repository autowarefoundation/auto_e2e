"""Raw nuPlan scenario packing for Reactive multi-task training."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import pickle
import sqlite3
import tarfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_processing.contract_versions import contract_versions
from navigation.contracts import canonical_json_bytes
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    NavigationRasterGeometry,
)

from .targets import (
    NuPlanReactiveTargets,
    build_nuplan_reactive_targets,
    nuplan_reactive_target_members,
)

NUPLAN_CAMERA_CHANNELS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)
NUPLAN_RECTIFICATION_POLICY_VERSION = "nuplan_rectified_pinhole_v1"
NUPLAN_PACK_MANIFEST_VERSION = "nuplan_reactive_manifest_v1"


@dataclasses.dataclass(frozen=True)
class NuPlanCameraBundle:
    """Rectified camera pixels and reference-pose projection matrices."""

    jpeg_by_channel: Mapping[str, bytes]
    projection_matrices: np.ndarray
    camera_visibility: np.ndarray
    metadata: Mapping[str, object]


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
    points = np.stack(
        [
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            np.zeros(x_grid.size, dtype=np.float64),
            np.ones(x_grid.size, dtype=np.float64),
        ],
        axis=0,
    )
    visible = np.zeros(x_grid.size, dtype=np.bool_)
    for matrix in matrices:
        projected = matrix @ points
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
        )
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
    maximum_range = np.zeros(angular_bins, dtype=np.float64)
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
) -> tuple[sqlite3.Row, dict[str, sqlite3.Row]]:
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
                int(reference["timestamp"]) - 50_000,
                int(reference["timestamp"]) + 50_000,
                int(reference["timestamp"]),
            ),
        ).fetchall()
    finally:
        connection.close()
    by_channel: dict[str, sqlite3.Row] = {}
    for row in rows:
        by_channel.setdefault(str(row["channel"]), row)
    missing = set(NUPLAN_CAMERA_CHANNELS) - set(by_channel)
    if missing:
        raise ValueError(
            f"nuPlan sample is missing required cameras: {sorted(missing)}"
        )
    return reference, by_channel


def load_nuplan_camera_bundle(
    scenario: Any,
    *,
    iteration: int = 0,
    image_size: int = 256,
    geometry: NavigationRasterGeometry = AUTOE2E_NAVIGATION_GEOMETRY,
) -> NuPlanCameraBundle:
    """Load, rectify, and pose-compensate all eight nuPlan cameras."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuPlan offline camera rectification requires OpenCV"
        ) from exc
    if image_size <= 0:
        raise ValueError("image_size must be positive")
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
    reference, rows = _camera_rows(log_file, lidar_token)
    reference_pose = _quaternion_transform(
        [reference["x"], reference["y"], reference["z"]],
        [
            reference["qw"],
            reference["qx"],
            reference["qy"],
            reference["qz"],
        ],
    )

    jpegs: dict[str, bytes] = {}
    matrices = []
    camera_metadata = []
    reference_timestamp = int(reference["timestamp"])
    for channel in NUPLAN_CAMERA_CHANNELS:
        row = rows[channel]
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
            distortion_raw or [],
            dtype=np.float64,
        ).reshape(-1)
        if not np.isfinite(distortion).all():
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
        resized = Image.fromarray(rectified).resize(
            (image_size, image_size),
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

        scaled_intrinsic = rectified_intrinsic.copy()
        scaled_intrinsic[0] *= image_size / native_width
        scaled_intrinsic[1] *= image_size / native_height
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
        matrix = scaled_intrinsic @ camera_from_reference[:3]
        matrices.append(matrix)
        camera_metadata.append({
            "channel": channel,
            "distortion": distortion.tolist(),
            "image_time_offset_us": (
                int(row["timestamp"]) - reference_timestamp
            ),
            "native_intrinsic": intrinsic.tolist(),
            "native_size_wh": [native_width, native_height],
            "rectified_intrinsic": rectified_intrinsic.tolist(),
            "scaled_rectified_intrinsic": scaled_intrinsic.tolist(),
            "sensor_to_ego": ego_from_camera.tolist(),
        })

    projection_matrices = np.stack(matrices).astype(np.float32)
    visibility = camera_visibility_from_projection_matrices(
        projection_matrices,
        image_width=image_size,
        image_height=image_size,
        geometry=geometry,
    )
    return NuPlanCameraBundle(
        jpeg_by_channel=jpegs,
        projection_matrices=projection_matrices,
        camera_visibility=visibility,
        metadata={
            "camera_order": list(NUPLAN_CAMERA_CHANNELS),
            "cameras": camera_metadata,
            "image_size": image_size,
            "rectification_policy": NUPLAN_RECTIFICATION_POLICY_VERSION,
            "reference_lidar_timestamp_us": reference_timestamp,
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
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3:
        raise ValueError("nuPlan merged point cloud has invalid shape")
    lidar_from_ego = np.asarray(
        scenario.get_lidar_to_ego_transform(),
        dtype=np.float64,
    )
    if lidar_from_ego.shape != (4, 4):
        raise ValueError("nuPlan lidar-to-ego transform is invalid")
    homogeneous = np.vstack([
        points[:3],
        np.ones(points.shape[1], dtype=np.float64),
    ])
    points_ego = (lidar_from_ego @ homogeneous)[:3].T
    return lidar_observability_from_points(
        points_ego,
        geometry=geometry,
    )


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


def _sample_identity(scenario: Any) -> tuple[str, str]:
    log_name = str(getattr(scenario, "log_name", ""))
    token = str(getattr(scenario, "token", ""))
    if not log_name or not token:
        raise ValueError("nuPlan scenario lacks log name or token")
    sample_digest = hashlib.sha256(
        f"{log_name}:{token}".encode("utf-8")
    ).hexdigest()[:24]
    log_digest = hashlib.sha256(
        log_name.encode("utf-8")
    ).hexdigest()[:20]
    return f"nuplan-{sample_digest}", f"nuplan-log-{log_digest}"


def nuplan_reactive_sample_members(
    scenario: Any,
    *,
    iteration: int = 0,
    image_size: int = 256,
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


def pack_nuplan_reactive_scenarios(
    scenarios: Iterable[Any],
    output_directory: str | Path,
    *,
    source_revision: str,
    map_version: str,
    image_size: int = 256,
    samples_per_shard: int = 1000,
    max_rejection_fraction: float = 0.0,
    sample_builder: Callable[..., tuple[str, str, dict[str, bytes]]] = (
        nuplan_reactive_sample_members
    ),
) -> dict[str, object]:
    """Pack raw scenarios into immutable Reactive training shards."""
    if not source_revision or not map_version:
        raise ValueError("nuPlan source and map revisions must be pinned")
    if samples_per_shard <= 0 or not 0.0 <= max_rejection_fraction < 1.0:
        raise ValueError("nuPlan packing limits are invalid")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("nuPlan output directory must be empty")

    accepted: list[tuple[str, str]] = []
    rejected: list[dict[str, str]] = []
    shard_names: list[str] = []
    archive: tarfile.TarFile | None = None
    try:
        for scenario_index, scenario in enumerate(scenarios):
            try:
                sample_uid, split_group_uid, members = sample_builder(
                    scenario,
                    iteration=0,
                    image_size=image_size,
                    source_revision=source_revision,
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
    finally:
        if archive is not None:
            archive.close()

    total = len(accepted) + len(rejected)
    if total == 0:
        raise ValueError("nuPlan scenario builder returned no scenarios")
    rejection_fraction = len(rejected) / total
    if not accepted or rejection_fraction > max_rejection_fraction:
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
    manifest: dict[str, object] = {
        "bev_segmentation_count": len(accepted),
        "bev_taxonomy_version": "bev_segmentation_v1",
        "camera_order": list(NUPLAN_CAMERA_CHANNELS),
        "contracts": contract_versions(),
        "dataset": "nuplan/nuplan-v1.1",
        "dataset_version": source_revision,
        "geometry_type": "rectified_pinhole",
        "has_bev_segmentation": True,
        "has_reactive_navigation": True,
        "has_route_reconstruction": True,
        "has_trajectory_xy": True,
        "map_context_channels": 14,
        "map_version": map_version,
        "navigation_geometry": (
            AUTOE2E_NAVIGATION_GEOMETRY.contract()
        ),
        "num_views": len(NUPLAN_CAMERA_CHANNELS),
        "projection_scope": "per_sample",
        "rejected_samples": rejected,
        "rejection_count": len(rejected),
        "rejection_fraction": rejection_fraction,
        "route_channels": 2,
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
        "split_group_count": len({
            group_uid for _, group_uid in accepted
        }),
        "split_policy": "log_level_hash_bucket",
        "total_samples": len(accepted),
        "trajectory_xy_count": len(accepted),
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
