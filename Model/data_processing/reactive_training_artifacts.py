"""Deterministic packed targets for Reactive multi-task training."""

from __future__ import annotations

import io
import math
import zipfile
from typing import Final

import numpy as np

from navigation.artifacts import encode_array
from navigation.contracts import canonical_json_bytes
from navigation.geometry import (
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
    NavigationRasterGeometry,
)


TRAJECTORY_XY_ARTIFACT_VERSION: Final = "trajectory_xy_v1"
BEV_SEGMENTATION_ARTIFACT_VERSION: Final = "bev_segmentation_v1"
REACTIVE_NAVIGATION_ARTIFACT_VERSION: Final = "sample_navigation_v3"
TRAJECTORY_XY_MEMBER: Final = "trajectory_xy.npz"
BEV_SEGMENTATION_MEMBER: Final = "bev_segmentation.npz"
BEV_SEGMENTATION_CLASSES: Final[tuple[str, ...]] = (
    "drivable_area",
    "lane_area",
    "intersection",
    "crosswalk",
    "stop_line",
    "vehicle",
    "vulnerable_road_user",
    "other_obstacle",
)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _encode_named_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        for name, array in sorted(arrays.items()):
            array_buffer = io.BytesIO()
            np.save(
                array_buffer,
                np.ascontiguousarray(array),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=_ZIP_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def _decode_named_arrays(payload: bytes) -> dict[str, np.ndarray]:
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        names = archive.namelist()
        if names != sorted(names) or any(
            not name.endswith(".npy") for name in names
        ):
            raise ValueError("packed target NPZ members are not canonical")
        for name in names:
            with archive.open(name) as stream:
                arrays[name.removesuffix(".npy")] = np.load(
                    io.BytesIO(stream.read()),
                    allow_pickle=False,
                )
    return arrays


def encode_trajectory_xy(
    trajectory_xy_m: np.ndarray,
    trajectory_valid: np.ndarray,
) -> bytes:
    xy = np.asarray(trajectory_xy_m, dtype=np.float32)
    valid = np.asarray(trajectory_valid, dtype=np.bool_)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("trajectory_xy_m must have shape [T,2]")
    if valid.shape != (xy.shape[0],):
        raise ValueError("trajectory_valid must have shape [T]")
    if not np.isfinite(xy[valid]).all():
        raise ValueError("valid trajectory positions must be finite")
    return _encode_named_arrays({
        "schema_version": np.frombuffer(
            TRAJECTORY_XY_ARTIFACT_VERSION.encode("ascii"),
            dtype=np.uint8,
        ),
        "trajectory_valid": valid.astype(np.uint8),
        "trajectory_xy_m": xy,
    })


def decode_trajectory_xy(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _decode_named_arrays(payload)
    required = {
        "schema_version",
        "trajectory_valid",
        "trajectory_xy_m",
    }
    if set(arrays) != required:
        raise ValueError("trajectory XY fields differ from contract")
    version = bytes(arrays["schema_version"]).decode("ascii")
    if version != TRAJECTORY_XY_ARTIFACT_VERSION:
        raise ValueError("unsupported trajectory XY artifact version")
    xy = np.asarray(arrays["trajectory_xy_m"], dtype=np.float32)
    valid_u8 = np.asarray(arrays["trajectory_valid"], dtype=np.uint8)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("trajectory XY artifact has invalid shape")
    if valid_u8.shape != (xy.shape[0],) or not np.isin(
        valid_u8,
        (0, 1),
    ).all():
        raise ValueError("trajectory validity artifact is invalid")
    valid = valid_u8.astype(np.bool_)
    if not np.isfinite(xy[valid]).all():
        raise ValueError("valid trajectory positions must be finite")
    return np.ascontiguousarray(xy), np.ascontiguousarray(valid)


def wgs84_future_to_ego_xy(
    gps_future_lat_lon: np.ndarray,
    *,
    current_latitude_deg: float,
    current_longitude_deg: float,
    heading_deg_cw_from_north: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project current+future WGS84 points into current ego FLU."""
    gps = np.asarray(gps_future_lat_lon, dtype=np.float64)
    if gps.shape != (65, 2):
        raise ValueError("gps future must have shape [65,2]")
    pose = np.asarray(
        [
            current_latitude_deg,
            current_longitude_deg,
            heading_deg_cw_from_north,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(gps).all() or not np.isfinite(pose).all():
        raise ValueError("geospatial trajectory contains non-finite values")
    if not np.allclose(
        gps[0],
        pose[:2],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("first GPS point differs from current pose")
    earth_radius_m = 6_378_137.0
    degrees_to_meters = earth_radius_m * math.pi / 180.0
    east = (
        (gps[1:, 1] - current_longitude_deg)
        * math.cos(math.radians(current_latitude_deg))
        * degrees_to_meters
    )
    north = (
        gps[1:, 0] - current_latitude_deg
    ) * degrees_to_meters
    heading = math.radians(heading_deg_cw_from_north)
    forward = east * math.sin(heading) + north * math.cos(heading)
    left = -east * math.cos(heading) + north * math.sin(heading)
    trajectory = np.column_stack([forward, left]).astype(np.float32)
    finite = np.asarray(
        np.isfinite(trajectory).all(axis=1),
        dtype=np.bool_,
    )
    trajectory[~finite] = 0.0
    return trajectory, finite


def encode_reactive_navigation(
    map_context: np.ndarray,
    route_target: np.ndarray,
    *,
    map_valid: bool,
    route_channel_valid: np.ndarray,
    geometry: NavigationRasterGeometry,
    metadata: dict[str, object] | None = None,
) -> dict[str, bytes]:
    """Encode the map/route contract used by nuPlan and L2D training."""
    map_array = np.asarray(map_context, dtype=np.float32)
    route_array = np.asarray(route_target, dtype=np.float32)
    channel_valid = np.asarray(route_channel_valid, dtype=np.bool_)
    expected_map = (
        MAP_CHANNEL_COUNT,
        geometry.height_px,
        geometry.width_px,
    )
    expected_route = (
        ROUTE_CHANNEL_COUNT,
        geometry.height_px,
        geometry.width_px,
    )
    if map_array.shape != expected_map:
        raise ValueError(f"map_context must have shape {expected_map}")
    if route_array.shape != expected_route:
        raise ValueError(f"route_target must have shape {expected_route}")
    if channel_valid.shape != (ROUTE_CHANNEL_COUNT,):
        raise ValueError("route_channel_valid must have shape [2]")
    if (
        not np.isfinite(map_array).all()
        or not np.isfinite(route_array).all()
        or float(map_array.min(initial=0.0)) < 0.0
        or float(map_array.max(initial=0.0)) > 1.0
        or float(route_array.min(initial=0.0)) < 0.0
        or float(route_array.max(initial=0.0)) > 1.0
    ):
        raise ValueError("navigation rasters must be finite and in [0,1]")
    navigation_metadata: dict[str, object] = {
        "schema_version": REACTIVE_NAVIGATION_ARTIFACT_VERSION,
        "geometry_id": geometry.geometry_id,
        "map_valid": bool(map_valid),
        "route_valid": bool(channel_valid.any()),
        "route_channel_valid": channel_valid.tolist(),
    }
    for key, value in (metadata or {}).items():
        if key in navigation_metadata:
            raise ValueError(
                f"reactive navigation metadata collides with {key!r}"
            )
        navigation_metadata[key] = value
    return {
        "map_semantic.npz": encode_array(map_array),
        "route_mask.npz": encode_array(route_array),
        "navigation_meta.json": canonical_json_bytes(
            navigation_metadata
        ),
    }


def encode_bev_segmentation(
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> bytes:
    target_f32 = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if (
        target_f32.ndim != 3
        or target_f32.shape[0] != len(BEV_SEGMENTATION_CLASSES)
    ):
        raise ValueError("BEV target must have shape [8,H,W]")
    if valid.shape != target_f32.shape:
        raise ValueError("BEV valid mask must match target")
    if (
        not np.isfinite(target_f32).all()
        or float(target_f32.min(initial=0.0)) < 0.0
        or float(target_f32.max(initial=0.0)) > 1.0
    ):
        raise ValueError("BEV target must be finite and in [0,1]")
    target_u8 = np.rint(target_f32 * 255.0).astype(np.uint8)
    valid_bits = np.packbits(valid.reshape(-1), bitorder="little")
    return _encode_named_arrays({
        "schema_version": np.frombuffer(
            BEV_SEGMENTATION_ARTIFACT_VERSION.encode("ascii"),
            dtype=np.uint8,
        ),
        "shape": np.asarray(target_f32.shape, dtype=np.int32),
        "target_u8": target_u8,
        "valid_bits": valid_bits,
    })


def decode_bev_segmentation(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _decode_named_arrays(payload)
    required = {
        "schema_version",
        "shape",
        "target_u8",
        "valid_bits",
    }
    if set(arrays) != required:
        raise ValueError("BEV segmentation fields differ from contract")
    version = bytes(arrays["schema_version"]).decode("ascii")
    if version != BEV_SEGMENTATION_ARTIFACT_VERSION:
        raise ValueError("unsupported BEV segmentation artifact version")
    shape_array = np.asarray(arrays["shape"], dtype=np.int32)
    if shape_array.shape != (3,):
        raise ValueError("BEV segmentation shape metadata is invalid")
    shape = tuple(int(value) for value in shape_array)
    if shape[0] != len(BEV_SEGMENTATION_CLASSES):
        raise ValueError("BEV segmentation class count differs from taxonomy")
    target_u8 = np.asarray(arrays["target_u8"], dtype=np.uint8)
    if target_u8.shape != shape:
        raise ValueError("BEV segmentation target shape differs from metadata")
    cell_count = int(np.prod(shape))
    valid = np.unpackbits(
        np.asarray(arrays["valid_bits"], dtype=np.uint8),
        count=cell_count,
        bitorder="little",
    ).astype(np.bool_).reshape(shape)
    return (
        np.ascontiguousarray(target_u8.astype(np.float32) / 255.0),
        np.ascontiguousarray(valid),
    )
