"""Deterministic scene and sample codecs for navigation shard members."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from .contracts import (
    Destination,
    DirectedLaneField,
    Maneuver,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    PolygonPrimitive,
    PolylinePrimitive,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
    StaticTrafficSignal,
    TransitionType,
    canonical_json_bytes,
)
from .rasterizer import EgoPose, NavigationRaster
from .supervision import (
    ROUTE_SUPERVISION_ARTIFACT_VERSION,
    RouteSupervision,
)


SCENE_NAVIGATION_ARTIFACT_VERSION = "scene_navigation_v1"
SAMPLE_NAVIGATION_ARTIFACT_VERSION = "sample_navigation_v2"
ROUTE_SUPERVISION_MEMBER = "route_supervision.npz"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def encode_array(array: np.ndarray) -> bytes:
    """Encode one array as deterministic compressed NPZ bytes."""
    array_buffer = io.BytesIO()
    np.save(array_buffer, np.ascontiguousarray(array), allow_pickle=False)
    info = zipfile.ZipInfo("array.npy", date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def decode_array(payload: bytes) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        names = archive.namelist()
        if names != ["array.npy"]:
            raise ValueError(f"navigation NPZ members differ from contract: {names}")
        with archive.open("array.npy") as stream:
            return np.load(io.BytesIO(stream.read()), allow_pickle=False)


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
            value = np.asarray(array)
            if value.ndim > 0:
                value = np.ascontiguousarray(value)
            np.save(
                array_buffer,
                value,
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
            raise ValueError("route supervision NPZ members are not canonical")
        for name in names:
            with archive.open(name) as stream:
                arrays[name.removesuffix(".npy")] = np.load(
                    io.BytesIO(stream.read()),
                    allow_pickle=False,
                )
    return arrays


def encode_route_supervision(supervision: RouteSupervision) -> bytes:
    return _encode_named_arrays(supervision.arrays())


def decode_route_supervision(members: dict[str, bytes]) -> RouteSupervision:
    if ROUTE_SUPERVISION_MEMBER not in members:
        raise ValueError("navigation sample is missing route supervision")
    arrays = _decode_named_arrays(members[ROUTE_SUPERVISION_MEMBER])
    required = {
        "distance_to_corridor_m",
        "distance_to_drivable_m",
        "drivable_available",
        "route_heading_sin",
        "route_heading_cos",
        "route_heading_valid",
        "destination_xy_m",
        "destination_visible",
    }
    if set(arrays) != required:
        raise ValueError(
            "route supervision fields differ from contract: "
            f"{sorted(arrays)}"
        )
    visible = np.asarray(arrays["destination_visible"])
    if visible.shape != () or int(visible) not in (0, 1):
        raise ValueError("destination visibility must be a binary scalar")
    drivable_available = np.asarray(arrays["drivable_available"])
    if (
        drivable_available.shape != ()
        or int(drivable_available) not in (0, 1)
    ):
        raise ValueError(
            "drivable availability must be a binary scalar"
        )
    return RouteSupervision(
        distance_to_corridor_m=arrays["distance_to_corridor_m"],
        distance_to_drivable_m=arrays["distance_to_drivable_m"],
        drivable_available=bool(int(drivable_available)),
        route_heading_sin=arrays["route_heading_sin"],
        route_heading_cos=arrays["route_heading_cos"],
        route_heading_valid=arrays["route_heading_valid"],
        destination_xy_m=arrays["destination_xy_m"],
        destination_visible=bool(int(visible)),
    )


def encode_scene_navigation(
    navigation_map: NavigationMap,
    route: NavigationRoute,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": SCENE_NAVIGATION_ARTIFACT_VERSION,
            "navigation_map": navigation_map,
            "navigation_route": route,
        }
    )


def _frame(data: dict[str, Any]) -> MapFrame:
    return MapFrame(**data)


def _polyline(data: dict[str, Any]) -> PolylinePrimitive:
    return PolylinePrimitive(
        primitive_id=data["primitive_id"],
        points_enu_m=np.asarray(data["points_enu_m"], dtype=np.float64),
        level=data.get("level"),
    )


def _polygon(data: dict[str, Any]) -> PolygonPrimitive:
    return PolygonPrimitive(
        primitive_id=data["primitive_id"],
        points_enu_m=np.asarray(data["points_enu_m"], dtype=np.float64),
        level=data.get("level"),
    )


def decode_scene_navigation(
    payload: bytes,
) -> tuple[NavigationMap, NavigationRoute]:
    data = json.loads(payload)
    if data.get("schema_version") != SCENE_NAVIGATION_ARTIFACT_VERSION:
        raise ValueError("unsupported scene navigation artifact version")
    map_data = data["navigation_map"]
    navigation_map = NavigationMap(
        schema_version=map_data["schema_version"],
        map_version=map_data["map_version"],
        provider=map_data["provider"],
        frame=_frame(map_data["frame"]),
        bounds_enu_m=tuple(map_data["bounds_enu_m"]),
        drivable_polygons=tuple(
            _polygon(value) for value in map_data["drivable_polygons"]
        ),
        lane_boundaries=tuple(
            _polyline(value) for value in map_data["lane_boundaries"]
        ),
        lane_centerlines=tuple(
            _polyline(value) for value in map_data["lane_centerlines"]
        ),
        intersection_polygons=tuple(
            _polygon(value) for value in map_data["intersection_polygons"]
        ),
        crosswalk_polygons=tuple(
            _polygon(value) for value in map_data["crosswalk_polygons"]
        ),
        stop_lines=tuple(
            _polyline(value) for value in map_data["stop_lines"]
        ),
        static_traffic_signals=tuple(
            StaticTrafficSignal(
                signal_id=value["signal_id"],
                position_enu_m=np.asarray(
                    value["position_enu_m"], dtype=np.float64
                ),
                level=value.get("level"),
            )
            for value in map_data["static_traffic_signals"]
        ),
        directed_lane_fields=tuple(
            DirectedLaneField(
                lane_id=value["lane_id"],
                centerline_enu_m=np.asarray(
                    value["centerline_enu_m"], dtype=np.float64
                ),
                level=value.get("level"),
            )
            for value in map_data["directed_lane_fields"]
        ),
        layer_availability=map_data["layer_availability"],
        provenance=map_data["provenance"],
    )

    route_data = data["navigation_route"]
    quality = route_data["quality"]
    provenance = route_data["provenance"]
    route = NavigationRoute(
        schema_version=route_data["schema_version"],
        route_id=route_data["route_id"],
        revision=int(route_data["revision"]),
        provider=route_data["provider"],
        timestamp_ns=int(route_data["timestamp_ns"]),
        valid_from_ns=int(route_data["valid_from_ns"]),
        map_version=route_data["map_version"],
        frame=_frame(route_data["frame"]),
        lane_sequence=tuple(
            RouteLaneSegment(
                lane_id=value["lane_id"],
                provider_segment_id=value["provider_segment_id"],
                centerline_enu_m=np.asarray(
                    value["centerline_enu_m"], dtype=np.float64
                ),
                left_boundary_enu_m=(
                    np.asarray(value["left_boundary_enu_m"], dtype=np.float64)
                    if value.get("left_boundary_enu_m") is not None
                    else None
                ),
                right_boundary_enu_m=(
                    np.asarray(value["right_boundary_enu_m"], dtype=np.float64)
                    if value.get("right_boundary_enu_m") is not None
                    else None
                ),
                level=value.get("level"),
                transition_from_previous=TransitionType(
                    value["transition_from_previous"]
                ),
                maneuver=Maneuver(value["maneuver"]),
                confidence=float(value["confidence"]),
            )
            for value in route_data["lane_sequence"]
        ),
        destination=Destination(
            position_enu_m=np.asarray(
                route_data["destination"]["position_enu_m"],
                dtype=np.float64,
            ),
            source=route_data["destination"]["source"],
        ),
        confidence=float(route_data["confidence"]),
        valid=bool(route_data["valid"]),
        quality=RouteQuality(
            matched_pose_ratio=float(quality["matched_pose_ratio"]),
            median_lateral_distance_m=float(
                quality["median_lateral_distance_m"]
            ),
            p95_lateral_distance_m=float(quality["p95_lateral_distance_m"]),
            median_heading_error_rad=float(
                quality["median_heading_error_rad"]
            ),
            p95_heading_error_rad=float(quality["p95_heading_error_rad"]),
            shortest_path_fill_count=int(
                quality["shortest_path_fill_count"]
            ),
            shortest_path_fill_length_m=float(
                quality["shortest_path_fill_length_m"]
            ),
            adjacent_transition_count=int(
                quality["adjacent_transition_count"]
            ),
            unresolved_discontinuities=int(
                quality["unresolved_discontinuities"]
            ),
            failure_reasons=tuple(quality["failure_reasons"]),
        ),
        estimated_destination=bool(route_data["estimated_destination"]),
        provenance=RouteProvenance(
            source_revision=provenance["source_revision"],
            matcher_version=provenance["matcher_version"],
            matcher_config_sha256=provenance["matcher_config_sha256"],
            map_sha256=provenance["map_sha256"],
            trace_sha256=provenance["trace_sha256"],
        ),
    )
    return navigation_map, route


def encode_sample_navigation(
    raster: NavigationRaster,
    *,
    route_supervision: RouteSupervision,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, bytes]:
    metadata = {
        "schema_version": SAMPLE_NAVIGATION_ARTIFACT_VERSION,
        "route_supervision_version": ROUTE_SUPERVISION_ARTIFACT_VERSION,
        **raster.metadata(),
    }
    for key, value in (extra_metadata or {}).items():
        if key in metadata:
            raise ValueError(
                f"extra navigation metadata collides with {key!r}"
            )
        metadata[key] = value
    return {
        "map_semantic.npz": encode_array(raster.map_context),
        "route_mask.npz": encode_array(raster.route_mask),
        ROUTE_SUPERVISION_MEMBER: encode_route_supervision(
            route_supervision
        ),
        "navigation_meta.json": canonical_json_bytes(metadata),
    }


def decode_sample_navigation(
    members: dict[str, bytes],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = {
        "map_semantic.npz",
        "route_mask.npz",
        ROUTE_SUPERVISION_MEMBER,
        "navigation_meta.json",
    }
    missing = required - set(members)
    if missing:
        raise ValueError(f"navigation sample is missing members: {sorted(missing)}")
    map_context = np.ascontiguousarray(
        decode_array(members["map_semantic.npz"]),
        dtype=np.float32,
    )
    route_mask = np.ascontiguousarray(
        decode_array(members["route_mask.npz"]),
        dtype=np.uint8,
    )
    metadata = json.loads(members["navigation_meta.json"])
    if metadata.get("schema_version") != SAMPLE_NAVIGATION_ARTIFACT_VERSION:
        raise ValueError("unsupported sample navigation artifact version")
    if (
        metadata.get("route_supervision_version")
        != ROUTE_SUPERVISION_ARTIFACT_VERSION
    ):
        raise ValueError("unsupported route supervision artifact version")
    return map_context, route_mask, metadata


def raster_from_sample_members(
    members: dict[str, bytes],
) -> NavigationRaster:
    map_context, route_mask, metadata = decode_sample_navigation(members)

    def pose(value: dict[str, Any]) -> EgoPose:
        return EgoPose(
            x_enu_m=float(value["x_enu_m"]),
            y_enu_m=float(value["y_enu_m"]),
            yaw_rad=float(value["yaw_rad"]),
            timestamp_ns=int(value["timestamp_ns"]),
        )

    return NavigationRaster(
        map_context=map_context,
        route_mask=route_mask,
        map_valid=bool(metadata["map_valid"]),
        route_valid=bool(metadata["route_valid"]),
        geometry_id=metadata["geometry_id"],
        render_pose=pose(metadata["render_pose_enu"]),
        sample_pose=pose(metadata["sample_pose_enu"]),
        renderer_version=metadata["renderer_version"],
        map_version=metadata["map_version"],
        route_id=metadata["route_id"],
        route_revision=int(metadata["route_revision"]),
        route_confidence=float(metadata["route_confidence"]),
        input_vector_sha256=metadata["input_vector_sha256"],
    )
