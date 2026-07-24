"""Thin Python binding for the authoritative C++ navigation rasterizer."""

from __future__ import annotations

import ctypes
import dataclasses
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    NavigationMap,
    NavigationRoute,
    PolygonPrimitive,
    PolylinePrimitive,
    StaticTrafficSignal,
    contract_sha256,
)
from .geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
    MapChannel,
    NavigationRasterGeometry,
)


class NativeRasterizerUnavailable(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class EgoPose:
    x_enu_m: float
    y_enu_m: float
    yaw_rad: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        values = (self.x_enu_m, self.y_enu_m, self.yaw_rad)
        if not np.isfinite(values).all():
            raise ValueError("ego pose contains non-finite values")
        if self.timestamp_ns < 0:
            raise ValueError("ego pose timestamp must be non-negative")


@dataclasses.dataclass(frozen=True)
class NavigationRaster:
    map_context: np.ndarray
    route_mask: np.ndarray
    map_valid: bool
    route_valid: bool
    geometry_id: str
    render_pose: EgoPose
    sample_pose: EgoPose
    renderer_version: str
    map_version: str
    route_id: str
    route_revision: int
    route_confidence: float
    input_vector_sha256: str

    def __post_init__(self) -> None:
        map_context = np.ascontiguousarray(self.map_context, dtype=np.float32)
        route_mask = np.ascontiguousarray(self.route_mask, dtype=np.uint8)
        if map_context.ndim != 3 or map_context.shape[0] != MAP_CHANNEL_COUNT:
            raise ValueError("map_context must have shape [14,H,W]")
        if route_mask.ndim != 3 or route_mask.shape[0] != ROUTE_CHANNEL_COUNT:
            raise ValueError("route_mask must have shape [2,H,W]")
        if map_context.shape[1:] != route_mask.shape[1:]:
            raise ValueError("map and route raster sizes differ")
        if not np.isfinite(map_context).all():
            raise ValueError("map_context contains non-finite values")
        if map_context.size and (
            float(map_context.min()) < 0.0
            or float(map_context.max()) > 1.0
        ):
            raise ValueError("map_context must be in [0,1]")
        if not np.isin(route_mask, (0, 1)).all():
            raise ValueError("route_mask must be binary")
        map_context.setflags(write=False)
        route_mask.setflags(write=False)
        object.__setattr__(self, "map_context", map_context)
        object.__setattr__(self, "route_mask", route_mask)

    @property
    def render_to_sample_se2(self) -> tuple[float, float, float]:
        dx_map = self.sample_pose.x_enu_m - self.render_pose.x_enu_m
        dy_map = self.sample_pose.y_enu_m - self.render_pose.y_enu_m
        cosine = float(np.cos(self.render_pose.yaw_rad))
        sine = float(np.sin(self.render_pose.yaw_rad))
        return (
            cosine * dx_map + sine * dy_map,
            -sine * dx_map + cosine * dy_map,
            self.sample_pose.yaw_rad - self.render_pose.yaw_rad,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "map_version": self.map_version,
            "route_id": self.route_id,
            "route_revision": self.route_revision,
            "render_timestamp_ns": self.render_pose.timestamp_ns,
            "localization_timestamp_ns": self.sample_pose.timestamp_ns,
            "sample_timestamp_ns": self.sample_pose.timestamp_ns,
            "render_pose_enu": dataclasses.asdict(self.render_pose),
            "sample_pose_enu": dataclasses.asdict(self.sample_pose),
            "render_to_sample_se2": list(self.render_to_sample_se2),
            "map_valid": self.map_valid,
            "route_valid": self.route_valid,
            "route_confidence": self.route_confidence,
            "renderer_version": self.renderer_version,
            "input_vector_sha256": self.input_vector_sha256,
        }


class _CGeometry(ctypes.Structure):
    _fields_ = [
        ("height_px", ctypes.c_int32),
        ("width_px", ctypes.c_int32),
        ("meters_per_pixel", ctypes.c_double),
        ("x_min_m", ctypes.c_double),
        ("x_max_m", ctypes.c_double),
        ("y_min_m", ctypes.c_double),
        ("y_max_m", ctypes.c_double),
    ]


class _CPose(ctypes.Structure):
    _fields_ = [
        ("x_enu_m", ctypes.c_double),
        ("y_enu_m", ctypes.c_double),
        ("yaw_rad", ctypes.c_double),
    ]


class _CPrimitive(ctypes.Structure):
    _fields_ = [
        ("point_offset", ctypes.c_int32),
        ("point_count", ctypes.c_int32),
        ("kind", ctypes.c_int32),
        ("channel", ctypes.c_int32),
        ("level", ctypes.c_int32),
        ("level_valid", ctypes.c_int32),
        ("width_m", ctypes.c_float),
        ("value", ctypes.c_float),
    ]


_NAV_LINE = 0
_NAV_POLYGON = 1
_NAV_POINT = 2
_NAV_DIRECTION_LINE = 3


def _library_candidates() -> list[Path]:
    configured = os.environ.get("AUTO_E2E_NAVIGATION_RASTERIZER_LIBRARY")
    if configured:
        return [Path(configured)]
    directory = Path(__file__).resolve().parent / "native"
    suffixes = [".dylib", ".so"] if platform.system() == "Darwin" else [".so"]
    return [
        directory / f"libnavigation_rasterizer{suffix}"
        for suffix in suffixes
    ]


def _load_library(path: str | Path | None = None) -> ctypes.CDLL:
    candidates = [Path(path)] if path is not None else _library_candidates()
    library_path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if library_path is None:
        raise NativeRasterizerUnavailable(
            "navigation rasterizer shared library is missing; run "
            "`python Model/navigation/native/build.py` during the image build"
        )
    library = ctypes.CDLL(str(library_path))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    float_pointer = ctypes.POINTER(ctypes.c_float)
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    library.nav_renderer_version.argtypes = []
    library.nav_renderer_version.restype = ctypes.c_char_p
    library.nav_render.argtypes = [
        double_pointer,
        ctypes.c_int32,
        ctypes.POINTER(_CPrimitive),
        ctypes.c_int32,
        double_pointer,
        ctypes.c_int32,
        int_pointer,
        ctypes.c_int32,
        double_pointer,
        ctypes.c_int32,
        _CPose,
        _CGeometry,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        float_pointer,
        byte_pointer,
    ]
    library.nav_render.restype = ctypes.c_int32
    library.nav_warp.argtypes = [
        float_pointer,
        byte_pointer,
        _CPose,
        _CPose,
        _CGeometry,
        float_pointer,
        byte_pointer,
    ]
    library.nav_warp.restype = ctypes.c_int32
    return library


def _c_geometry(geometry: NavigationRasterGeometry) -> _CGeometry:
    return _CGeometry(
        height_px=geometry.height_px,
        width_px=geometry.width_px,
        meters_per_pixel=geometry.meters_per_pixel,
        x_min_m=geometry.x_min_m,
        x_max_m=geometry.x_max_m,
        y_min_m=geometry.y_min_m,
        y_max_m=geometry.y_max_m,
    )


def _c_pose(pose: EgoPose) -> _CPose:
    return _CPose(
        x_enu_m=pose.x_enu_m,
        y_enu_m=pose.y_enu_m,
        yaw_rad=pose.yaw_rad,
    )


@dataclasses.dataclass(frozen=True)
class _PrimitiveSpec:
    points: np.ndarray
    kind: int
    channel: int
    width_m: float
    level: int | None
    value: float = 1.0


def _polygon_spec(
    primitive: PolygonPrimitive,
    channel: MapChannel,
) -> _PrimitiveSpec:
    return _PrimitiveSpec(
        points=primitive.points_enu_m[:, :2],
        kind=_NAV_POLYGON,
        channel=int(channel),
        width_m=0.0,
        level=primitive.level,
    )


def _line_spec(
    primitive: PolylinePrimitive,
    channel: MapChannel,
    *,
    width_m: float,
) -> _PrimitiveSpec:
    return _PrimitiveSpec(
        points=primitive.points_enu_m[:, :2],
        kind=_NAV_LINE,
        channel=int(channel),
        width_m=width_m,
        level=primitive.level,
    )


def _map_primitive_specs(
    navigation_map: NavigationMap,
    geometry: NavigationRasterGeometry,
) -> list[_PrimitiveSpec]:
    xmin, ymin, xmax, ymax = navigation_map.bounds_enu_m
    known_bounds = np.asarray(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=np.float64,
    )
    specs = [
        _PrimitiveSpec(
            points=known_bounds,
            kind=_NAV_POLYGON,
            channel=int(MapChannel.KNOWN_MAP_AREA),
            width_m=0.0,
            level=None,
        )
    ]
    specs.extend(
        _polygon_spec(primitive, MapChannel.DRIVABLE_AREA)
        for primitive in navigation_map.drivable_polygons
    )
    specs.extend(
        _line_spec(primitive, MapChannel.LANE_BOUNDARY, width_m=1.0)
        for primitive in navigation_map.lane_boundaries
    )
    specs.extend(
        _line_spec(primitive, MapChannel.LANE_CENTERLINE, width_m=1.0)
        for primitive in navigation_map.lane_centerlines
    )
    specs.extend(
        _polygon_spec(primitive, MapChannel.INTERSECTION)
        for primitive in navigation_map.intersection_polygons
    )
    specs.extend(
        _polygon_spec(primitive, MapChannel.CROSSWALK)
        for primitive in navigation_map.crosswalk_polygons
    )
    specs.extend(
        _line_spec(primitive, MapChannel.STOP_LINE, width_m=1.0)
        for primitive in navigation_map.stop_lines
    )
    specs.extend(
        _PrimitiveSpec(
            points=signal.position_enu_m[:2].reshape(1, 2),
            kind=_NAV_POINT,
            channel=int(MapChannel.STATIC_TRAFFIC_SIGNAL),
            width_m=2.0,
            level=signal.level,
        )
        for signal in navigation_map.static_traffic_signals
    )
    specs.extend(
        _PrimitiveSpec(
            points=field.centerline_enu_m[:, :2],
            kind=_NAV_DIRECTION_LINE,
            channel=int(MapChannel.TRAFFIC_DIRECTION_VALID),
            width_m=geometry.route_corridor_width_m,
            level=field.level,
        )
        for field in navigation_map.directed_lane_fields
    )
    return specs


def _pack_primitives(
    specs: list[_PrimitiveSpec],
) -> tuple[np.ndarray, Any]:
    points: list[np.ndarray] = []
    primitive_values: list[_CPrimitive] = []
    offset = 0
    for spec in specs:
        values = np.ascontiguousarray(spec.points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("native primitive points must have shape [N,2]")
        points.append(values)
        primitive_values.append(
            _CPrimitive(
                point_offset=offset,
                point_count=len(values),
                kind=spec.kind,
                channel=spec.channel,
                level=spec.level or 0,
                level_valid=int(spec.level is not None),
                width_m=spec.width_m,
                value=spec.value,
            )
        )
        offset += len(values)
    packed_points = (
        np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float64)
        if points
        else np.empty((0, 2), dtype=np.float64)
    )
    primitive_array_type = _CPrimitive * max(1, len(primitive_values))
    primitive_array = primitive_array_type(
        *(primitive_values or [_CPrimitive()])
    )
    return packed_points, primitive_array


def _pack_route(
    route: NavigationRoute | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    if route is None or not route.valid:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.asarray([0], dtype=np.int32),
            np.zeros(2, dtype=np.float64),
            None,
        )
    points = [
        segment.centerline_enu_m[:, :2]
        for segment in route.lane_sequence
    ]
    offsets = [0]
    for line in points:
        offsets.append(offsets[-1] + len(line))
    active_level = next(
        (
            segment.level
            for segment in route.lane_sequence
            if segment.level is not None
        ),
        None,
    )
    return (
        np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float64),
        np.asarray(offsets, dtype=np.int32),
        np.ascontiguousarray(route.destination.position_enu_m[:2], dtype=np.float64),
        active_level,
    )


class NativeNavigationRasterizer:
    def __init__(
        self,
        geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
        *,
        library_path: str | Path | None = None,
    ) -> None:
        self.geometry = geometry
        self._library = _load_library(library_path)
        version = self._library.nav_renderer_version()
        self.renderer_version = version.decode("ascii")
        self._hash_cache: dict[int, tuple[Any, str]] = {}

    def _object_hash(self, value: Any) -> str:
        key = id(value)
        cached = self._hash_cache.get(key)
        if cached is not None and cached[0] is value:
            return cached[1]
        digest = contract_sha256(value)
        self._hash_cache[key] = (value, digest)
        return digest

    def render(
        self,
        navigation_map: NavigationMap | None,
        route: NavigationRoute | None,
        pose: EgoPose,
    ) -> NavigationRaster:
        map_valid = navigation_map is not None
        route_valid = bool(
            map_valid
            and route is not None
            and route.valid
            and route.map_version == navigation_map.map_version
        )
        effective_route = route if route_valid else None
        specs = (
            _map_primitive_specs(navigation_map, self.geometry)
            if navigation_map is not None
            else []
        )
        map_points, primitive_array = _pack_primitives(specs)
        route_points, route_offsets, destination, active_level = _pack_route(
            effective_route
        )
        output_map = np.zeros(
            (
                MAP_CHANNEL_COUNT,
                self.geometry.height_px,
                self.geometry.width_px,
            ),
            dtype=np.float32,
        )
        output_route = np.zeros(
            (
                ROUTE_CHANNEL_COUNT,
                self.geometry.height_px,
                self.geometry.width_px,
            ),
            dtype=np.uint8,
        )
        status = self._library.nav_render(
            map_points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(map_points),
            primitive_array,
            len(specs),
            route_points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(route_points),
            route_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max(0, len(route_offsets) - 1),
            destination.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(route_valid),
            _c_pose(pose),
            _c_geometry(self.geometry),
            self.geometry.route_corridor_width_m,
            self.geometry.destination_marker_radius_m,
            self.geometry.route_rear_clip_m,
            active_level or 0,
            int(active_level is not None),
            output_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if status:
            raise RuntimeError(f"native navigation render failed: status={status}")

        vector_hashes = []
        if navigation_map is not None:
            vector_hashes.append(self._object_hash(navigation_map))
        if effective_route is not None:
            vector_hashes.append(self._object_hash(effective_route))
        combined_hash = (
            contract_sha256(vector_hashes)
            if vector_hashes
            else contract_sha256({"map_valid": False, "route_valid": False})
        )
        return NavigationRaster(
            map_context=output_map,
            route_mask=output_route,
            map_valid=map_valid,
            route_valid=route_valid,
            geometry_id=self.geometry.geometry_id,
            render_pose=pose,
            sample_pose=pose,
            renderer_version=self.renderer_version,
            map_version=(
                navigation_map.map_version if navigation_map is not None else ""
            ),
            route_id=(
                effective_route.route_id if effective_route is not None else ""
            ),
            route_revision=(
                effective_route.revision if effective_route is not None else 0
            ),
            route_confidence=(
                effective_route.confidence if effective_route is not None else 0.0
            ),
            input_vector_sha256=combined_hash,
        )

    def warp(
        self,
        raster: NavigationRaster,
        sample_pose: EgoPose,
    ) -> NavigationRaster:
        if raster.geometry_id != self.geometry.geometry_id:
            raise ValueError(
                "cannot warp a raster from a different geometry contract"
            )
        output_map = np.zeros_like(raster.map_context, dtype=np.float32)
        output_route = np.zeros_like(raster.route_mask, dtype=np.uint8)
        status = self._library.nav_warp(
            raster.map_context.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            raster.route_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            _c_pose(raster.render_pose),
            _c_pose(sample_pose),
            _c_geometry(self.geometry),
            output_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if status:
            raise RuntimeError(f"native navigation warp failed: status={status}")
        return NavigationRaster(
            map_context=output_map,
            route_mask=output_route,
            map_valid=raster.map_valid,
            route_valid=raster.route_valid,
            geometry_id=raster.geometry_id,
            render_pose=raster.render_pose,
            sample_pose=sample_pose,
            renderer_version=raster.renderer_version,
            map_version=raster.map_version,
            route_id=raster.route_id,
            route_revision=raster.route_revision,
            route_confidence=raster.route_confidence,
            input_vector_sha256=raster.input_vector_sha256,
        )
