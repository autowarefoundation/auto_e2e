"""Provider-independent vector contracts for navigation input."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


NAVIGATION_SCHEMA_VERSION = "v2"


class TransitionType(str, enum.Enum):
    FOLLOW = "follow"
    LEFT_ADJACENT = "left_adjacent"
    RIGHT_ADJACENT = "right_adjacent"
    MERGE = "merge"
    SPLIT = "split"


class Maneuver(str, enum.Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    U_TURN = "u_turn"
    MERGE = "merge"
    EXIT = "exit"
    DESTINATION = "destination"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class MapFrame:
    """Metric ENU frame used by canonical map and route vectors."""

    frame_id: str
    origin_latitude_deg: float
    origin_longitude_deg: float
    projection: str

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if not -90.0 <= self.origin_latitude_deg <= 90.0:
            raise ValueError("origin latitude is outside WGS84 bounds")
        if not -180.0 <= self.origin_longitude_deg <= 180.0:
            raise ValueError("origin longitude is outside WGS84 bounds")
        if not self.projection:
            raise ValueError("projection must not be empty")


def _points(
    value: Any,
    *,
    name: str,
    minimum: int,
    dimensions: int = 3,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, dimensions):
        raise ValueError(
            f"{name} must have shape [N,2] or [N,{dimensions}], "
            f"got {array.shape}"
        )
    if array.shape[0] < minimum:
        raise ValueError(f"{name} requires at least {minimum} points")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if array.shape[1] == 2 and dimensions == 3:
        array = np.column_stack(
            [array, np.zeros(array.shape[0], dtype=np.float64)]
        )
    array = np.ascontiguousarray(array, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclasses.dataclass(frozen=True)
class PolylinePrimitive:
    primitive_id: str
    points_enu_m: np.ndarray
    level: int | None = None

    def __post_init__(self) -> None:
        if not self.primitive_id:
            raise ValueError("primitive_id must not be empty")
        object.__setattr__(
            self,
            "points_enu_m",
            _points(
                self.points_enu_m,
                name=f"polyline {self.primitive_id}",
                minimum=2,
            ),
        )


@dataclasses.dataclass(frozen=True)
class PolygonPrimitive:
    primitive_id: str
    points_enu_m: np.ndarray
    level: int | None = None

    def __post_init__(self) -> None:
        if not self.primitive_id:
            raise ValueError("primitive_id must not be empty")
        object.__setattr__(
            self,
            "points_enu_m",
            _points(
                self.points_enu_m,
                name=f"polygon {self.primitive_id}",
                minimum=3,
            ),
        )


@dataclasses.dataclass(frozen=True)
class DirectedLaneField:
    lane_id: str
    centerline_enu_m: np.ndarray
    level: int | None = None
    road_class: str | None = None
    lane_subtype: str | None = None
    one_way: bool | None = None
    carriageway_id: str | None = None
    median_separated: bool | None = None
    barrier_separated: bool | None = None
    successor_lane_ids: tuple[str, ...] = ()
    predecessor_lane_ids: tuple[str, ...] = ()
    left_adjacent_lane_id: str | None = None
    right_adjacent_lane_id: str | None = None
    is_intersection: bool = False
    turn_direction: str | None = None
    provider_attributes: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )
    left_boundary_attributes: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )
    right_boundary_attributes: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id must not be empty")
        object.__setattr__(
            self,
            "centerline_enu_m",
            _points(
                self.centerline_enu_m,
                name=f"directed lane {self.lane_id}",
                minimum=2,
            ),
        )
        for field_name in (
            "road_class",
            "lane_subtype",
            "carriageway_id",
            "left_adjacent_lane_id",
            "right_adjacent_lane_id",
            "turn_direction",
        ):
            value = getattr(self, field_name)
            if value is not None:
                normalized = str(value).strip()
                object.__setattr__(
                    self,
                    field_name,
                    normalized if normalized else None,
                )
        for field_name in ("successor_lane_ids", "predecessor_lane_ids"):
            values = tuple(str(value) for value in getattr(self, field_name))
            if any(not value for value in values) or len(values) != len(
                set(values)
            ):
                raise ValueError(f"{field_name} must contain unique lane IDs")
            object.__setattr__(self, field_name, values)
        for field_name in (
            "provider_attributes",
            "left_boundary_attributes",
            "right_boundary_attributes",
        ):
            attributes = {
                str(key): str(value)
                for key, value in getattr(self, field_name).items()
            }
            object.__setattr__(
                self,
                field_name,
                dict(sorted(attributes.items())),
            )


@dataclasses.dataclass(frozen=True)
class StaticTrafficSignal:
    signal_id: str
    position_enu_m: np.ndarray
    level: int | None = None

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must not be empty")
        point = np.asarray(self.position_enu_m, dtype=np.float64)
        if point.shape not in ((2,), (3,)):
            raise ValueError(
                "traffic signal position must have shape [2] or [3]"
            )
        if not np.isfinite(point).all():
            raise ValueError("traffic signal position contains non-finite values")
        if point.shape == (2,):
            point = np.append(point, 0.0)
        point = np.ascontiguousarray(point, dtype=np.float64)
        point.setflags(write=False)
        object.__setattr__(self, "position_enu_m", point)


@dataclasses.dataclass(frozen=True)
class RouteLaneSegment:
    lane_id: str
    provider_segment_id: str
    centerline_enu_m: np.ndarray
    left_boundary_enu_m: np.ndarray | None = None
    right_boundary_enu_m: np.ndarray | None = None
    level: int | None = None
    transition_from_previous: TransitionType = TransitionType.FOLLOW
    maneuver: Maneuver = Maneuver.UNKNOWN
    confidence: float = 1.0
    connected_from_previous: bool = True

    def __post_init__(self) -> None:
        if not self.lane_id or not self.provider_segment_id:
            raise ValueError("route lane identifiers must not be empty")
        object.__setattr__(
            self,
            "centerline_enu_m",
            _points(
                self.centerline_enu_m,
                name=f"route centerline {self.lane_id}",
                minimum=2,
            ),
        )
        for field_name in ("left_boundary_enu_m", "right_boundary_enu_m"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _points(
                        value,
                        name=f"{field_name} {self.lane_id}",
                        minimum=2,
                    ),
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("route lane confidence must be in [0,1]")


@dataclasses.dataclass(frozen=True)
class Destination:
    position_enu_m: np.ndarray
    source: str

    def __post_init__(self) -> None:
        point = np.asarray(self.position_enu_m, dtype=np.float64)
        if point.shape not in ((2,), (3,)):
            raise ValueError("destination must have shape [2] or [3]")
        if not np.isfinite(point).all():
            raise ValueError("destination contains non-finite values")
        if point.shape == (2,):
            point = np.append(point, 0.0)
        point = np.ascontiguousarray(point, dtype=np.float64)
        point.setflags(write=False)
        object.__setattr__(self, "position_enu_m", point)
        if not self.source:
            raise ValueError("destination source must not be empty")


@dataclasses.dataclass(frozen=True)
class RouteQuality:
    matched_pose_ratio: float
    median_lateral_distance_m: float
    p95_lateral_distance_m: float
    median_heading_error_rad: float
    p95_heading_error_rad: float
    shortest_path_fill_count: int = 0
    shortest_path_fill_length_m: float = 0.0
    adjacent_transition_count: int = 0
    unresolved_discontinuities: int = 0
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.matched_pose_ratio <= 1.0:
            raise ValueError("matched_pose_ratio must be in [0,1]")
        numeric = (
            self.median_lateral_distance_m,
            self.p95_lateral_distance_m,
            self.median_heading_error_rad,
            self.p95_heading_error_rad,
            self.shortest_path_fill_length_m,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("route quality distances and errors must be finite")
        counts = (
            self.shortest_path_fill_count,
            self.adjacent_transition_count,
            self.unresolved_discontinuities,
        )
        if any(value < 0 for value in counts):
            raise ValueError("route quality counts must be non-negative")


@dataclasses.dataclass(frozen=True)
class RouteProvenance:
    source_revision: str
    matcher_version: str
    matcher_config_sha256: str
    map_sha256: str
    trace_sha256: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if not getattr(self, field.name):
                raise ValueError(f"{field.name} must not be empty")


@dataclasses.dataclass(frozen=True)
class NavigationRoute:
    route_id: str
    revision: int
    provider: str
    timestamp_ns: int
    valid_from_ns: int
    map_version: str
    frame: MapFrame
    lane_sequence: tuple[RouteLaneSegment, ...]
    destination: Destination
    confidence: float
    valid: bool
    quality: RouteQuality
    estimated_destination: bool
    provenance: RouteProvenance
    schema_version: str = NAVIGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = (
            self.route_id,
            self.provider,
            self.map_version,
            self.schema_version,
        )
        if any(not value for value in required):
            raise ValueError("route metadata strings must not be empty")
        if self.revision < 0:
            raise ValueError("route revision must be non-negative")
        if self.timestamp_ns < 0 or self.valid_from_ns < 0:
            raise ValueError("route timestamps must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("route confidence must be in [0,1]")
        object.__setattr__(self, "lane_sequence", tuple(self.lane_sequence))
        if self.valid and not self.lane_sequence:
            raise ValueError("a valid route requires at least one lane segment")


@dataclasses.dataclass(frozen=True)
class NavigationMap:
    map_version: str
    provider: str
    frame: MapFrame
    bounds_enu_m: tuple[float, float, float, float]
    drivable_polygons: tuple[PolygonPrimitive, ...] = ()
    lane_boundaries: tuple[PolylinePrimitive, ...] = ()
    lane_centerlines: tuple[PolylinePrimitive, ...] = ()
    intersection_polygons: tuple[PolygonPrimitive, ...] = ()
    crosswalk_polygons: tuple[PolygonPrimitive, ...] = ()
    stop_lines: tuple[PolylinePrimitive, ...] = ()
    static_traffic_signals: tuple[StaticTrafficSignal, ...] = ()
    directed_lane_fields: tuple[DirectedLaneField, ...] = ()
    layer_availability: Mapping[str, bool] = dataclasses.field(
        default_factory=dict
    )
    provenance: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: str = NAVIGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.map_version or not self.provider or not self.schema_version:
            raise ValueError("map metadata strings must not be empty")
        bounds = tuple(float(value) for value in self.bounds_enu_m)
        if (
            len(bounds) != 4
            or not all(math.isfinite(value) for value in bounds)
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
        ):
            raise ValueError("bounds_enu_m must be (xmin,ymin,xmax,ymax)")
        object.__setattr__(self, "bounds_enu_m", bounds)
        sequence_fields = (
            "drivable_polygons",
            "lane_boundaries",
            "lane_centerlines",
            "intersection_polygons",
            "crosswalk_polygons",
            "stop_lines",
            "static_traffic_signals",
            "directed_lane_fields",
        )
        for field_name in sequence_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "layer_availability",
            dict(sorted(self.layer_availability.items())),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a contract object deterministically for hashing/artifacts."""
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def contract_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
