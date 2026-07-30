"""Offline OSM lane-graph adapter for the canonical navigation contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    DirectedLaneField,
    Maneuver,
    MapFrame,
    NavigationMap,
    PolygonPrimitive,
    PolylinePrimitive,
    StaticTrafficSignal,
    _points,
)


OSM_ADAPTER_VERSION = "osm_lane_graph_adapter_v2"
OSM_LANE_GRAPH_SCHEMA_VERSION = "osm_lane_graph_v2"
SUPPORTED_OSM_LANE_GRAPH_SCHEMA_VERSIONS = frozenset(
    {"osm_lane_graph_v1", OSM_LANE_GRAPH_SCHEMA_VERSION}
)


def _provider_segment_id(value: Mapping[str, Any]) -> str:
    direction = str(value["direction"])
    if direction not in {"forward", "reverse"}:
        raise ValueError("OSM lane direction must be forward or reverse")
    lane_index = int(value["lane_index"])
    if lane_index < 0:
        raise ValueError("OSM lane_index must be non-negative")
    return f"{int(value['way_id'])}:{direction}:{lane_index}"


def _canonical_lane_id(map_version: str, provider_segment_id: str) -> str:
    return f"osm:{map_version}:{provider_segment_id}"


@dataclasses.dataclass(frozen=True)
class OSMLaneSegment:
    lane_id: str
    provider_segment_id: str
    centerline_enu_m: np.ndarray
    left_boundary_enu_m: np.ndarray | None
    right_boundary_enu_m: np.ndarray | None
    successor_ids: tuple[str, ...]
    left_adjacent_id: str | None
    right_adjacent_id: str | None
    maneuver: Maneuver
    level: int | None
    confidence: float
    predecessor_ids: tuple[str, ...] = ()
    road_class: str | None = None
    lane_subtype: str | None = None
    one_way: bool | None = None
    carriageway_id: str | None = None
    median_separated: bool | None = None
    barrier_separated: bool | None = None
    is_intersection: bool = False
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
        if not self.lane_id or not self.provider_segment_id:
            raise ValueError("OSM lane identifiers must not be empty")
        object.__setattr__(
            self,
            "centerline_enu_m",
            _points(
                self.centerline_enu_m,
                name=f"OSM lane {self.lane_id}",
                minimum=2,
            ),
        )
        for name in ("left_boundary_enu_m", "right_boundary_enu_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _points(
                        value,
                        name=f"OSM {name} {self.lane_id}",
                        minimum=2,
                    ),
                )
        object.__setattr__(self, "successor_ids", tuple(self.successor_ids))
        object.__setattr__(self, "predecessor_ids", tuple(self.predecessor_ids))
        for field_name in (
            "provider_attributes",
            "left_boundary_attributes",
            "right_boundary_attributes",
        ):
            object.__setattr__(
                self,
                field_name,
                dict(sorted(getattr(self, field_name).items())),
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OSM lane confidence must be in [0,1]")


def _optional_points(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> np.ndarray | None:
    if value is None:
        return None
    return _points(value, name=name, minimum=minimum)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key) and str(item)
    }


def _first_value(
    value: Mapping[str, Any],
    attributes: Mapping[str, str],
    *names: str,
) -> str | None:
    for name in names:
        candidate = value.get(name, attributes.get(name))
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def _polyline_primitives(
    values: Sequence[Mapping[str, Any]],
    prefix: str,
) -> tuple[PolylinePrimitive, ...]:
    return tuple(
        PolylinePrimitive(
            primitive_id=f"osm:{prefix}:{value['id']}",
            points_enu_m=value["points_enu_m"],
            level=(
                int(value["level"])
                if value.get("level") is not None
                else None
            ),
        )
        for value in sorted(values, key=lambda item: str(item["id"]))
    )


def _polygon_primitives(
    values: Sequence[Mapping[str, Any]],
    prefix: str,
) -> tuple[PolygonPrimitive, ...]:
    return tuple(
        PolygonPrimitive(
            primitive_id=f"osm:{prefix}:{value['id']}",
            points_enu_m=value["points_enu_m"],
            level=(
                int(value["level"])
                if value.get("level") is not None
                else None
            ),
        )
        for value in sorted(values, key=lambda item: str(item["id"]))
    )


class OSMMapAdapter:
    """Adapt a prebuilt local OSM lane graph without runtime network access."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        source_sha256: str,
    ) -> None:
        if (
            payload.get("schema_version")
            not in SUPPORTED_OSM_LANE_GRAPH_SCHEMA_VERSIONS
        ):
            raise ValueError("unsupported OSM lane-graph schema")
        if not source_sha256:
            raise ValueError("source_sha256 must not be empty")
        self.payload = dict(payload)
        self.source_sha256 = source_sha256
        self.map_version = str(payload["map_version"])
        if not self.map_version:
            raise ValueError("OSM map_version must not be empty")
        frame = payload["frame"]
        self.frame = MapFrame(
            frame_id=str(frame["frame_id"]),
            origin_latitude_deg=float(frame["origin_latitude_deg"]),
            origin_longitude_deg=float(frame["origin_longitude_deg"]),
            projection=str(frame["projection"]),
        )
        self.lane_segments = self._parse_lanes(payload["lanes"])
        self.lanes_by_id = {
            lane.lane_id: lane for lane in self.lane_segments
        }
        self.navigation_map = self._build_navigation_map()

    @classmethod
    def from_file(cls, path: str | Path) -> "OSMMapAdapter":
        """Load one deterministic regional lane graph from local storage."""
        source = Path(path)
        data = source.read_bytes()
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError("OSM lane graph is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("OSM lane graph root must be an object")
        return cls(
            payload,
            source_sha256=hashlib.sha256(data).hexdigest(),
        )

    def _parse_lanes(
        self,
        values: Sequence[Mapping[str, Any]],
    ) -> tuple[OSMLaneSegment, ...]:
        if not values:
            raise ValueError("OSM lane graph contains no lanes")
        provider_ids = [_provider_segment_id(value) for value in values]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("OSM lane graph has duplicate directed lanes")
        canonical = {
            provider_id: _canonical_lane_id(
                self.map_version,
                provider_id,
            )
            for provider_id in provider_ids
        }

        def resolve(reference: Any) -> str | None:
            if reference is None:
                return None
            provider_id = str(reference)
            if provider_id not in canonical:
                raise ValueError(
                    f"OSM lane reference {provider_id!r} is missing"
                )
            return canonical[provider_id]

        def resolve_required(reference: Any) -> str:
            resolved = resolve(reference)
            if resolved is None:
                raise ValueError("OSM successor reference must not be null")
            return resolved

        lanes = []
        for value, provider_id in sorted(
            zip(values, provider_ids),
            key=lambda pair: pair[1],
        ):
            try:
                maneuver = Maneuver(str(value.get("maneuver", "unknown")))
            except ValueError as error:
                raise ValueError(
                    f"unsupported OSM lane maneuver {value.get('maneuver')!r}"
                ) from error
            attributes = _string_mapping(value.get("tags", {}))
            one_way_value = value.get(
                "one_way",
                value.get("oneway", attributes.get("oneway")),
            )
            lanes.append(
                OSMLaneSegment(
                    lane_id=canonical[provider_id],
                    provider_segment_id=provider_id,
                    centerline_enu_m=value["centerline_enu_m"],
                    left_boundary_enu_m=_optional_points(
                        value.get("left_boundary_enu_m"),
                        name=f"OSM left boundary {provider_id}",
                        minimum=2,
                    ),
                    right_boundary_enu_m=_optional_points(
                        value.get("right_boundary_enu_m"),
                        name=f"OSM right boundary {provider_id}",
                        minimum=2,
                    ),
                    successor_ids=tuple(
                        resolve_required(reference)
                        for reference in value.get("successors", [])
                    ),
                    left_adjacent_id=resolve(
                        value.get("left_adjacent")
                    ),
                    right_adjacent_id=resolve(
                        value.get("right_adjacent")
                    ),
                    maneuver=maneuver,
                    level=(
                        int(value["level"])
                        if value.get("level") is not None
                        else None
                    ),
                    confidence=float(value.get("confidence", 1.0)),
                    road_class=_first_value(
                        value,
                        attributes,
                        "road_class",
                        "highway",
                    ),
                    lane_subtype=_first_value(
                        value,
                        attributes,
                        "lane_subtype",
                        "lane_type",
                        "subtype",
                    ),
                    one_way=_optional_bool(one_way_value),
                    carriageway_id=_first_value(
                        value,
                        attributes,
                        "carriageway_id",
                        "carriageway",
                    ),
                    median_separated=_optional_bool(
                        value.get(
                            "median_separated",
                            attributes.get("median"),
                        )
                    ),
                    barrier_separated=_optional_bool(
                        value.get(
                            "barrier_separated",
                            attributes.get("barrier"),
                        )
                    ),
                    is_intersection=bool(value.get("is_intersection", False)),
                    provider_attributes=attributes,
                    left_boundary_attributes=_string_mapping(
                        value.get("left_boundary_attributes", {})
                    ),
                    right_boundary_attributes=_string_mapping(
                        value.get("right_boundary_attributes", {})
                    ),
                )
            )
        predecessors: dict[str, list[str]] = {
            lane.lane_id: [] for lane in lanes
        }
        for lane in lanes:
            for successor_id in lane.successor_ids:
                predecessors[successor_id].append(lane.lane_id)
        return tuple(
            dataclasses.replace(
                lane,
                predecessor_ids=tuple(sorted(predecessors[lane.lane_id])),
            )
            for lane in lanes
        )

    def _build_navigation_map(self) -> NavigationMap:
        semantic = self.payload.get("semantic", {})
        lane_boundaries: dict[str, PolylinePrimitive] = {}
        lane_centerlines = []
        drivable_polygons = []
        directed_fields = []
        for lane in self.lane_segments:
            lane_centerlines.append(
                PolylinePrimitive(
                    f"{lane.lane_id}:centerline",
                    lane.centerline_enu_m,
                    lane.level,
                )
            )
            directed_fields.append(
                DirectedLaneField(
                    lane_id=lane.lane_id,
                    centerline_enu_m=lane.centerline_enu_m,
                    level=lane.level,
                    road_class=lane.road_class,
                    lane_subtype=lane.lane_subtype,
                    one_way=lane.one_way,
                    carriageway_id=lane.carriageway_id,
                    median_separated=lane.median_separated,
                    barrier_separated=lane.barrier_separated,
                    successor_lane_ids=lane.successor_ids,
                    predecessor_lane_ids=lane.predecessor_ids,
                    left_adjacent_lane_id=lane.left_adjacent_id,
                    right_adjacent_lane_id=lane.right_adjacent_id,
                    is_intersection=lane.is_intersection,
                    turn_direction=(
                        lane.maneuver.value
                        if lane.maneuver
                        in {
                            Maneuver.LEFT,
                            Maneuver.RIGHT,
                            Maneuver.U_TURN,
                        }
                        else None
                    ),
                    provider_attributes=lane.provider_attributes,
                    left_boundary_attributes=lane.left_boundary_attributes,
                    right_boundary_attributes=lane.right_boundary_attributes,
                )
            )
            for side, boundary in (
                ("left", lane.left_boundary_enu_m),
                ("right", lane.right_boundary_enu_m),
            ):
                if boundary is None:
                    continue
                digest = hashlib.sha256(
                    np.ascontiguousarray(boundary, dtype="<f8").tobytes()
                ).hexdigest()[:16]
                lane_boundaries.setdefault(
                    digest,
                    PolylinePrimitive(
                        f"osm:boundary:{digest}:{side}",
                        boundary,
                        lane.level,
                    ),
                )
            if (
                lane.left_boundary_enu_m is not None
                and lane.right_boundary_enu_m is not None
            ):
                polygon = np.concatenate(
                    [
                        lane.left_boundary_enu_m,
                        lane.right_boundary_enu_m[::-1],
                    ],
                    axis=0,
                )
                drivable_polygons.append(
                    PolygonPrimitive(
                        f"{lane.lane_id}:drivable",
                        polygon,
                        lane.level,
                    )
                )

        intersections = _polygon_primitives(
            semantic.get("intersections", []),
            "intersection",
        )
        crosswalks = _polygon_primitives(
            semantic.get("crosswalks", []),
            "crosswalk",
        )
        stop_lines = _polyline_primitives(
            semantic.get("stop_lines", []),
            "stop_line",
        )
        signals = tuple(
            StaticTrafficSignal(
                signal_id=f"osm:signal:{value['id']}",
                position_enu_m=value["position_enu_m"],
                level=(
                    int(value["level"])
                    if value.get("level") is not None
                    else None
                ),
            )
            for value in sorted(
                semantic.get("traffic_signals", []),
                key=lambda item: str(item["id"]),
            )
        )

        all_points = [
            primitive.points_enu_m
            for collection in (
                drivable_polygons,
                lane_boundaries.values(),
                lane_centerlines,
                intersections,
                crosswalks,
                stop_lines,
            )
            for primitive in collection
        ]
        all_points.extend(
            signal.position_enu_m.reshape(1, 3) for signal in signals
        )
        if not all_points:
            raise ValueError("OSM lane graph has no renderable geometry")
        points = np.concatenate(all_points, axis=0)
        x_min = float(points[:, 0].min())
        y_min = float(points[:, 1].min())
        x_max = float(points[:, 0].max())
        y_max = float(points[:, 1].max())
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        levels_present = any(
            lane.level is not None for lane in self.lane_segments
        )
        return NavigationMap(
            map_version=self.map_version,
            provider="osm",
            frame=self.frame,
            bounds_enu_m=(
                x_min,
                y_min,
                x_max,
                y_max,
            ),
            drivable_polygons=tuple(drivable_polygons),
            lane_boundaries=tuple(
                lane_boundaries[key] for key in sorted(lane_boundaries)
            ),
            lane_centerlines=tuple(lane_centerlines),
            intersection_polygons=intersections,
            crosswalk_polygons=crosswalks,
            stop_lines=stop_lines,
            static_traffic_signals=signals,
            directed_lane_fields=tuple(directed_fields),
            layer_availability={
                "crosswalk": bool(crosswalks),
                "drivable_area": bool(drivable_polygons),
                "intersection": bool(intersections),
                "lane_boundary": bool(lane_boundaries),
                "lane_centerline": bool(lane_centerlines),
                "road_level": levels_present,
                "static_traffic_signal": bool(signals),
                "stop_line": bool(stop_lines),
                "traffic_direction": bool(directed_fields),
                "lane_semantics": any(
                    lane.road_class is not None
                    or lane.one_way is not None
                    or lane.carriageway_id is not None
                    for lane in directed_fields
                ),
                "lane_topology": (
                    bool(directed_fields)
                    and self.payload.get("schema_version")
                    == OSM_LANE_GRAPH_SCHEMA_VERSION
                ),
            },
            provenance={
                "adapter_version": OSM_ADAPTER_VERSION,
                "source_sha256": self.source_sha256,
            },
        )

    def extract(self) -> NavigationMap:
        return self.navigation_map
