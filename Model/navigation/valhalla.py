"""Vehicle-local Valhalla provider and OSM lane-sequence resolver."""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import math
import re
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .contracts import (
    Destination,
    Maneuver,
    MapFrame,
    NavigationRoute,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
    TransitionType,
)
from .osm_adapter import OSMLaneSegment, OSMMapAdapter


VALHALLA_PROVIDER_VERSION = "valhalla_route_provider_v1"
OSM_LANE_RESOLVER_VERSION = "osm_lane_sequence_resolver_v1"


@dataclasses.dataclass(frozen=True)
class GeoRoutePose:
    latitude_deg: float
    longitude_deg: float
    heading_deg_cw_from_north: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("route pose latitude is outside WGS84")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("route pose longitude is outside WGS84")
        if not math.isfinite(self.heading_deg_cw_from_north):
            raise ValueError("route pose heading must be finite")
        if self.timestamp_ns < 0:
            raise ValueError("route pose timestamp must be non-negative")


@dataclasses.dataclass(frozen=True)
class ProviderManeuver:
    begin_shape_index: int
    end_shape_index: int
    instruction: str
    maneuver_type: str
    begin_heading_deg: float | None = None
    end_heading_deg: float | None = None
    turn_lanes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.begin_shape_index < 0
            or self.end_shape_index < self.begin_shape_index
        ):
            raise ValueError("provider maneuver shape range is invalid")
        object.__setattr__(self, "turn_lanes", tuple(self.turn_lanes))


@dataclasses.dataclass(frozen=True)
class ProviderRoute:
    request_id: str
    map_version: str
    timestamp_ns: int
    shape_wgs84: np.ndarray
    maneuvers: tuple[ProviderManeuver, ...]
    provider: str = "valhalla"

    def __post_init__(self) -> None:
        if not self.request_id or not self.map_version or not self.provider:
            raise ValueError("provider route metadata must not be empty")
        if self.timestamp_ns < 0:
            raise ValueError("provider route timestamp must be non-negative")
        shape = np.asarray(self.shape_wgs84, dtype=np.float64)
        if shape.ndim != 2 or shape.shape[1] != 2 or len(shape) < 2:
            raise ValueError("provider route shape must be [N,2], N >= 2")
        if not np.isfinite(shape).all():
            raise ValueError("provider route shape contains non-finite values")
        if (
            np.any(shape[:, 0] < -90.0)
            or np.any(shape[:, 0] > 90.0)
            or np.any(shape[:, 1] < -180.0)
            or np.any(shape[:, 1] > 180.0)
        ):
            raise ValueError("provider route shape is outside WGS84")
        shape = np.ascontiguousarray(shape)
        shape.setflags(write=False)
        object.__setattr__(self, "shape_wgs84", shape)
        object.__setattr__(self, "maneuvers", tuple(self.maneuvers))


class RouteProvider(Protocol):
    def plan(
        self,
        current_pose: GeoRoutePose,
        destination: GeoRoutePose,
        map_version: str,
    ) -> ProviderRoute: ...


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def decode_polyline6(value: str) -> np.ndarray:
    """Decode a Valhalla encoded polyline with six decimal-place precision."""
    if not value:
        raise ValueError("Valhalla route shape is empty")
    coordinates: list[tuple[float, float]] = []
    index = 0
    latitude = 0
    longitude = 0
    while index < len(value):
        deltas = []
        for _ in range(2):
            result = 0
            shift = 0
            while True:
                if index >= len(value):
                    raise ValueError("truncated Valhalla encoded polyline")
                byte = ord(value[index]) - 63
                index += 1
                if byte < 0:
                    raise ValueError("invalid Valhalla encoded polyline")
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
                if shift > 60:
                    raise ValueError("invalid Valhalla polyline varint")
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        latitude += deltas[0]
        longitude += deltas[1]
        coordinates.append((latitude / 1e6, longitude / 1e6))
    return np.asarray(coordinates, dtype=np.float64)


def _default_transport(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = response.read()
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise ValueError("Valhalla response root must be an object")
    return decoded


def _require_local_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Valhalla endpoint must be a local HTTP URL")
    hostname = parsed.hostname
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("Valhalla endpoint must resolve to loopback")
    return endpoint.rstrip("/") + "/route"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


class ValhallaRouteProvider:
    """Request routes from a vehicle-local Valhalla service."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8002",
        *,
        timeout_seconds: float = 2.0,
        transport: Transport | None = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("Valhalla timeout must be positive")
        self.route_endpoint = _require_local_endpoint(endpoint)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or _default_transport

    def plan(
        self,
        current_pose: GeoRoutePose,
        destination: GeoRoutePose,
        map_version: str,
    ) -> ProviderRoute:
        if not map_version:
            raise ValueError("map_version must not be empty")
        request_identity = json.dumps(
            {
                "current": dataclasses.asdict(current_pose),
                "destination": dataclasses.asdict(destination),
                "map_version": map_version,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        request_id = "auto-e2e-" + hashlib.sha256(
            request_identity
        ).hexdigest()[:24]
        payload = {
            "id": request_id,
            "locations": [
                {
                    "lat": current_pose.latitude_deg,
                    "lon": current_pose.longitude_deg,
                    "heading": current_pose.heading_deg_cw_from_north,
                    "type": "break",
                },
                {
                    "lat": destination.latitude_deg,
                    "lon": destination.longitude_deg,
                    "type": "break",
                },
            ],
            "costing": "auto",
            "shape_format": "polyline6",
            "units": "kilometers",
            "directions_options": {
                "units": "kilometers",
            },
        }
        response = self.transport(
            self.route_endpoint,
            payload,
            self.timeout_seconds,
        )
        trip = response.get("trip")
        if not isinstance(trip, Mapping):
            raise ValueError("Valhalla response has no trip object")
        legs = trip.get("legs")
        if not isinstance(legs, Sequence) or not legs:
            raise ValueError("Valhalla response has no route legs")

        shape_parts: list[np.ndarray] = []
        maneuvers: list[ProviderManeuver] = []
        shape_offset = 0
        for leg in legs:
            if not isinstance(leg, Mapping):
                raise ValueError("Valhalla route leg must be an object")
            shape = decode_polyline6(str(leg.get("shape", "")))
            if shape_parts and np.array_equal(shape_parts[-1][-1], shape[0]):
                shape = shape[1:]
                leg_offset = shape_offset - 1
            else:
                leg_offset = shape_offset
            if len(shape):
                shape_parts.append(shape)
                shape_offset += len(shape)
            for value in leg.get("maneuvers", []):
                if not isinstance(value, Mapping):
                    raise ValueError("Valhalla maneuver must be an object")
                lanes = value.get("turn_lanes", ())
                if isinstance(lanes, str):
                    lanes = (lanes,)
                maneuvers.append(
                    ProviderManeuver(
                        begin_shape_index=(
                            int(value.get("begin_shape_index", 0))
                            + leg_offset
                        ),
                        end_shape_index=(
                            int(value.get("end_shape_index", 0))
                            + leg_offset
                        ),
                        instruction=str(value.get("instruction", "")),
                        maneuver_type=str(value.get("type", "unknown")),
                        begin_heading_deg=_optional_float(
                            value.get("begin_heading")
                        ),
                        end_heading_deg=_optional_float(
                            value.get("end_heading")
                        ),
                        turn_lanes=tuple(str(lane) for lane in lanes),
                    )
                )
        if not shape_parts:
            raise ValueError("Valhalla response has no decoded route shape")
        shape_wgs84 = np.concatenate(shape_parts, axis=0)
        if len(shape_wgs84) < 2:
            raise ValueError("Valhalla route shape has fewer than two points")
        response_id = response.get("id")
        if response_id not in (None, request_id):
            raise ValueError("Valhalla response ID differs from request ID")
        return ProviderRoute(
            request_id=request_id,
            map_version=map_version,
            timestamp_ns=current_pose.timestamp_ns,
            shape_wgs84=shape_wgs84,
            maneuvers=tuple(maneuvers),
        )


@dataclasses.dataclass(frozen=True)
class LaneSequenceResolverConfig:
    maximum_distance_m: float = 8.0
    maximum_heading_error_rad: float = math.radians(60.0)
    minimum_matched_ratio: float = 0.75
    maximum_p95_distance_m: float = 5.0
    maximum_p95_heading_error_rad: float = math.radians(45.0)
    maneuver_mismatch_cost: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_distance_m,
            self.maximum_heading_error_rad,
            self.maximum_p95_distance_m,
            self.maximum_p95_heading_error_rad,
            self.maneuver_mismatch_cost,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("lane resolver thresholds must be positive")
        if not 0.0 <= self.minimum_matched_ratio <= 1.0:
            raise ValueError("minimum_matched_ratio must be in [0,1]")

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                dataclasses.asdict(self),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> tuple[float, float]:
    best_distance = math.inf
    best_heading = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        delta = end[:2] - start[:2]
        length_sq = float(np.dot(delta, delta))
        if length_sq <= 1e-12:
            continue
        fraction = min(
            1.0,
            max(
                0.0,
                float(np.dot(point - start[:2], delta) / length_sq),
            ),
        )
        nearest = start[:2] + fraction * delta
        distance = float(np.linalg.norm(point - nearest))
        heading = math.atan2(float(delta[1]), float(delta[0]))
        if distance < best_distance:
            best_distance = distance
            best_heading = heading
    return best_distance, best_heading


def _shape_headings(points: np.ndarray) -> np.ndarray:
    delta = np.diff(points[:, :2], axis=0)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate([headings, headings[-1:]])


def _desired_maneuver(
    maneuvers: Sequence[ProviderManeuver],
    shape_index: int,
) -> Maneuver:
    for maneuver in maneuvers:
        if maneuver.begin_shape_index <= shape_index <= maneuver.end_shape_index:
            text = " ".join(
                [
                    maneuver.instruction.lower(),
                    maneuver.maneuver_type.lower(),
                    *[lane.lower() for lane in maneuver.turn_lanes],
                ]
            )
            if "u-turn" in text or "uturn" in text:
                return Maneuver.U_TURN
            if "left" in text:
                return Maneuver.LEFT
            if "right" in text:
                return Maneuver.RIGHT
            if "merge" in text:
                return Maneuver.MERGE
            if "exit" in text:
                return Maneuver.EXIT
            return Maneuver.STRAIGHT
    return Maneuver.UNKNOWN


def _wgs84_to_local_enu(
    points_wgs84: np.ndarray,
    frame: MapFrame,
) -> np.ndarray:
    match = re.search(r"EPSG:(\d+)", frame.projection)
    if match is None:
        raise ValueError(
            "OSM map frame projection must include an EPSG code"
        )
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{match.group(1)}",
        always_xy=True,
    )
    origin_x, origin_y = transformer.transform(
        frame.origin_longitude_deg,
        frame.origin_latitude_deg,
    )
    x, y = transformer.transform(
        points_wgs84[:, 1],
        points_wgs84[:, 0],
    )
    return np.column_stack(
        [
            np.asarray(x) - origin_x,
            np.asarray(y) - origin_y,
            np.zeros(len(points_wgs84)),
        ]
    )


class LocalLaneSequenceResolver:
    """Resolve provider shape intent against one cached directed lane graph."""

    def __init__(
        self,
        adapter: OSMMapAdapter,
        *,
        config: LaneSequenceResolverConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or LaneSequenceResolverConfig()

    def _shortest_successor_path(
        self,
        start: OSMLaneSegment,
        goal: OSMLaneSegment,
    ) -> list[OSMLaneSegment]:
        queue = deque([(start.lane_id, [start.lane_id])])
        visited = {start.lane_id}
        while queue:
            lane_id, path = queue.popleft()
            lane = self.adapter.lanes_by_id[lane_id]
            for successor in sorted(lane.successor_ids):
                if successor in visited:
                    continue
                next_path = [*path, successor]
                if successor == goal.lane_id:
                    return [
                        self.adapter.lanes_by_id[item]
                        for item in next_path
                    ]
                visited.add(successor)
                queue.append((successor, next_path))
        return []

    @staticmethod
    def _relation(
        previous: OSMLaneSegment,
        current: OSMLaneSegment,
    ) -> TransitionType | None:
        if current.lane_id == previous.lane_id:
            return TransitionType.FOLLOW
        if current.lane_id in previous.successor_ids:
            return TransitionType.FOLLOW
        if current.lane_id == previous.left_adjacent_id:
            return TransitionType.LEFT_ADJACENT
        if current.lane_id == previous.right_adjacent_id:
            return TransitionType.RIGHT_ADJACENT
        return None

    def resolve(
        self,
        provider_route: ProviderRoute,
        *,
        revision: int,
    ) -> NavigationRoute:
        if provider_route.map_version != self.adapter.map_version:
            raise ValueError("provider route and OSM map versions differ")
        if revision <= 0:
            raise ValueError("route revision must be positive")
        points = _wgs84_to_local_enu(
            provider_route.shape_wgs84,
            self.adapter.frame,
        )
        headings = _shape_headings(points)

        matched: list[tuple[int, OSMLaneSegment, float, float]] = []
        for index, (point, heading) in enumerate(zip(points, headings)):
            desired = _desired_maneuver(provider_route.maneuvers, index)
            options = []
            for lane in self.adapter.lane_segments:
                distance, lane_heading = _point_to_polyline(
                    point[:2],
                    lane.centerline_enu_m,
                )
                heading_error = abs(_wrap_angle(float(heading - lane_heading)))
                if (
                    distance > self.config.maximum_distance_m
                    or heading_error
                    > self.config.maximum_heading_error_rad
                ):
                    continue
                mismatch = (
                    desired not in (Maneuver.UNKNOWN, Maneuver.STRAIGHT)
                    and lane.maneuver not in (Maneuver.UNKNOWN, desired)
                )
                score = (
                    distance
                    + heading_error
                    + self.config.maneuver_mismatch_cost * int(mismatch)
                    + (1.0 - lane.confidence)
                )
                options.append(
                    (score, lane.lane_id, lane, distance, heading_error)
                )
            if options:
                _, _, lane, distance, heading_error = min(options)
                matched.append(
                    (index, lane, distance, heading_error)
                )

        deduplicated: list[OSMLaneSegment] = []
        for _, lane, _, _ in matched:
            if not deduplicated or deduplicated[-1].lane_id != lane.lane_id:
                deduplicated.append(lane)
        resolved: list[tuple[OSMLaneSegment, TransitionType]] = []
        fill_count = 0
        fill_length = 0.0
        adjacent_count = 0
        unresolved = 0
        if deduplicated:
            resolved.append((deduplicated[0], TransitionType.FOLLOW))
        for lane in deduplicated[1:]:
            previous = resolved[-1][0]
            relation = self._relation(previous, lane)
            if relation is not None:
                resolved.append((lane, relation))
                if relation in (
                    TransitionType.LEFT_ADJACENT,
                    TransitionType.RIGHT_ADJACENT,
                ):
                    adjacent_count += 1
                continue
            path = self._shortest_successor_path(previous, lane)
            if path:
                fill_count += 1
                for filled in path[1:]:
                    resolved.append((filled, TransitionType.FOLLOW))
                    fill_length += float(
                        np.linalg.norm(
                            np.diff(
                                filled.centerline_enu_m[:, :2],
                                axis=0,
                            ),
                            axis=1,
                        ).sum()
                    )
            else:
                unresolved += 1
                resolved.append((lane, TransitionType.FOLLOW))

        distances = np.asarray(
            [item[2] for item in matched],
            dtype=np.float64,
        )
        heading_errors = np.asarray(
            [item[3] for item in matched],
            dtype=np.float64,
        )
        matched_ratio = len(matched) / len(points)
        median_distance = (
            float(np.median(distances)) if len(distances) else 0.0
        )
        p95_distance = (
            float(np.quantile(distances, 0.95)) if len(distances) else 0.0
        )
        median_heading = (
            float(np.median(heading_errors))
            if len(heading_errors)
            else 0.0
        )
        p95_heading = (
            float(np.quantile(heading_errors, 0.95))
            if len(heading_errors)
            else 0.0
        )
        failures = []
        if not resolved:
            failures.append("no_lane_sequence")
        if matched_ratio < self.config.minimum_matched_ratio:
            failures.append("matched_ratio_below_threshold")
        if p95_distance > self.config.maximum_p95_distance_m:
            failures.append("p95_distance_above_threshold")
        if p95_heading > self.config.maximum_p95_heading_error_rad:
            failures.append("p95_heading_above_threshold")
        if unresolved:
            failures.append("unresolved_discontinuity")

        trace_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                provider_route.shape_wgs84,
                dtype="<f8",
            ).tobytes()
        ).hexdigest()
        quality = RouteQuality(
            matched_pose_ratio=matched_ratio,
            median_lateral_distance_m=median_distance,
            p95_lateral_distance_m=p95_distance,
            median_heading_error_rad=median_heading,
            p95_heading_error_rad=p95_heading,
            shortest_path_fill_count=fill_count,
            shortest_path_fill_length_m=fill_length,
            adjacent_transition_count=adjacent_count,
            unresolved_discontinuities=unresolved,
            failure_reasons=tuple(failures),
        )
        lane_confidence = (
            float(np.mean([lane.confidence for lane, _ in resolved]))
            if resolved
            else 0.0
        )
        confidence = matched_ratio * lane_confidence
        confidence *= math.exp(-p95_distance / 10.0)
        confidence *= math.exp(-p95_heading / math.pi)
        confidence *= math.exp(-float(unresolved))
        confidence = min(1.0, max(0.0, confidence))
        segments = tuple(
            RouteLaneSegment(
                lane_id=lane.lane_id,
                provider_segment_id=lane.provider_segment_id,
                centerline_enu_m=lane.centerline_enu_m,
                left_boundary_enu_m=lane.left_boundary_enu_m,
                right_boundary_enu_m=lane.right_boundary_enu_m,
                level=lane.level,
                transition_from_previous=transition,
                maneuver=(
                    Maneuver.DESTINATION
                    if index == len(resolved) - 1
                    else lane.maneuver
                ),
                confidence=lane.confidence,
            )
            for index, (lane, transition) in enumerate(resolved)
        )
        route_hash = hashlib.sha256(
            (
                provider_route.request_id
                + self.adapter.map_version
                + self.config.sha256()
            ).encode("utf-8")
        ).hexdigest()[:24]
        return NavigationRoute(
            route_id=f"valhalla:{route_hash}",
            revision=revision,
            provider="valhalla_osm",
            timestamp_ns=provider_route.timestamp_ns,
            valid_from_ns=provider_route.timestamp_ns,
            map_version=self.adapter.map_version,
            frame=self.adapter.frame,
            lane_sequence=segments,
            destination=Destination(
                position_enu_m=points[-1],
                source="user_destination",
            ),
            confidence=confidence,
            valid=not failures,
            quality=quality,
            estimated_destination=False,
            provenance=RouteProvenance(
                source_revision=provider_route.request_id,
                matcher_version=OSM_LANE_RESOLVER_VERSION,
                matcher_config_sha256=self.config.sha256(),
                map_sha256=self.adapter.source_sha256,
                trace_sha256=trace_sha256,
            ),
        )
