"""Deterministic map/route and GNSS/INS scene labelers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from navigation.contracts import (
    DirectedLaneField,
    Maneuver,
    RouteLaneSegment,
    TransitionType,
)
from navigation.geodesy import wgs84_to_map_xy

from .published_snapshot import CanonicalSceneEvidence
from .schema import LabelObservation, make_observation


MAP_ROUTE_LABELER_VERSION = "odd_deterministic_v3"
KINEMATICS_LABELER_VERSION = "odd_deterministic_kinematics_v3"
KINEMATICS_POLICY_VERSION = "odd_gnss_ins_kinematics_v2"
INTERVAL_NS = 1_000_000_000
STATIONARY_EPSILON_KPH = 0.5
STATIONARY_DWELL_NS = 1_000_000_000
STRONG_RESPONSE_DWELL_NS = 500_000_000
MAX_GAP_FLOOR_NS = 500_000_000
MAX_GAP_PERIOD_MULTIPLIER = 3
KINEMATIC_KEYS = (
    "odd.ego.speed_bin",
    "event.ego.motion_state",
    "event.ego.maneuver",
    "event.ego.strong_response",
)
MAP_ROUTE_KEYS = (
    "odd.road.context",
    "odd.road.type",
    "odd.road.division",
    "odd.road.horizontal_geometry",
    "odd.road.vertical_geometry",
    "odd.road.junction_type",
    "odd.road.junction_position",
    "odd.route.action",
    "odd.road.lane_count_bin",
    "odd.road.directionality",
    "odd.road.lane_type_present",
    "odd.road.special_structure",
    "odd.traffic_control.present",
    "odd.road.junction_control",
)
LOCAL_MAP_MAX_DISTANCE_M = 8.0
LOCAL_ROUTE_MAX_DISTANCE_M = 10.0
LOCAL_MATCH_MAX_HEADING_ERROR_RAD = math.radians(75.0)
LOCAL_ROUTE_MIN_SEGMENT_CONFIDENCE = 0.5


def _local_xy(path: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
    radius = 6_371_008.8
    lat = np.radians(path[:, 0])
    lon = np.radians(path[:, 1])
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    east = radius * (lon - lon0) * math.cos(lat0)
    north = radius * (lat - lat0)
    return np.column_stack([east, north])


def _smoothed(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) < width:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _interval_slices(timestamps: np.ndarray) -> list[tuple[int, int, int, int]]:
    start = int(timestamps[0])
    end = int(timestamps[-1]) + max(1, int(np.median(np.diff(timestamps))))
    result: list[tuple[int, int, int, int]] = []
    interval_start = start
    while interval_start < end:
        interval_end = min(interval_start + INTERVAL_NS, end)
        left = int(np.searchsorted(timestamps, interval_start, side="left"))
        right = int(np.searchsorted(timestamps, interval_end, side="left"))
        if left >= len(timestamps):
            left = len(timestamps) - 1
        if right <= left:
            right = min(len(timestamps), left + 1)
        result.append((interval_start, interval_end, left, right))
        interval_start = interval_end
    return result


def _speed_bin(speed_kph: float, *, stationary_dwell_met: bool) -> str:
    if (
        abs(speed_kph) <= STATIONARY_EPSILON_KPH
        and stationary_dwell_met
    ):
        return "stationary"
    if speed_kph < 5.0:
        return "creeping"
    if speed_kph < 30.0:
        return "low_speed"
    if speed_kph <= 60.0:
        return "medium_speed"
    return "high_speed"


def _motion_state(
    speed_kph: float,
    acceleration_mps2: float,
    *,
    stationary_dwell_met: bool,
    previously_stationary: bool,
) -> str:
    if stationary_dwell_met:
        return "stopped"
    if previously_stationary and speed_kph > 1.0:
        return "starting"
    if speed_kph < 5.0:
        return "creeping"
    if acceleration_mps2 >= 0.75:
        return "accelerating"
    if acceleration_mps2 <= -0.75:
        return "decelerating"
    return "moving"


def _expected_period_ns(
    evidence: CanonicalSceneEvidence,
    timestamps: np.ndarray,
) -> int:
    rates: list[float] = []
    if evidence.capability_manifest is not None:
        for name in ("gnss", "ins"):
            channel = evidence.capability_manifest.channels.get(name)
            if (
                channel is not None
                and channel.availability != "absent"
                and channel.nominal_rate_hz is not None
                and channel.nominal_rate_hz > 0.0
            ):
                rates.append(float(channel.nominal_rate_hz))
    if rates:
        return max(1, int(round(1_000_000_000 / max(rates))))
    return max(1, int(np.median(np.diff(timestamps))))


def _kinematic_derivatives(
    xy: np.ndarray,
    yaw: np.ndarray,
    timestamps: np.ndarray,
    *,
    maximum_gap_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    speed_mps = np.full(len(timestamps), np.nan, dtype=np.float64)
    acceleration_mps2 = np.full(len(timestamps), np.nan, dtype=np.float64)
    yaw_rate_radps = np.full(len(timestamps), np.nan, dtype=np.float64)
    segment_ids = np.full(len(timestamps), -1, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(timestamps) > maximum_gap_ns) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(timestamps)]))
    for segment_id, (start, end) in enumerate(zip(starts, ends, strict=True)):
        segment_ids[start:end] = segment_id
        if end - start < 3:
            continue
        seconds = (
            timestamps[start:end] - timestamps[start]
        ).astype(np.float64) / 1e9
        vx = np.gradient(xy[start:end, 0], seconds)
        vy = np.gradient(xy[start:end, 1], seconds)
        segment_speed = _smoothed(np.hypot(vx, vy))
        speed_mps[start:end] = segment_speed
        acceleration_mps2[start:end] = _smoothed(
            np.gradient(segment_speed, seconds)
        )
        yaw_rate_radps[start:end] = _smoothed(
            np.gradient(yaw[start:end], seconds)
        )
    return speed_mps, acceleration_mps2, yaw_rate_radps, segment_ids


def _missing_intervals(
    timestamps: np.ndarray,
    *,
    expected_period_ns: int,
    maximum_gap_ns: int,
) -> tuple[tuple[int, int], ...]:
    gaps = np.flatnonzero(np.diff(timestamps) > maximum_gap_ns)
    return tuple(
        (
            int(timestamps[index]) + expected_period_ns,
            int(timestamps[index + 1]),
        )
        for index in gaps
    )


def _overlaps_missing_interval(
    start_ns: int,
    end_ns: int,
    missing_intervals: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        missing_start < end_ns and missing_end > start_ns
        for missing_start, missing_end in missing_intervals
    )


def _heading_change(
    yaw: np.ndarray,
    timestamps: np.ndarray,
    segment_ids: np.ndarray,
    center: int,
) -> float:
    segment_id = segment_ids[center]
    segment_indexes = np.flatnonzero(segment_ids == segment_id)
    if len(segment_indexes) < 2:
        return 0.0
    window_ns = 1_500_000_000
    center_timestamp = int(timestamps[center])
    before = max(
        int(segment_indexes[0]),
        int(np.searchsorted(timestamps, center_timestamp - window_ns)),
    )
    after = min(
        int(segment_indexes[-1]),
        int(
            np.searchsorted(
                timestamps,
                center_timestamp + window_ns,
                side="right",
            )
            - 1
        ),
    )
    return float(yaw[after] - yaw[before])


def label_kinematics(
    evidence: CanonicalSceneEvidence,
) -> tuple[LabelObservation, ...]:
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    xy = _local_xy(path, float(path[0, 0]), float(path[0, 1]))
    yaw = np.unwrap(np.radians(90.0 - path[:, 2]))
    expected_period_ns = _expected_period_ns(evidence, timestamps)
    maximum_gap_ns = max(
        MAX_GAP_FLOOR_NS,
        expected_period_ns * MAX_GAP_PERIOD_MULTIPLIER,
    )
    speed_mps, acceleration, yaw_rate, segment_ids = _kinematic_derivatives(
        xy,
        yaw,
        timestamps,
        maximum_gap_ns=maximum_gap_ns,
    )
    missing_intervals = _missing_intervals(
        timestamps,
        expected_period_ns=expected_period_ns,
        maximum_gap_ns=maximum_gap_ns,
    )

    observations: list[LabelObservation] = []
    previous_speed_kph: float | None = None
    previously_stationary = False
    stationary_duration_ns = 0
    strong_response_duration_ns = 0
    for start_ns, end_ns, left, right in _interval_slices(timestamps):
        provenance = {
            "labeler_version": KINEMATICS_LABELER_VERSION,
            "kinematics_policy_version": KINEMATICS_POLICY_VERSION,
            "expected_period_ns": expected_period_ns,
            "maximum_gap_ns": maximum_gap_ns,
            "stationary_epsilon_kph": STATIONARY_EPSILON_KPH,
            "stationary_dwell_ns": STATIONARY_DWELL_NS,
            "strong_response_dwell_ns": STRONG_RESPONSE_DWELL_NS,
        }
        interval_values = np.column_stack(
            (
                speed_mps[left:right],
                acceleration[left:right],
                yaw_rate[left:right],
            )
        )
        missing_reason: str | None = None
        if _overlaps_missing_interval(
            start_ns,
            end_ns,
            missing_intervals,
        ):
            missing_reason = "timestamp_gap"
        elif not np.isfinite(interval_values).all():
            missing_reason = "insufficient_contiguous_motion_samples"
        if missing_reason is not None:
            unavailable_provenance = {
                **provenance,
                "reason": missing_reason,
            }
            for key in KINEMATIC_KEYS:
                observations.append(
                    make_observation(
                        scene_uid=evidence.scene_uid,
                        key=key,
                        status="not_observable",
                        confidence=1.0,
                        source="gnss_ins",
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        provenance=unavailable_provenance,
                    )
                )
            previous_speed_kph = None
            previously_stationary = False
            stationary_duration_ns = 0
            strong_response_duration_ns = 0
            continue

        speed = float(np.median(speed_mps[left:right]))
        speed_kph = speed * 3.6
        accel = float(np.median(acceleration[left:right]))
        yaw_rate_value = float(np.median(yaw_rate[left:right]))
        interval_duration_ns = end_ns - start_ns
        if abs(speed_kph) <= STATIONARY_EPSILON_KPH:
            stationary_duration_ns += interval_duration_ns
        else:
            stationary_duration_ns = 0
        stationary_dwell_met = stationary_duration_ns >= STATIONARY_DWELL_NS
        common = {
            "scene_uid": evidence.scene_uid,
            "confidence": 0.97,
            "source": "gnss_ins",
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "measurements": {
                "ego_speed_kph": speed_kph,
                "ego_speed_mps": speed,
                "longitudinal_acceleration_mps2": accel,
                "yaw_rate_radps": yaw_rate_value,
            },
            "provenance": provenance,
        }
        observations.append(
            make_observation(
                key="odd.ego.speed_bin",
                status="valid",
                values=(
                    _speed_bin(
                        speed_kph,
                        stationary_dwell_met=stationary_dwell_met,
                    ),
                ),
                **common,
            )
        )
        motion_state = _motion_state(
            speed_kph,
            accel,
            stationary_dwell_met=stationary_dwell_met,
            previously_stationary=previously_stationary,
        )
        observations.append(
            make_observation(
                key="event.ego.motion_state",
                status="valid",
                values=(motion_state,),
                **common,
            )
        )

        center = (left + right - 1) // 2
        heading_change = _heading_change(
            yaw,
            timestamps,
            segment_ids,
            center,
        )
        if stationary_dwell_met:
            maneuver = "stop"
        elif abs(heading_change) >= math.radians(150):
            maneuver = "u_turn"
        elif heading_change >= math.radians(12):
            maneuver = "turn_left"
        elif heading_change <= -math.radians(12):
            maneuver = "turn_right"
        else:
            maneuver = "lane_follow"
        observations.append(
            make_observation(
                key="event.ego.maneuver",
                status="valid",
                values=(maneuver,),
                **common,
            )
        )

        if accel <= -3.0:
            strong_response_duration_ns += interval_duration_ns
        else:
            strong_response_duration_ns = 0
        strong_response_dwell_met = (
            strong_response_duration_ns >= STRONG_RESPONSE_DWELL_NS
        )
        if (
            strong_response_dwell_met
            and accel <= -5.0
            and speed_kph <= 1.0
            and previous_speed_kph is not None
            and previous_speed_kph >= 15.0
        ):
            strong_response = "emergency_stop"
        elif strong_response_dwell_met:
            strong_response = "hard_brake"
        else:
            strong_response = "none"
        observations.append(
            make_observation(
                key="event.ego.strong_response",
                status="valid",
                values=(strong_response,),
                **common,
            )
        )
        previous_speed_kph = speed_kph
        previously_stationary = stationary_dwell_met
    return tuple(observations)


def _point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> tuple[float, int]:
    points = np.asarray(polyline, dtype=np.float64)[:, :2]
    starts = points[:-1]
    vectors = points[1:] - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    parameters = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, None] * vectors
    distances = np.linalg.norm(closest - point, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def _inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    vertices = np.asarray(polygon, dtype=np.float64)[:, :2]
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance_to_polygons(point: np.ndarray, polygons: Iterable[np.ndarray]) -> float:
    distance = math.inf
    for polygon in polygons:
        vertices = np.asarray(polygon, dtype=np.float64)[:, :2]
        if _inside_polygon(point, vertices):
            return 0.0
        closed = np.concatenate([vertices, vertices[:1]], axis=0)
        candidate, _ = _point_to_polyline(point, closed)
        distance = min(distance, candidate)
    return distance


def _route_action(maneuver: Maneuver, transition: TransitionType) -> str | None:
    if transition == TransitionType.MERGE:
        return "merge"
    if transition == TransitionType.SPLIT:
        return "diverge"
    return {
        Maneuver.STRAIGHT: "straight",
        Maneuver.LEFT: "turn_left",
        Maneuver.RIGHT: "turn_right",
        Maneuver.U_TURN: "u_turn",
        Maneuver.MERGE: "merge",
        Maneuver.EXIT: "diverge",
        Maneuver.DESTINATION: "lane_follow",
    }.get(maneuver)


def _horizontal_geometry(centerline: np.ndarray, segment_index: int) -> str:
    points = np.asarray(centerline, dtype=np.float64)[:, :2]
    before = max(0, segment_index - 2)
    after = min(len(points) - 2, segment_index + 2)
    first = points[min(before + 1, len(points) - 1)] - points[before]
    second = points[after + 1] - points[after]
    if np.linalg.norm(first) < 1e-6 or np.linalg.norm(second) < 1e-6:
        return "straight"
    cross = float(first[0] * second[1] - first[1] * second[0])
    dot = float(np.dot(first, second))
    angle = math.atan2(cross, dot)
    if angle >= math.radians(5):
        return "curve_left"
    if angle <= -math.radians(5):
        return "curve_right"
    return "straight"


def _vertical_geometry(centerline: np.ndarray, segment_index: int) -> str:
    points = np.asarray(centerline, dtype=np.float64)
    if points.shape[1] < 3:
        return "level"
    start = max(0, segment_index - 2)
    end = min(len(points) - 1, segment_index + 3)
    planar = float(np.linalg.norm(points[end, :2] - points[start, :2]))
    if planar < 2.0:
        return "level"
    slope = float((points[end, 2] - points[start, 2]) / planar)
    if slope > 0.02:
        return "uphill"
    if slope < -0.02:
        return "downhill"
    return "level"


def _near_polyline(
    point: np.ndarray,
    polylines: Iterable[np.ndarray],
    threshold_m: float,
) -> bool:
    return any(
        _point_to_polyline(point, polyline)[0] <= threshold_m
        for polyline in polylines
    )


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _undirected_heading_error(first: float, second: float) -> float:
    directed_error = abs(_wrap_angle(first - second))
    return min(directed_error, math.pi - directed_error)


def _polyline_heading(polyline: np.ndarray, segment_index: int) -> float:
    points = np.asarray(polyline, dtype=np.float64)[:, :2]
    delta = points[segment_index + 1] - points[segment_index]
    return math.atan2(float(delta[1]), float(delta[0]))


def _match_lane(
    point: np.ndarray,
    heading_rad: float,
    lanes: Iterable[DirectedLaneField],
) -> tuple[DirectedLaneField, float, int, float] | None:
    candidates: list[
        tuple[float, str, DirectedLaneField, float, int, float]
    ] = []
    for lane in lanes:
        distance, segment_index = _point_to_polyline(
            point, lane.centerline_enu_m
        )
        lane_heading = _polyline_heading(
            lane.centerline_enu_m, segment_index
        )
        heading_error = _undirected_heading_error(
            heading_rad, lane_heading
        )
        if (
            distance > LOCAL_MAP_MAX_DISTANCE_M
            or heading_error > LOCAL_MATCH_MAX_HEADING_ERROR_RAD
        ):
            continue
        score = distance + 2.0 * heading_error
        candidates.append(
            (
                score,
                lane.lane_id,
                lane,
                distance,
                segment_index,
                heading_error,
            )
        )
    if not candidates:
        return None
    _, _, lane, distance, segment_index, heading_error = min(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[3],
            candidate[5],
            candidate[4],
        ),
    )
    return lane, distance, segment_index, heading_error


def _match_route_segment(
    point: np.ndarray,
    heading_rad: float,
    segments: Iterable[RouteLaneSegment],
) -> tuple[RouteLaneSegment, float, int, float] | None:
    candidates: list[
        tuple[float, str, RouteLaneSegment, float, int, float]
    ] = []
    for segment in segments:
        if segment.confidence < LOCAL_ROUTE_MIN_SEGMENT_CONFIDENCE:
            continue
        distance, segment_index = _point_to_polyline(
            point, segment.centerline_enu_m
        )
        route_heading = _polyline_heading(
            segment.centerline_enu_m, segment_index
        )
        heading_error = abs(_wrap_angle(heading_rad - route_heading))
        if (
            distance > LOCAL_ROUTE_MAX_DISTANCE_M
            or heading_error > LOCAL_MATCH_MAX_HEADING_ERROR_RAD
        ):
            continue
        score = (
            distance
            + 2.0 * heading_error
            + (1.0 - segment.confidence)
        )
        candidates.append(
            (
                score,
                segment.lane_id,
                segment,
                distance,
                segment_index,
                heading_error,
            )
        )
    if not candidates:
        return None
    _, _, segment, distance, segment_index, heading_error = min(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[3],
            candidate[5],
            candidate[4],
        ),
    )
    return segment, distance, segment_index, heading_error


def _match_confidence(distance_m: float, heading_error_rad: float) -> float:
    confidence = math.exp(-distance_m / LOCAL_MAP_MAX_DISTANCE_M)
    confidence *= math.exp(
        -heading_error_rad / LOCAL_MATCH_MAX_HEADING_ERROR_RAD
    )
    return min(1.0, max(0.0, confidence))


def _road_type(road_class: str | None) -> str | None:
    if road_class is None:
        return None
    value = road_class.strip().lower().replace("-", "_")
    aliases = {
        "motorway": "motorway",
        "motorway_link": "ramp",
        "trunk": "trunk",
        "trunk_link": "ramp",
        "primary": "primary",
        "primary_link": "ramp",
        "secondary": "secondary",
        "secondary_link": "ramp",
        "tertiary": "tertiary",
        "tertiary_link": "ramp",
        "residential": "residential",
        "service": "service",
        "parking_aisle": "parking_aisle",
        "living_street": "shared_space",
        "shared_space": "shared_space",
        "ramp": "ramp",
    }
    return aliases.get(value)


def _road_context(
    lane: DirectedLaneField,
    road_type: str | None,
) -> str | None:
    if road_type in {"motorway", "residential", "parking_aisle"}:
        return {
            "motorway": "motorway",
            "residential": "residential",
            "parking_aisle": "parking",
        }[road_type]
    attributes = {
        key.lower(): value.strip().lower()
        for key, value in lane.provider_attributes.items()
    }
    location = attributes.get("location")
    if location in {"urban", "city"}:
        return "urban"
    if location in {"nonurban", "rural"}:
        return "rural"
    landuse = attributes.get("landuse")
    if landuse in {"industrial", "commercial"}:
        return "industrial"
    return None


def _lane_type(lane: DirectedLaneField) -> str | None:
    value = (lane.lane_subtype or "").strip().lower().replace("-", "_")
    if value in {"road", "lane", "driving", "vehicle", "general"}:
        return "general"
    if "bus" in value:
        return "bus"
    if value in {"bicycle", "bike", "cycleway", "cycle_lane"}:
        return "bicycle"
    if "tram" in value:
        return "tram"
    if "emergency" in value:
        return "emergency"
    if "turn" in value:
        return "turn_only"
    if "parking" in value:
        return "parking"
    if "shared" in value:
        return "shared"
    return None


def _nearby_lane_context(
    point: np.ndarray,
    matched_lane: DirectedLaneField,
    matched_index: int,
    lanes: tuple[DirectedLaneField, ...],
) -> tuple[tuple[DirectedLaneField, ...], tuple[DirectedLaneField, ...]]:
    matched_heading = _polyline_heading(
        matched_lane.centerline_enu_m, matched_index
    )
    same: list[DirectedLaneField] = []
    opposite: list[DirectedLaneField] = []
    for lane in lanes:
        distance, segment_index = _point_to_polyline(
            point, lane.centerline_enu_m
        )
        if distance > 12.0:
            continue
        lane_heading = _polyline_heading(
            lane.centerline_enu_m, segment_index
        )
        if math.cos(_wrap_angle(lane_heading - matched_heading)) >= 0.0:
            same.append(lane)
        else:
            opposite.append(lane)
    return tuple(same), tuple(opposite)


def _same_direction_carriageway_lanes(
    matched_lane: DirectedLaneField,
    nearby_same: tuple[DirectedLaneField, ...],
    lanes_by_id: dict[str, DirectedLaneField],
    *,
    topology_available: bool,
) -> tuple[DirectedLaneField, ...]:
    if matched_lane.carriageway_id is not None:
        return tuple(
            lane
            for lane in nearby_same
            if lane.carriageway_id == matched_lane.carriageway_id
        )
    if not topology_available:
        return ()
    lane_ids = {matched_lane.lane_id}
    frontier = [matched_lane.lane_id]
    while frontier:
        lane_id = frontier.pop()
        lane = lanes_by_id[lane_id]
        for adjacent_id in (
            lane.left_adjacent_lane_id,
            lane.right_adjacent_lane_id,
        ):
            if adjacent_id is None or adjacent_id in lane_ids:
                continue
            adjacent = lanes_by_id.get(adjacent_id)
            if adjacent is None or adjacent not in nearby_same:
                continue
            lane_ids.add(adjacent_id)
            frontier.append(adjacent_id)
    return tuple(lanes_by_id[lane_id] for lane_id in sorted(lane_ids))


def _lane_count_bin(count: int) -> str:
    if count <= 1:
        return "one"
    if count == 2:
        return "two"
    if count == 3:
        return "three"
    return "four_plus"


def _directionality(
    lane: DirectedLaneField,
    nearby_opposite: tuple[DirectedLaneField, ...],
    *,
    topology_available: bool,
) -> str | None:
    if lane.one_way is not None:
        return "one_way" if lane.one_way else "two_way"
    if nearby_opposite:
        return "two_way"
    if topology_available:
        return "one_way"
    return None


def _division(
    lane: DirectedLaneField,
    nearby_opposite: tuple[DirectedLaneField, ...],
) -> str | None:
    if lane.median_separated is True or lane.barrier_separated is True:
        return "divided"
    if (
        lane.median_separated is False
        and lane.barrier_separated is False
    ):
        return "undivided"
    if lane.carriageway_id is None or not nearby_opposite:
        return None
    opposite_carriageways = {
        other.carriageway_id
        for other in nearby_opposite
        if other.carriageway_id is not None
    }
    if lane.carriageway_id in opposite_carriageways:
        return "undivided"
    if opposite_carriageways:
        return "divided"
    return None


def _junction_position(
    point: np.ndarray,
    center: int,
    local_xy: np.ndarray,
    lane: DirectedLaneField,
    lanes_by_id: dict[str, DirectedLaneField],
    intersection_polygons: list[np.ndarray],
) -> tuple[str, float]:
    distance = _distance_to_polygons(point, intersection_polygons)
    if lane.is_intersection or any(
        _inside_polygon(point, polygon)
        for polygon in intersection_polygons
    ):
        return "inside", distance
    if distance <= 30.0:
        future = local_xy[center:min(len(local_xy), center + 51)]
        past = local_xy[max(0, center - 50):center]
        future_inside = any(
            _inside_polygon(candidate, polygon)
            for candidate in future
            for polygon in intersection_polygons
        )
        past_inside = any(
            _inside_polygon(candidate, polygon)
            for candidate in past
            for polygon in intersection_polygons
        )
        if future_inside:
            return "approach", distance
        if past_inside:
            return "exit", distance
    successors = [
        lanes_by_id[lane_id]
        for lane_id in lane.successor_lane_ids
        if lane_id in lanes_by_id
    ]
    predecessors = [
        lanes_by_id[lane_id]
        for lane_id in lane.predecessor_lane_ids
        if lane_id in lanes_by_id
    ]
    if any(item.is_intersection for item in successors):
        return "approach", distance
    if any(item.is_intersection for item in predecessors):
        return "exit", distance
    return "midblock", distance


def _junction_arm_headings(
    lane: DirectedLaneField,
    lanes_by_id: dict[str, DirectedLaneField],
) -> tuple[float, ...]:
    headings: list[float] = []
    predecessors = [
        lanes_by_id[lane_id]
        for lane_id in lane.predecessor_lane_ids
        if lane_id in lanes_by_id
    ]
    successors = [
        lanes_by_id[lane_id]
        for lane_id in lane.successor_lane_ids
        if lane_id in lanes_by_id
    ]
    if predecessors:
        for item in predecessors:
            headings.append(
                _wrap_angle(
                    _polyline_heading(
                        item.centerline_enu_m,
                        len(item.centerline_enu_m) - 2,
                    )
                    + math.pi
                )
            )
    else:
        headings.append(
            _wrap_angle(_polyline_heading(lane.centerline_enu_m, 0) + math.pi)
        )
    if successors:
        for item in successors:
            headings.append(_polyline_heading(item.centerline_enu_m, 0))
    elif lane.is_intersection:
        headings.append(
            _polyline_heading(
                lane.centerline_enu_m,
                len(lane.centerline_enu_m) - 2,
            )
        )
    unique: list[float] = []
    for heading in sorted(headings):
        if all(
            abs(_wrap_angle(heading - existing)) > math.radians(25.0)
            for existing in unique
        ):
            unique.append(heading)
    return tuple(unique)


def _junction_type(
    lane: DirectedLaneField,
    position: str,
    lanes_by_id: dict[str, DirectedLaneField],
) -> tuple[str | None, int, bool]:
    if position == "midblock":
        return "none", 0, False
    attributes = {
        key.lower(): value.lower()
        for key, value in lane.provider_attributes.items()
    }
    if any(
        "roundabout" in value
        for key, value in attributes.items()
        if key in {"junction", "subtype", "type"}
    ):
        return "roundabout", 0, False
    predecessor_count = len(lane.predecessor_lane_ids)
    successor_count = len(lane.successor_lane_ids)
    if predecessor_count > 1 and successor_count <= 1:
        return "merge", predecessor_count + successor_count, False
    arms = _junction_arm_headings(lane, lanes_by_id)
    branch_count = len(arms)
    if branch_count == 4:
        return "crossroad", branch_count, False
    if branch_count == 3:
        ordered = sorted((heading + 2 * math.pi) % (2 * math.pi) for heading in arms)
        gaps = [
            ordered[(index + 1) % 3]
            - ordered[index]
            if index < 2
            else ordered[0] + 2 * math.pi - ordered[2]
            for index in range(3)
        ]
        largest_gap_deg = math.degrees(max(gaps))
        if largest_gap_deg >= 150.0:
            return "t_junction", branch_count, False
        if largest_gap_deg <= 140.0:
            return "y_junction", branch_count, False
        return None, branch_count, True
    if successor_count > 1:
        return "diverge", branch_count, False
    return None, branch_count, False


def label_map_route(
    evidence: CanonicalSceneEvidence,
) -> tuple[LabelObservation, ...]:
    navigation_map = evidence.navigation_map
    route = evidence.navigation_route
    if navigation_map is None:
        provenance = {
            "labeler_version": MAP_ROUTE_LABELER_VERSION,
            "map_available": False,
            "route_available": route is not None,
            "reason": "canonical map is unavailable",
        }
        return tuple(
            make_observation(
                scene_uid=evidence.scene_uid,
                key=key,
                status="unavailable",
                confidence=1.0,
                source="map_route",
                start_timestamp_ns=evidence.start_timestamp_ns,
                end_timestamp_ns=evidence.end_timestamp_ns,
                provenance=provenance,
            )
            for key in MAP_ROUTE_KEYS
        )
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    local_xy = wgs84_to_map_xy(
        path,
        navigation_map.frame,
    )
    route_quality = route.quality if route is not None else None
    provenance: dict[str, object] = {
        "labeler_version": MAP_ROUTE_LABELER_VERSION,
        "map_version": navigation_map.map_version,
        "quality_policy": "ego_local_map_match_v1",
        "route_available": route is not None,
    }
    if route is not None and route_quality is not None:
        provenance.update(
            {
                "route_id": route.route_id,
                "route_valid_global": route.valid,
                "route_confidence_global": route.confidence,
                "matched_pose_ratio_global": (
                    route_quality.matched_pose_ratio
                ),
                "unresolved_discontinuities_global": (
                    route_quality.unresolved_discontinuities
                ),
            }
        )
    observations: list[LabelObservation] = []

    intersection_polygons = [
        polygon.points_enu_m
        for polygon in navigation_map.intersection_polygons
    ]
    crosswalk_polygons = [
        polygon.points_enu_m for polygon in navigation_map.crosswalk_polygons
    ]
    stop_lines = [line.points_enu_m for line in navigation_map.stop_lines]
    signal_positions = [
        signal.position_enu_m.reshape(1, -1)
        for signal in navigation_map.static_traffic_signals
    ]
    lane_fields = navigation_map.directed_lane_fields
    lanes_by_id = {lane.lane_id: lane for lane in lane_fields}
    topology_available = bool(
        navigation_map.layer_availability.get("lane_topology", False)
    )

    for start_ns, end_ns, left, right in _interval_slices(timestamps):
        center = (left + right - 1) // 2
        point = local_xy[center]
        heading_rad = math.radians(90.0 - float(path[center, 2]))
        lane_match = _match_lane(point, heading_rad, lane_fields)
        if lane_match is None:
            for key in MAP_ROUTE_KEYS:
                status = (
                    "unavailable"
                    if not lane_fields or key != "odd.route.action"
                    else "ambiguous"
                )
                observations.append(
                    make_observation(
                        scene_uid=evidence.scene_uid,
                        key=key,
                        status=status,
                        confidence=0.0,
                        source="map_route",
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        provenance={
                            **provenance,
                            "reason": "ego local map match unavailable",
                        },
                    )
                )
            continue

        lane, map_distance, lane_index, map_heading_error = lane_match
        map_confidence = _match_confidence(
            map_distance, map_heading_error
        )
        interval_provenance = {
            **provenance,
            "matched_lane_id": lane.lane_id,
            "local_map_match": {
                "distance_m": map_distance,
                "heading_error_rad": map_heading_error,
                "heading_semantics": "undirected_centerline_geometry",
            },
        }
        common: dict[str, object] = {
            "scene_uid": evidence.scene_uid,
            "status": "valid",
            "confidence": map_confidence,
            "source": "map_route",
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "measurements": {
                "map_lateral_distance_m": map_distance,
                "map_heading_error_rad": map_heading_error,
            },
            "provenance": interval_provenance,
        }

        road_type = _road_type(lane.road_class)
        road_context = _road_context(lane, road_type)
        observations.append(
            make_observation(
                key="odd.road.context",
                status="valid" if road_context else "unavailable",
                values=(road_context,) if road_context else (),
                confidence=map_confidence if road_context else 0.0,
                scene_uid=evidence.scene_uid,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=common["measurements"],
                provenance=interval_provenance,
            )
        )
        observations.append(
            make_observation(
                key="odd.road.type",
                status="valid" if road_type else "unavailable",
                values=(road_type,) if road_type else (),
                confidence=map_confidence if road_type else 0.0,
                scene_uid=evidence.scene_uid,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=common["measurements"],
                provenance=interval_provenance,
            )
        )
        observations.append(
            make_observation(
                key="odd.road.horizontal_geometry",
                values=(
                    _horizontal_geometry(
                        lane.centerline_enu_m, lane_index
                    ),
                ),
                **common,
            )
        )
        observations.append(
            make_observation(
                key="odd.road.vertical_geometry",
                values=(
                    _vertical_geometry(
                        lane.centerline_enu_m, lane_index
                    ),
                ),
                **common,
            )
        )

        junction_position, distance_to_intersection = _junction_position(
            point,
            center,
            local_xy,
            lane,
            lanes_by_id,
            intersection_polygons,
        )
        observations.append(
            make_observation(
                key="odd.road.junction_position",
                values=(junction_position,),
                **common,
            )
        )

        junction_type, branch_count, bedrock_eligible = _junction_type(
            lane, junction_position, lanes_by_id
        )
        junction_status = (
            "valid"
            if junction_type is not None
            else "ambiguous"
            if bedrock_eligible
            else "unavailable"
        )
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.junction_type",
                status=junction_status,
                values=(junction_type,) if junction_type else (),
                confidence=map_confidence if junction_type else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements={
                    "distance_to_intersection_m": (
                        distance_to_intersection
                        if math.isfinite(distance_to_intersection)
                        else -1.0
                    ),
                    "junction_branch_count": branch_count,
                },
                provenance={
                    **interval_provenance,
                    "bedrock_eligible": bedrock_eligible,
                    "candidate_values": (
                        ["t_junction", "y_junction"]
                        if bedrock_eligible
                        else []
                    ),
                },
            )
        )

        nearby_same, nearby_opposite = _nearby_lane_context(
            point, lane, lane_index, lane_fields
        )
        carriageway_lanes = _same_direction_carriageway_lanes(
            lane,
            nearby_same,
            lanes_by_id,
            topology_available=topology_available,
        )
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.lane_count_bin",
                status="valid" if carriageway_lanes else "unavailable",
                values=(
                    (_lane_count_bin(len(carriageway_lanes)),)
                    if carriageway_lanes
                    else ()
                ),
                confidence=map_confidence if carriageway_lanes else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements={
                    **common["measurements"],
                    "same_direction_lane_count": len(carriageway_lanes),
                },
                provenance=interval_provenance,
            )
        )
        directionality = _directionality(
            lane,
            nearby_opposite,
            topology_available=topology_available,
        )
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.directionality",
                status="valid" if directionality else "unavailable",
                values=(directionality,) if directionality else (),
                confidence=map_confidence if directionality else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=common["measurements"],
                provenance=interval_provenance,
            )
        )
        division = _division(lane, nearby_opposite)
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.division",
                status="valid" if division else "unavailable",
                values=(division,) if division else (),
                confidence=map_confidence if division else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=common["measurements"],
                provenance=interval_provenance,
            )
        )
        lane_types = sorted(
            {
                value
                for item in carriageway_lanes or (lane,)
                if (value := _lane_type(item)) is not None
            }
        )
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.lane_type_present",
                status="valid" if lane_types else "unavailable",
                values=lane_types,
                confidence=map_confidence if lane_types else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=common["measurements"],
                provenance=interval_provenance,
            )
        )

        route_match = (
            _match_route_segment(
                point, heading_rad, route.lane_sequence
            )
            if route is not None and route.lane_sequence
            else None
        )
        route_status = "unavailable"
        route_values: tuple[str, ...] = ()
        route_confidence = 0.0
        route_measurements: dict[str, float] = {}
        route_provenance = {
            **interval_provenance,
            "route_quality_policy": "local_segment_match_v1",
            "intent_semantics": (
                "reconstructed_from_ego_trace"
                if route is not None and route.estimated_destination
                else "planned_route"
            ),
            "estimated_destination": (
                route.estimated_destination if route is not None else None
            ),
        }
        if route_match is not None:
            (
                route_segment,
                route_distance,
                _,
                route_heading_error,
            ) = route_match
            route_measurements = {
                "route_lateral_distance_m": route_distance,
                "route_heading_error_rad": route_heading_error,
                "route_segment_confidence": route_segment.confidence,
            }
            route_provenance["matched_route_lane_id"] = (
                route_segment.lane_id
            )
            if route_segment.connected_from_previous:
                action = _route_action(
                    route_segment.maneuver,
                    route_segment.transition_from_previous,
                )
                if action is not None:
                    route_status = "valid"
                    route_values = (action,)
                    route_confidence = min(
                        route_segment.confidence,
                        _match_confidence(
                            route_distance, route_heading_error
                        ),
                    )
                else:
                    route_status = "ambiguous"
            else:
                route_status = "ambiguous"
                route_provenance["reason"] = (
                    "matched route segment begins at a local discontinuity"
                )
        elif route is not None and route.lane_sequence:
            route_status = "ambiguous"
            route_provenance["reason"] = (
                "no route segment passed local distance and heading gates"
            )
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.route.action",
                status=route_status,
                values=route_values,
                confidence=route_confidence,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements=route_measurements,
                provenance=route_provenance,
            )
        )

        structures: list[str] = []
        if any(
            _inside_polygon(point, polygon)
            or _distance_to_polygons(point, (polygon,)) <= 10.0
            for polygon in crosswalk_polygons
        ):
            structures.append("pedestrian_crossing")
        observations.append(
            make_observation(
                key="odd.road.special_structure",
                values=tuple(structures or ["none"]),
                **common,
            )
        )

        near_signal = any(
            float(np.linalg.norm(position[0, :2] - point)) <= 40.0
            for position in signal_positions
        )
        near_stop_line = _near_polyline(point, stop_lines, 30.0)
        controls: list[str] = []
        if near_signal:
            controls.append("traffic_light")
        if near_stop_line:
            controls.append("stop_sign")
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.traffic_control.present",
                status="valid" if controls else "unavailable",
                values=controls,
                confidence=map_confidence if controls else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                provenance=interval_provenance,
            )
        )
        if near_signal:
            control = "traffic_light"
            control_status = "valid"
        elif near_stop_line:
            control = None
            control_status = "ambiguous"
        else:
            control = None
            control_status = "unavailable"
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.junction_control",
                status=control_status,
                values=(control,) if control else (),
                confidence=map_confidence if control else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                provenance=interval_provenance,
            )
        )
    return tuple(observations)
