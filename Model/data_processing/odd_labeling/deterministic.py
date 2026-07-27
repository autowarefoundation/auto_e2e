"""Deterministic map/route and GNSS/INS scene labelers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from navigation.contracts import Maneuver, TransitionType

from .published_snapshot import CanonicalSceneEvidence
from .schema import LabelObservation, make_observation


LABELER_VERSION = "odd_deterministic_v1"
INTERVAL_NS = 1_000_000_000
STATIONARY_EPSILON_KPH = 0.5


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
        if right <= left:
            right = min(len(timestamps), left + 1)
        result.append((interval_start, interval_end, left, right))
        interval_start = interval_end
    return result


def _speed_bin(speed_kph: float) -> str:
    if abs(speed_kph) <= STATIONARY_EPSILON_KPH:
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
    previous_speed_kph: float,
) -> str:
    if speed_kph <= STATIONARY_EPSILON_KPH:
        return "stopped"
    if previous_speed_kph <= STATIONARY_EPSILON_KPH and speed_kph > 1.0:
        return "starting"
    if speed_kph < 5.0:
        return "creeping"
    if acceleration_mps2 >= 0.75:
        return "accelerating"
    if acceleration_mps2 <= -0.75:
        return "decelerating"
    return "moving"


def label_kinematics(
    evidence: CanonicalSceneEvidence,
) -> tuple[LabelObservation, ...]:
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    xy = _local_xy(path, float(path[0, 0]), float(path[0, 1]))
    vx = np.gradient(xy[:, 0], seconds)
    vy = np.gradient(xy[:, 1], seconds)
    speed_mps = _smoothed(np.hypot(vx, vy))
    acceleration = _smoothed(np.gradient(speed_mps, seconds))
    yaw = np.unwrap(np.radians(90.0 - path[:, 2]))
    yaw_rate = _smoothed(np.gradient(yaw, seconds))

    observations: list[LabelObservation] = []
    previous_speed_kph = float(speed_mps[0] * 3.6)
    for start_ns, end_ns, left, right in _interval_slices(timestamps):
        speed = float(np.median(speed_mps[left:right]))
        speed_kph = speed * 3.6
        accel = float(np.median(acceleration[left:right]))
        yaw_rate_value = float(np.median(yaw_rate[left:right]))
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
            "provenance": {
                "labeler_version": LABELER_VERSION,
                "stationary_epsilon_kph": STATIONARY_EPSILON_KPH,
            },
        }
        observations.append(
            make_observation(
                key="odd.ego.speed_bin",
                status="valid",
                values=(_speed_bin(speed_kph),),
                **common,
            )
        )
        motion_state = _motion_state(speed_kph, accel, previous_speed_kph)
        observations.append(
            make_observation(
                key="event.ego.motion_state",
                status="valid",
                values=(motion_state,),
                **common,
            )
        )

        center = (left + right - 1) // 2
        radius = max(1, int(round(1.5 / max(np.median(np.diff(seconds)), 1e-3))))
        before = max(0, center - radius)
        after = min(len(yaw) - 1, center + radius)
        heading_change = float(yaw[after] - yaw[before])
        if abs(heading_change) >= math.radians(150):
            maneuver = "u_turn"
        elif heading_change >= math.radians(12):
            maneuver = "turn_left"
        elif heading_change <= -math.radians(12):
            maneuver = "turn_right"
        elif speed_kph <= STATIONARY_EPSILON_KPH:
            maneuver = "stop"
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

        if accel <= -5.0 and speed_kph <= 1.0 and previous_speed_kph >= 15.0:
            strong_response = "emergency_stop"
        elif accel <= -3.0:
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


def _lane_context(
    point: np.ndarray,
    route_tangent: np.ndarray,
    lane_fields: Iterable[np.ndarray],
) -> tuple[int, int]:
    same = 0
    opposite = 0
    tangent_norm = float(np.linalg.norm(route_tangent))
    if tangent_norm < 1e-6:
        return same, opposite
    for centerline in lane_fields:
        distance, index = _point_to_polyline(point, centerline)
        if distance > 8.0:
            continue
        points = np.asarray(centerline, dtype=np.float64)[:, :2]
        lane_tangent = points[index + 1] - points[index]
        if float(np.dot(route_tangent, lane_tangent)) >= 0.0:
            same += 1
        else:
            opposite += 1
    return same, opposite


def label_map_route(
    evidence: CanonicalSceneEvidence,
) -> tuple[LabelObservation, ...]:
    navigation_map = evidence.navigation_map
    route = evidence.navigation_route
    path = evidence.path_latlon_heading_timestamp
    timestamps = path[:, 3].astype(np.int64)
    local_xy = _local_xy(
        path,
        navigation_map.frame.origin_latitude_deg,
        navigation_map.frame.origin_longitude_deg,
    )
    quality = route.quality
    quality_ok = (
        route.valid
        and route.confidence >= 0.6
        and quality.matched_pose_ratio >= 0.8
        and quality.unresolved_discontinuities == 0
    )
    confidence = min(
        float(route.confidence),
        float(quality.matched_pose_ratio),
    )
    provenance = {
        "labeler_version": LABELER_VERSION,
        "map_version": navigation_map.map_version,
        "route_id": route.route_id,
        "route_confidence": route.confidence,
        "matched_pose_ratio": quality.matched_pose_ratio,
    }
    observations: list[LabelObservation] = []

    if not quality_ok or not route.lane_sequence:
        for key in (
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
        ):
            observations.append(
                make_observation(
                    scene_uid=evidence.scene_uid,
                    key=key,
                    status="unavailable",
                    confidence=0.0,
                    source="map_route",
                    start_timestamp_ns=evidence.start_timestamp_ns,
                    end_timestamp_ns=evidence.end_timestamp_ns,
                    provenance=provenance,
                )
            )
        return tuple(observations)

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
    lane_fields = [
        lane.centerline_enu_m for lane in navigation_map.directed_lane_fields
    ]

    for start_ns, end_ns, left, right in _interval_slices(timestamps):
        center = (left + right - 1) // 2
        point = local_xy[center]
        best_segment = None
        best_distance = math.inf
        best_line_index = 0
        for segment in route.lane_sequence:
            distance, line_index = _point_to_polyline(
                point, segment.centerline_enu_m
            )
            if distance < best_distance:
                best_segment = segment
                best_distance = distance
                best_line_index = line_index
        if best_segment is None or best_distance > 15.0:
            for key in (
                "odd.road.horizontal_geometry",
                "odd.road.vertical_geometry",
                "odd.road.junction_type",
                "odd.road.junction_position",
                "odd.route.action",
            ):
                observations.append(
                    make_observation(
                        scene_uid=evidence.scene_uid,
                        key=key,
                        status="ambiguous",
                        confidence=0.0,
                        source="map_route",
                        start_timestamp_ns=start_ns,
                        end_timestamp_ns=end_ns,
                        provenance=provenance,
                    )
                )
            continue

        common = {
            "scene_uid": evidence.scene_uid,
            "status": "valid",
            "confidence": confidence,
            "source": "map_route",
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "measurements": {"route_lateral_distance_m": best_distance},
            "provenance": provenance,
        }
        action = _route_action(
            best_segment.maneuver,
            best_segment.transition_from_previous,
        )
        observations.append(
            make_observation(
                key="odd.route.action",
                status="valid" if action else "ambiguous",
                values=(action,) if action else (),
                confidence=confidence if action else 0.0,
                source="map_route",
                scene_uid=evidence.scene_uid,
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements={"route_lateral_distance_m": best_distance},
                provenance=provenance,
            )
        )
        observations.append(
            make_observation(
                key="odd.road.horizontal_geometry",
                values=(
                    _horizontal_geometry(
                        best_segment.centerline_enu_m, best_line_index
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
                        best_segment.centerline_enu_m, best_line_index
                    ),
                ),
                **common,
            )
        )

        inside_intersection = any(
            _inside_polygon(point, polygon)
            for polygon in intersection_polygons
        )
        distance_to_intersection = _distance_to_polygons(
            point, intersection_polygons
        )
        if inside_intersection:
            junction_position = "inside"
        elif distance_to_intersection <= 30.0:
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
            junction_position = "approach" if future_inside else (
                "exit" if past_inside else "midblock"
            )
        else:
            junction_position = "midblock"
        observations.append(
            make_observation(
                key="odd.road.junction_position",
                values=(junction_position,),
                **common,
            )
        )

        if best_segment.transition_from_previous == TransitionType.MERGE:
            junction_type = "merge"
        elif best_segment.transition_from_previous == TransitionType.SPLIT:
            junction_type = "diverge"
        elif junction_position == "midblock":
            junction_type = "none"
        else:
            junction_type = None
        observations.append(
            make_observation(
                scene_uid=evidence.scene_uid,
                key="odd.road.junction_type",
                status="valid" if junction_type else "ambiguous",
                values=(junction_type,) if junction_type else (),
                confidence=confidence if junction_type else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                measurements={
                    "distance_to_intersection_m": (
                        distance_to_intersection
                        if math.isfinite(distance_to_intersection)
                        else -1.0
                    )
                },
                provenance=provenance,
            )
        )

        route_points = np.asarray(
            best_segment.centerline_enu_m, dtype=np.float64
        )[:, :2]
        route_tangent = (
            route_points[best_line_index + 1] - route_points[best_line_index]
        )
        same_lanes, opposite_lanes = _lane_context(
            point, route_tangent, lane_fields
        )
        if same_lanes > 0:
            lane_count = (
                "one"
                if same_lanes == 1
                else "two"
                if same_lanes == 2
                else "three"
                if same_lanes == 3
                else "four_plus"
            )
            observations.append(
                make_observation(
                    key="odd.road.lane_count_bin",
                    values=(lane_count,),
                    **common,
                )
            )
            observations.append(
                make_observation(
                    key="odd.road.directionality",
                    values=("two_way" if opposite_lanes else "one_way",),
                    **common,
                )
            )
            observations.append(
                make_observation(
                    key="odd.road.lane_type_present",
                    values=("general",),
                    **common,
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
                confidence=confidence if controls else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                provenance=provenance,
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
                confidence=confidence if control else 0.0,
                source="map_route",
                start_timestamp_ns=start_ns,
                end_timestamp_ns=end_ns,
                provenance=provenance,
            )
        )
    return tuple(observations)
