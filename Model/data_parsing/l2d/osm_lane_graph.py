"""OSM ways and nodes to the canonical ``osm_lane_graph_v1`` lane graph.

L2D ships no HD map, so the Reactive 14-channel map is reproduced from
OpenStreetMap: every drivable way is split at junctions into directed lanes
with centerlines, boundaries, successors, neighbours, and levels, and the
point tags that HD maps carry as geometry (traffic signals, stop signs,
crossings) become stop lines, signals, and crosswalk polygons. The payload is
what :class:`navigation.osm_adapter.OSMMapAdapter` reads, so L2D renders
through the same ``NavigationMap`` and native rasterizer as nuPlan and
KITScenes.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from navigation.osm_adapter import OSM_LANE_GRAPH_SCHEMA_VERSION

OSM_LANE_GRAPH_BUILDER_VERSION = "osm_lane_graph_builder_v1"

DRIVABLE_HIGHWAYS = frozenset({
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "secondary",
    "secondary_link",
    "service",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "unclassified",
})

# Lane widths when a way carries no width tag (German design guidance).
_DEFAULT_LANE_WIDTH_M = {
    "motorway": 3.75,
    "motorway_link": 3.5,
    "trunk": 3.5,
    "trunk_link": 3.5,
    "primary": 3.25,
    "primary_link": 3.25,
    "secondary": 3.25,
    "secondary_link": 3.25,
    "tertiary": 3.0,
    "tertiary_link": 3.0,
    "unclassified": 3.0,
    "residential": 3.0,
    "living_street": 2.75,
    "service": 2.75,
}
_CROSSWALK_DEPTH_M = 3.0
_INTERSECTION_MARGIN_M = 2.0
_STRAIGHT_LIMIT_RAD = math.radians(30.0)


@dataclasses.dataclass(frozen=True)
class OSMNode:
    node_id: int
    longitude_deg: float
    latitude_deg: float
    tags: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class OSMWay:
    way_id: int
    node_ids: tuple[int, ...]
    tags: Mapping[str, str]


def utm_epsg(longitude_deg: float, latitude_deg: float) -> int:
    """ETRS89 UTM zone for Germany, WGS84 UTM elsewhere."""
    zone = int((longitude_deg + 180.0) // 6.0) + 1
    if 47.0 <= latitude_deg <= 56.0 and zone in (32, 33):
        return 25800 + zone
    return (32600 if latitude_deg >= 0.0 else 32700) + zone


def _float_tag(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    multiplier = 1.0
    if text.endswith(" ft"):
        text, multiplier = text[:-3].strip(), 0.3048
    elif text.endswith("m"):
        text = text[:-1].strip()
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _int_tag(value: str | None) -> int | None:
    number = _float_tag(value)
    return int(number) if number is not None and number >= 1.0 else None


def _level(tags: Mapping[str, str]) -> int:
    """OSM layer, defaulting bridges up and tunnels down."""
    layer = tags.get("layer")
    if layer is not None:
        try:
            return int(float(layer))
        except ValueError:
            pass
    if tags.get("bridge", "no") not in ("no", ""):
        return 1
    if tags.get("tunnel", "no") not in ("no", ""):
        return -1
    return 0


def lane_layout(tags: Mapping[str, str]) -> tuple[int, int, float]:
    """(forward lanes, reverse lanes, lane width) of one way."""
    oneway = str(tags.get("oneway", "no")).strip().lower()
    if tags.get("highway") in ("motorway", "motorway_link") and oneway == "no":
        oneway = "yes"
    if tags.get("junction") in ("roundabout", "circular") and oneway == "no":
        oneway = "yes"
    total = _int_tag(tags.get("lanes"))
    forward = _int_tag(tags.get("lanes:forward"))
    backward = _int_tag(tags.get("lanes:backward"))
    if oneway in ("yes", "true", "1"):
        forward, backward = forward or total or 1, 0
    elif oneway == "-1":
        forward, backward = 0, backward or total or 1
    else:
        if forward is None and backward is None:
            half = max(1, (total or 2) // 2)
            forward, backward = half, max(1, (total or 2) - half)
        elif forward is None:
            assert backward is not None
            forward = max(1, (total or backward + 1) - backward)
        elif backward is None:
            backward = max(1, (total or forward + 1) - forward)
    assert forward is not None and backward is not None
    lanes = forward + backward
    width = _float_tag(tags.get("width"))
    default = _DEFAULT_LANE_WIDTH_M.get(tags.get("highway", ""), 3.0)
    lane_width = min(5.0, max(2.25, width / lanes)) if width else default
    return forward, backward, lane_width


def _offset_polyline(points: np.ndarray, offset_m: float) -> np.ndarray:
    """Shift a polyline left by ``offset_m`` (right when negative)."""
    if offset_m == 0.0:
        return points.copy()
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-9)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return points + normals * offset_m


def _heading(points: np.ndarray, *, at_end: bool) -> float:
    delta = points[-1] - points[-2] if at_end else points[1] - points[0]
    return math.atan2(float(delta[1]), float(delta[0]))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _rounded(points: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


@dataclasses.dataclass(frozen=True)
class _Segment:
    """A way between two junction nodes, in its OSM drawing direction."""

    segment_id: int
    way_id: int
    node_ids: tuple[int, ...]
    points: np.ndarray
    tags: Mapping[str, str]
    level: int
    forward_lanes: int
    reverse_lanes: int
    lane_width_m: float

    @property
    def half_width_m(self) -> float:
        return 0.5 * (self.forward_lanes + self.reverse_lanes) * self.lane_width_m


class _Projector:
    def __init__(self, origin_lon: float, origin_lat: float, epsg: int) -> None:
        from pyproj import Transformer

        self._transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        self._origin = self._transformer.transform(origin_lon, origin_lat)

    def __call__(self, lon: Sequence[float], lat: Sequence[float]) -> np.ndarray:
        x, y = self._transformer.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.column_stack([np.asarray(x) - self._origin[0], np.asarray(y) - self._origin[1]])


def _split_segments(
    ways: Sequence[OSMWay],
    nodes: Mapping[int, OSMNode],
    project: _Projector,
) -> list[_Segment]:
    usage: dict[int, int] = defaultdict(int)
    for way in ways:
        for index, node_id in enumerate(way.node_ids):
            # Way ends and every shared node are junctions.
            usage[node_id] += 2 if index in (0, len(way.node_ids) - 1) else 1
    segments: list[_Segment] = []
    per_way: dict[int, int] = defaultdict(int)
    for way in sorted(ways, key=lambda item: item.way_id):
        present = [node_id for node_id in way.node_ids if node_id in nodes]
        if len(present) < 2:
            continue
        forward, reverse, width = lane_layout(way.tags)
        level = _level(way.tags)
        start = 0
        for index in range(1, len(present)):
            if index != len(present) - 1 and usage[present[index]] < 2:
                continue
            node_ids = tuple(present[start:index + 1])
            points = project(
                [nodes[n].longitude_deg for n in node_ids],
                [nodes[n].latitude_deg for n in node_ids],
            )
            keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.05])
            if keep.sum() >= 2:
                segments.append(_Segment(
                    segment_id=way.way_id * 1000 + per_way[way.way_id],
                    way_id=way.way_id,
                    node_ids=node_ids,
                    points=points[keep],
                    tags=way.tags,
                    level=level,
                    forward_lanes=forward,
                    reverse_lanes=reverse,
                    lane_width_m=width,
                ))
                per_way[way.way_id] += 1
            start = index
    return segments


def _lane_offsets(segment: _Segment, direction: str) -> list[float]:
    """Left offsets of each lane centerline in the segment's drawing frame.

    Right-hand traffic: forward lanes lie right of the drawn line, reverse
    lanes left of it; a one-way road is centred on the line. Index 0 is the
    leftmost lane in the direction of travel (on a two-way road, the one
    next to the road centre), so index - 1 is always the left neighbour.
    """
    width = segment.lane_width_m
    if segment.forward_lanes and segment.reverse_lanes:
        count = segment.forward_lanes if direction == "forward" else segment.reverse_lanes
        sign = -1.0 if direction == "forward" else 1.0
        return [sign * (index + 0.5) * width for index in range(count)]
    count = segment.forward_lanes or segment.reverse_lanes
    left_edge = 0.5 * count * width
    # A reverse one-way travels against the drawing, so its left is the
    # drawing's right.
    sign = 1.0 if direction == "forward" else -1.0
    return [sign * (left_edge - (index + 0.5) * width) for index in range(count)]


def turn_maneuver(incoming_heading_rad: float, outgoing_heading_rad: float) -> str:
    """The maneuver of leaving one lane for the next at a junction."""
    turn = _wrap(outgoing_heading_rad - incoming_heading_rad)
    if abs(turn) <= _STRAIGHT_LIMIT_RAD:
        return "straight"
    if abs(turn) >= math.radians(150.0):
        return "u_turn"
    return "left" if turn > 0.0 else "right"


_LaneCenter = tuple[str, str, np.ndarray]  # (provider id, direction, centerline)


def _directed_lanes(
    segments: Sequence[_Segment],
) -> tuple[list[dict[str, Any]], dict[int, list[_LaneCenter]], dict[int, list[_LaneCenter]]]:
    """Lanes, lanes ending at each node, and lanes of each segment."""
    lanes: list[dict[str, Any]] = []
    by_segment: dict[int, list[_LaneCenter]] = defaultdict(list)
    # node id -> lanes starting there (provider id, start heading vector)
    starts: dict[int, list[tuple[str, float, int, str]]] = defaultdict(list)
    ends: list[tuple[dict[str, Any], int, float, int, str]] = []
    approach: dict[int, list[_LaneCenter]] = defaultdict(list)
    for segment in segments:
        for direction in ("forward", "reverse"):
            count = segment.forward_lanes if direction == "forward" else segment.reverse_lanes
            if not count:
                continue
            offsets = _lane_offsets(segment, direction)
            ids = [f"{segment.segment_id}:{direction}:{index}" for index in range(count)]
            for index, offset in enumerate(offsets):
                center = _offset_polyline(segment.points, offset)
                left = _offset_polyline(segment.points, offset + 0.5 * segment.lane_width_m)
                right = _offset_polyline(segment.points, offset - 0.5 * segment.lane_width_m)
                if direction == "reverse":
                    center, left, right = center[::-1], right[::-1], left[::-1]
                # Lane index 0 is innermost: its left neighbour is index-1 toward the centre.
                lane: dict[str, Any] = {
                    "way_id": segment.segment_id,
                    "osm_way_id": segment.way_id,
                    "direction": direction,
                    "lane_index": index,
                    "centerline_enu_m": _rounded(center),
                    "left_boundary_enu_m": _rounded(left),
                    "right_boundary_enu_m": _rounded(right),
                    "successors": [],
                    "left_adjacent": ids[index - 1] if index > 0 else None,
                    "right_adjacent": ids[index + 1] if index + 1 < count else None,
                    # OSM has no lanes inside junctions, so a lane between
                    # junctions never turns; the turn is between lanes and
                    # is read off consecutive route lanes instead.
                    "maneuver": "straight",
                    "level": segment.level,
                    "confidence": 1.0 if any(key in segment.tags for key in ("lanes", "lanes:forward", "lanes:backward")) else 0.8,
                }
                lanes.append(lane)
                first, last = (
                    (segment.node_ids[0], segment.node_ids[-1])
                    if direction == "forward"
                    else (segment.node_ids[-1], segment.node_ids[0])
                )
                starts[first].append((ids[index], _heading(center, at_end=False), segment.segment_id, direction))
                ends.append((lane, last, _heading(center, at_end=True), segment.segment_id, direction))
                approach[last].append((ids[index], direction, center))
                by_segment[segment.segment_id].append((ids[index], direction, center))
    for lane, node_id, heading, segment_id, direction in ends:
        candidates = [
            (lane_id, start_heading)
            for lane_id, start_heading, other_segment, other_direction in starts.get(node_id, ())
            if not (other_segment == segment_id and other_direction != direction)
        ]
        if not candidates:
            # Dead end: a U-turn onto the opposite direction is the only way on.
            candidates = [
                (lane_id, start_heading)
                for lane_id, start_heading, _, _ in starts.get(node_id, ())
            ]
        lane["successors"] = sorted(
            lane_id for lane_id, start_heading in candidates
            if abs(_wrap(start_heading - heading)) < math.radians(170.0) or len(candidates) == 1
        )
    return lanes, approach, by_segment


def _rectangle(center: np.ndarray, along: np.ndarray, half_length: float, half_width: float) -> np.ndarray:
    along = along / max(float(np.linalg.norm(along)), 1e-9)
    across = np.array([-along[1], along[0]])
    return np.asarray([
        center - along * half_length - across * half_width,
        center + along * half_length - across * half_width,
        center + along * half_length + across * half_width,
        center - along * half_length + across * half_width,
    ])


def _semantics(
    segments: Sequence[_Segment],
    nodes: Mapping[int, OSMNode],
    approach: Mapping[int, list[_LaneCenter]],
    by_segment: Mapping[int, list[_LaneCenter]],
    project: _Projector,
) -> dict[str, list[dict[str, Any]]]:
    incident: dict[int, list[tuple[_Segment, int]]] = defaultdict(list)
    for segment in segments:
        for position, node_id in enumerate(segment.node_ids):
            incident[node_id].append((segment, position))
    end_degree: dict[int, int] = defaultdict(int)
    for segment in segments:
        end_degree[segment.node_ids[0]] += 1
        end_degree[segment.node_ids[-1]] += 1

    def where(node_id: int) -> np.ndarray:
        node = nodes[node_id]
        return project([node.longitude_deg], [node.latitude_deg])[0]

    def road_direction(segment: _Segment, position: int) -> np.ndarray:
        points = segment.points
        index = min(max(position, 0), len(points) - 1)
        if index + 1 < len(points):
            return points[index + 1] - points[index]
        return points[index] - points[index - 1]

    intersections: list[dict[str, Any]] = []
    radius_of: dict[int, float] = {}
    for node_id in sorted(end_degree):
        if end_degree[node_id] < 3:
            continue
        radius = max(segment.half_width_m for segment, _ in incident[node_id]) + _INTERSECTION_MARGIN_M
        radius_of[node_id] = radius
        center = where(node_id)
        angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False) + math.pi / 8.0
        polygon = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        intersections.append({
            "id": str(node_id),
            "points_enu_m": _rounded(polygon),
            "level": min(segment.level for segment, _ in incident[node_id]),
        })

    signals: list[dict[str, Any]] = []
    stop_lines: list[dict[str, Any]] = []
    crosswalks: list[dict[str, Any]] = []
    for node_id in sorted(incident):
        tags = nodes[node_id].tags
        highway = tags.get("highway", "")
        level = min(segment.level for segment, _ in incident[node_id])
        center = where(node_id)
        if highway == "traffic_signals" or tags.get("crossing") == "traffic_signals":
            signals.append({"id": str(node_id), "position_enu_m": [round(float(center[0]), 3), round(float(center[1]), 3)], "level": level})
        if highway in ("traffic_signals", "stop"):
            # Junction nodes stop traffic before the intersection; a node on
            # the approach road stops it at the node, in the tagged direction.
            wanted = _tagged_directions(tags)
            back = radius_of.get(node_id, 0.0)
            for lane_id, direction, centerline in approach.get(node_id, ()):
                if direction in wanted:
                    line = _stop_line_before_end(centerline, back)
                    if line is not None:
                        stop_lines.append({"id": f"{node_id}:{lane_id}", "points_enu_m": _rounded(line), "level": level})
            for segment, position in incident[node_id]:
                if position in (0, len(segment.node_ids) - 1):
                    continue
                for lane_id, direction, centerline in by_segment.get(segment.segment_id, ()):
                    if direction in wanted:
                        line = _stop_line_at(centerline, center)
                        stop_lines.append({"id": f"{node_id}:{lane_id}", "points_enu_m": _rounded(line), "level": level})
        if highway == "crossing" or "crossing" in tags:
            segment, position = max(incident[node_id], key=lambda item: item[0].half_width_m)
            along = road_direction(segment, position)
            polygon = _rectangle(center, along, _CROSSWALK_DEPTH_M / 2.0, segment.half_width_m + 0.5)
            crosswalks.append({"id": str(node_id), "points_enu_m": _rounded(polygon), "level": level})
    return {
        "intersections": intersections,
        "crosswalks": crosswalks,
        "stop_lines": stop_lines,
        "traffic_signals": signals,
    }


def _tagged_directions(tags: Mapping[str, str]) -> frozenset[str]:
    """Lane directions a signal or stop sign applies to (both when untagged)."""
    value = tags.get("traffic_signals:direction") or tags.get("direction") or ""
    value = value.strip().lower()
    if value == "forward":
        return frozenset({"forward"})
    if value == "backward":
        return frozenset({"reverse"})
    return frozenset({"forward", "reverse"})


def _across(point: np.ndarray, along: np.ndarray) -> np.ndarray:
    along = along / max(float(np.linalg.norm(along)), 1e-9)
    normal = np.array([-along[1], along[0]])
    return np.asarray([point - normal * 1.5, point + normal * 1.5])


def _stop_line_at(centerline: np.ndarray, point: np.ndarray) -> np.ndarray:
    """A line across the lane at the centerline point closest to ``point``."""
    points = np.asarray(centerline, dtype=np.float64)
    best = (math.inf, points[0], points[1] - points[0])
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        ratio = float(np.clip(np.dot(point - start, delta) / max(float(np.dot(delta, delta)), 1e-12), 0.0, 1.0))
        foot = start + ratio * delta
        distance = float(np.linalg.norm(point - foot))
        if distance < best[0]:
            best = (distance, foot, delta)
    return _across(best[1], best[2])


def _stop_line_before_end(centerline: np.ndarray, back_m: float) -> np.ndarray | None:
    """A line across the lane, ``back_m`` before its end."""
    points = np.asarray(centerline, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= back_m + 0.5:
        return None
    target = total - back_m
    walked = 0.0
    for index, length in enumerate(lengths):
        if walked + length >= target:
            ratio = (target - walked) / max(length, 1e-9)
            point = points[index] + ratio * (points[index + 1] - points[index])
            return _across(point, points[index + 1] - points[index])
        walked += length
    return None


def build_osm_lane_graph(
    nodes: Mapping[int, OSMNode],
    ways: Iterable[OSMWay],
    *,
    origin_longitude_deg: float,
    origin_latitude_deg: float,
    map_version: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical lane graph of the given drivable ways."""
    drivable = [way for way in ways if way.tags.get("highway") in DRIVABLE_HIGHWAYS]
    if not drivable:
        raise ValueError("no drivable OSM ways in the region")
    epsg = utm_epsg(origin_longitude_deg, origin_latitude_deg)
    project = _Projector(origin_longitude_deg, origin_latitude_deg, epsg)
    segments = _split_segments(drivable, nodes, project)
    lanes, approach, by_segment = _directed_lanes(segments)
    if not lanes:
        raise ValueError("OSM ways produced no lanes")
    return {
        "schema_version": OSM_LANE_GRAPH_SCHEMA_VERSION,
        "map_version": map_version,
        "frame": {
            "frame_id": f"{map_version}:local_enu",
            "origin_latitude_deg": float(origin_latitude_deg),
            "origin_longitude_deg": float(origin_longitude_deg),
            "projection": f"EPSG:{epsg} local ENU",
        },
        "lanes": sorted(lanes, key=lambda lane: (lane["way_id"], lane["direction"], lane["lane_index"])),
        "semantic": _semantics(segments, nodes, approach, by_segment, project),
        "provenance": {"builder_version": OSM_LANE_GRAPH_BUILDER_VERSION, **dict(provenance)},
    }
