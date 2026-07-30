"""Lanelet2 semantic-map adapter for the canonical navigation contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import (
    DirectedLaneField,
    MapFrame,
    NavigationMap,
    PolygonPrimitive,
    PolylinePrimitive,
    StaticTrafficSignal,
)


LANELET2_ADAPTER_VERSION = "lanelet2_adapter_v2"


def _attr(obj: Any, name: str, default: str = "") -> str:
    attributes = getattr(obj, "attributes", {})
    try:
        return str(attributes[name]) if name in attributes else default
    except (KeyError, TypeError):
        return default


def _attributes(obj: Any) -> dict[str, str]:
    values = getattr(obj, "attributes", {})
    try:
        items = values.items()
    except AttributeError:
        try:
            items = ((key, values[key]) for key in values)
        except (KeyError, TypeError):
            return {}
    return {
        str(key): str(value)
        for key, value in items
        if str(key) and str(value)
    }


def _first_attribute(
    attributes: dict[str, str],
    *names: str,
) -> str | None:
    for name in names:
        value = attributes.get(name, "").strip()
        if value:
            return value
    return None


def _optional_bool(
    attributes: dict[str, str],
    *names: str,
) -> bool | None:
    value = _first_attribute(attributes, *names)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _separation_semantics(
    lane_attributes: dict[str, str],
    left_attributes: dict[str, str],
    right_attributes: dict[str, str],
) -> tuple[bool | None, bool | None]:
    boundary_values = {
        value.lower()
        for attributes in (left_attributes, right_attributes)
        for key, value in attributes.items()
        if key in {"type", "subtype", "barrier", "road_border"}
    }
    median = _optional_bool(
        lane_attributes,
        "median_separated",
        "median",
        "divided",
    )
    barrier = _optional_bool(
        lane_attributes,
        "barrier_separated",
        "barrier",
    )
    if median is None and any("median" in value for value in boundary_values):
        median = True
    barrier_types = (
        "barrier",
        "concrete",
        "guard_rail",
        "guardrail",
        "jersey",
    )
    if barrier is None and any(
        token in value
        for value in boundary_values
        for token in barrier_types
    ):
        barrier = True
    return median, barrier


def _level(obj: Any) -> int | None:
    raw = _attr(obj, "layer") or _attr(obj, "level")
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _points(linestring: Iterable[Any]) -> np.ndarray:
    values = [
        [
            float(point.x),
            float(point.y),
            float(getattr(point, "z", 0.0)),
        ]
        for point in linestring
    ]
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


def _polygon_points(lanelet: Any) -> np.ndarray:
    polygon2d = getattr(lanelet, "polygon2d", None)
    if callable(polygon2d):
        points = _points(polygon2d())
        if len(points) >= 3:
            return points
    left = _points(lanelet.leftBound)
    right = _points(lanelet.rightBound)
    if len(left) < 2 or len(right) < 2:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate([left, right[::-1]], axis=0)


def _centroid(parameters: Iterable[Any]) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for parameter in parameters:
        try:
            array = _points(parameter)
        except (AttributeError, TypeError):
            continue
        if len(array):
            points.append(array)
    if not points:
        return None
    return np.concatenate(points, axis=0).mean(axis=0)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Lanelet2MapAdapter:
    """Extract deterministic semantic primitives from a KITScenes ``SceneMap``."""

    def __init__(
        self,
        scene_map: Any,
        *,
        map_version: str,
        map_sha256: str,
        frame_id: str,
        source_revision: str,
    ) -> None:
        if not map_version or not map_sha256 or not frame_id:
            raise ValueError("map adapter metadata must not be empty")
        self.scene_map = scene_map
        self.map_version = map_version
        self.map_sha256 = map_sha256
        self.frame_id = frame_id
        self.source_revision = source_revision

    @property
    def lanelets_by_id(self) -> dict[int, Any]:
        return {
            int(lanelet.id): lanelet
            for lanelet in self.scene_map.lanelet_map.laneletLayer
        }

    def _can_pass(self, lanelet: Any) -> bool:
        try:
            return bool(self.scene_map.traffic_rules.canPass(lanelet))
        except Exception:
            return True

    def _is_intersection(self, lanelet: Any) -> bool:
        turn = _attr(lanelet, "turn_direction").lower()
        if turn in {"left", "right", "u_turn"}:
            return True
        graph = self.scene_map.routing_graph
        try:
            if len(list(graph.following(lanelet))) > 1:
                return True
        except Exception:
            pass
        try:
            if len(list(graph.previous(lanelet))) > 1:
                return True
        except Exception:
            pass
        return False

    def _related_lane_ids(
        self,
        lanelet: Any,
        relation: str,
    ) -> tuple[str, ...]:
        try:
            related = getattr(self.scene_map.routing_graph, relation)(lanelet)
        except Exception:
            return ()
        if related is None:
            return ()
        if not isinstance(related, (list, tuple)):
            related = (related,)
        return tuple(
            f"lanelet2:{int(item.id)}"
            for item in sorted(related, key=lambda value: int(value.id))
            if item is not None and self._can_pass(item)
        )

    def _traffic_signals(self) -> tuple[StaticTrafficSignal, ...]:
        signals: list[StaticTrafficSignal] = []
        layer = getattr(
            self.scene_map.lanelet_map,
            "regulatoryElementLayer",
            (),
        )
        for regulatory in sorted(layer, key=lambda item: int(item.id)):
            subtype = (
                _attr(regulatory, "subtype")
                or _attr(regulatory, "type")
            ).lower()
            if "traffic_light" not in subtype and "traffic_signal" not in subtype:
                continue
            parameters = getattr(regulatory, "parameters", {})
            candidates: list[Any] = []
            for role in ("refers", "light_bulbs", "traffic_light"):
                try:
                    candidates.extend(parameters.get(role, ()))
                except AttributeError:
                    if role in parameters:
                        candidates.extend(parameters[role])
            position = _centroid(candidates)
            if position is None:
                continue
            signals.append(
                StaticTrafficSignal(
                    signal_id=f"lanelet2:signal:{int(regulatory.id)}",
                    position_enu_m=position,
                    level=_level(regulatory),
                )
            )
        return tuple(signals)

    def extract(self) -> NavigationMap:
        drivable: list[PolygonPrimitive] = []
        boundaries: dict[int | str, PolylinePrimitive] = {}
        centerlines: list[PolylinePrimitive] = []
        intersections: list[PolygonPrimitive] = []
        crosswalks: list[PolygonPrimitive] = []
        directions: list[DirectedLaneField] = []

        for lanelet in sorted(
            self.scene_map.lanelet_map.laneletLayer,
            key=lambda item: int(item.id),
        ):
            lanelet_id = int(lanelet.id)
            canonical_id = f"lanelet2:{lanelet_id}"
            level = _level(lanelet)
            polygon = _polygon_points(lanelet)
            subtype = _attr(lanelet, "subtype").lower()
            is_crosswalk = subtype == "crosswalk"
            can_pass = self._can_pass(lanelet)

            if len(polygon) >= 3:
                primitive = PolygonPrimitive(
                    primitive_id=f"{canonical_id}:polygon",
                    points_enu_m=polygon,
                    level=level,
                )
                if is_crosswalk:
                    crosswalks.append(primitive)
                elif can_pass:
                    drivable.append(primitive)
                    if self._is_intersection(lanelet):
                        intersections.append(primitive)

            for side, boundary in (
                ("left", lanelet.leftBound),
                ("right", lanelet.rightBound),
            ):
                points = _points(boundary)
                if len(points) < 2:
                    continue
                native_id = getattr(boundary, "id", f"{lanelet_id}:{side}")
                if native_id in boundaries:
                    continue
                boundaries[native_id] = PolylinePrimitive(
                    primitive_id=f"lanelet2:boundary:{native_id}",
                    points_enu_m=points,
                    level=_level(boundary) if _level(boundary) is not None else level,
                )

            centerline = _points(lanelet.centerline)
            if len(centerline) >= 2 and not is_crosswalk:
                centerlines.append(
                    PolylinePrimitive(
                        primitive_id=f"{canonical_id}:centerline",
                        points_enu_m=centerline,
                        level=level,
                    )
                )
                if can_pass:
                    lane_attributes = _attributes(lanelet)
                    left_attributes = _attributes(lanelet.leftBound)
                    right_attributes = _attributes(lanelet.rightBound)
                    median_separated, barrier_separated = (
                        _separation_semantics(
                            lane_attributes,
                            left_attributes,
                            right_attributes,
                        )
                    )
                    left_ids = self._related_lane_ids(lanelet, "left")
                    right_ids = self._related_lane_ids(lanelet, "right")
                    directions.append(
                        DirectedLaneField(
                            lane_id=canonical_id,
                            centerline_enu_m=centerline,
                            level=level,
                            road_class=_first_attribute(
                                lane_attributes,
                                "road_class",
                                "highway",
                                "road_type",
                            ),
                            lane_subtype=_first_attribute(
                                lane_attributes,
                                "lane_subtype",
                                "lane_type",
                                "subtype",
                            ),
                            one_way=_optional_bool(
                                lane_attributes,
                                "oneway",
                                "one_way",
                            ),
                            carriageway_id=_first_attribute(
                                lane_attributes,
                                "carriageway_id",
                                "carriageway",
                                "road_id",
                                "way_id",
                            ),
                            median_separated=median_separated,
                            barrier_separated=barrier_separated,
                            successor_lane_ids=self._related_lane_ids(
                                lanelet, "following"
                            ),
                            predecessor_lane_ids=self._related_lane_ids(
                                lanelet, "previous"
                            ),
                            left_adjacent_lane_id=(
                                left_ids[0] if left_ids else None
                            ),
                            right_adjacent_lane_id=(
                                right_ids[0] if right_ids else None
                            ),
                            is_intersection=self._is_intersection(lanelet),
                            turn_direction=_first_attribute(
                                lane_attributes, "turn_direction"
                            ),
                            provider_attributes=lane_attributes,
                            left_boundary_attributes=left_attributes,
                            right_boundary_attributes=right_attributes,
                        )
                    )

        stop_lines = tuple(
            PolylinePrimitive(
                primitive_id=f"lanelet2:stop_line:{index}",
                points_enu_m=points,
            )
            for index, points in enumerate(self.scene_map.get_stop_lines())
            if len(points) >= 2
        )
        signals = self._traffic_signals()

        all_points = [
            primitive.points_enu_m for primitive in drivable
        ]
        all_points.extend(
            primitive.points_enu_m for primitive in boundaries.values()
        )
        all_points.extend(
            primitive.points_enu_m for primitive in centerlines
        )
        all_points.extend(
            primitive.points_enu_m for primitive in crosswalks
        )
        all_points.extend(
            primitive.points_enu_m for primitive in stop_lines
        )
        all_points.extend(
            signal.position_enu_m.reshape(1, 3) for signal in signals
        )
        if not all_points:
            raise ValueError("Lanelet2 map contains no renderable primitives")
        stacked = np.concatenate(all_points, axis=0)
        bounds = (
            float(stacked[:, 0].min()),
            float(stacked[:, 1].min()),
            float(stacked[:, 0].max()),
            float(stacked[:, 1].max()),
        )

        frame = MapFrame(
            frame_id=self.frame_id,
            origin_latitude_deg=float(self.scene_map._origin_lat),
            origin_longitude_deg=float(self.scene_map._origin_lon),
            projection="EPSG:32632 local ENU",
        )
        levels_present = any(
            primitive.level is not None
            for primitive in (
                list(drivable)
                + list(boundaries.values())
                + list(centerlines)
            )
        )
        return NavigationMap(
            map_version=self.map_version,
            provider="lanelet2",
            frame=frame,
            bounds_enu_m=bounds,
            drivable_polygons=tuple(drivable),
            lane_boundaries=tuple(
                boundaries[key]
                for key in sorted(boundaries, key=lambda value: str(value))
            ),
            lane_centerlines=tuple(centerlines),
            intersection_polygons=tuple(intersections),
            crosswalk_polygons=tuple(crosswalks),
            stop_lines=stop_lines,
            static_traffic_signals=signals,
            directed_lane_fields=tuple(directions),
            layer_availability={
                "crosswalk": bool(crosswalks),
                "drivable_area": bool(drivable),
                "intersection": bool(intersections),
                "lane_boundary": bool(boundaries),
                "lane_centerline": bool(centerlines),
                "road_level": levels_present,
                "static_traffic_signal": bool(signals),
                "stop_line": bool(stop_lines),
                "traffic_direction": bool(directions),
                "lane_semantics": any(
                    lane.road_class is not None
                    or lane.one_way is not None
                    or lane.carriageway_id is not None
                    for lane in directions
                ),
                "lane_topology": bool(directions),
            },
            provenance={
                "adapter_version": LANELET2_ADAPTER_VERSION,
                "map_sha256": self.map_sha256,
                "source_revision": self.source_revision,
            },
        )
