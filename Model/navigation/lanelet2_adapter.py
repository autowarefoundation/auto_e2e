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


LANELET2_ADAPTER_VERSION = "lanelet2_adapter_v1"


def _attr(obj: Any, name: str, default: str = "") -> str:
    attributes = getattr(obj, "attributes", {})
    try:
        return str(attributes[name]) if name in attributes else default
    except (KeyError, TypeError):
        return default


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
                    directions.append(
                        DirectedLaneField(
                            lane_id=canonical_id,
                            centerline_enu_m=centerline,
                            level=level,
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
            },
            provenance={
                "adapter_version": LANELET2_ADAPTER_VERSION,
                "map_sha256": self.map_sha256,
                "source_revision": self.source_revision,
            },
        )
