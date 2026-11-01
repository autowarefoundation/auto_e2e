"""Map context and route mask from the scene's HD map, as in nuPlan training.

AlpaSim loads each scene's HD map from its USDZ as a trajdata ``VectorMap`` in
the scene's local frame, the frame of the ego poses. The driver opens the same
USDZ (the AlpaSim patch passes the scene id) and draws the map with the nuPlan
training rasterizer (``data_parsing/nuplan/targets.py``), so the model sees
the inputs it was trained on:

* map context: lanes stand in for nuPlan's roadblocks in the drivable area,
  lane centerlines and edges are the lane features, wait lines the stop lines.
  The NuRec maps carry no intersection areas, so that channel stays empty.
* route channel 0: nuPlan draws the route's roadblocks. Here the lanes under
  the route AlpaSim sends, together with their side-by-side neighbours, form
  the roadblocks.
* route channel 1: the nuPlan mission goal is where the logged drive ends;
  here it is the end of the recorded ego trajectory.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
from data_parsing.nuplan.targets import (
    NavigationMapLayers,
    destination_heatmap,
    global_to_ego,
    rasterize_navigation_map,
    rasterize_polygons,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY, ROUTE_CHANNEL_COUNT

logger = logging.getLogger(__name__)

SCENE_ROOT = Path("/mnt/nre-data")
"""Where the driver config mounts AlpaSim's scene cache in the driver container."""

_STOP_LINE_HALF_WIDTH_M = 0.25
"""nuPlan stop lines are polygons; NuRec wait lines are polylines."""


def _lane_polygon(lane: Any) -> np.ndarray | None:
    if lane.left_edge is None or lane.right_edge is None:
        return None
    return np.concatenate([lane.left_edge.xy, lane.right_edge.xy[::-1]])


def _segment_quads(polyline_xy: np.ndarray, half_width_m: float) -> list[np.ndarray]:
    quads = []
    for start, end in zip(polyline_xy[:-1], polyline_xy[1:]):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            continue
        normal = np.array([-direction[1], direction[0]]) / length * half_width_m
        quads.append(np.stack([start + normal, end + normal, end - normal, start - normal]))
    return quads


@dataclasses.dataclass(frozen=True)
class SceneMap:
    """One scene's map primitives, in the scene's local frame."""

    lanes: dict[str, Any]
    lane_polygons: dict[str, np.ndarray]
    layers: NavigationMapLayers
    destination_xy: np.ndarray
    vector_map: Any

    @classmethod
    def from_vector_map(cls, vector_map: Any, destination_xy: np.ndarray) -> SceneMap:
        from trajdata.maps.vec_map_elements import MapElementType

        elements = vector_map.elements
        lanes = dict(elements.get(MapElementType.ROAD_LANE, {}))
        lane_polygons = {
            lane_id: polygon
            for lane_id, lane in lanes.items()
            if (polygon := _lane_polygon(lane)) is not None
        }
        road_areas = [
            area.exterior_polygon.xy
            for area in elements.get(MapElementType.ROAD_AREA, {}).values()
        ]
        layers = NavigationMapLayers(
            drivable_area=[*lane_polygons.values(), *road_areas],
            crosswalk=[
                crosswalk.polygon.xy
                for crosswalk in elements.get(MapElementType.PED_CROSSWALK, {}).values()
            ],
            stop_line=[
                quad
                for wait_line in elements.get(MapElementType.WAIT_LINE, {}).values()
                for quad in _segment_quads(wait_line.polyline.xy, _STOP_LINE_HALF_WIDTH_M)
            ],
            lane_centerlines=[lane.center.xy for lane in lanes.values()],
            lane_boundaries=[
                edge.xy
                for lane in lanes.values()
                for edge in (lane.left_edge, lane.right_edge)
                if edge is not None
            ],
        )
        return cls(
            lanes=lanes,
            lane_polygons=lane_polygons,
            layers=layers,
            destination_xy=np.asarray(destination_xy, dtype=np.float64)[:2],
            vector_map=vector_map,
        )

    def map_context(self, reference_pose: tuple[float, float, float]) -> tuple[np.ndarray, bool]:
        return rasterize_navigation_map(self.layers, reference_pose, AUTOE2E_NAVIGATION_GEOMETRY)

    def roadblock_lanes(self, route_xyz: np.ndarray) -> set[str]:
        """Lanes under the route waypoints plus their side-by-side neighbours."""
        selected: set[str] = set()
        pending = [self.vector_map.get_closest_lane(point).id for point in route_xyz]
        while pending:
            lane_id = pending.pop()
            if lane_id in selected or lane_id not in self.lanes:
                continue
            selected.add(lane_id)
            lane = self.lanes[lane_id]
            pending.extend({*lane.adj_lanes_left, *lane.adj_lanes_right})
        return selected

    def route_mask(
        self, route_xyz: np.ndarray, reference_pose: tuple[float, float, float]
    ) -> tuple[np.ndarray, bool]:
        """``[2, H, W]`` route target and ``route_valid``, as nuPlan training builds them."""
        geometry = AUTOE2E_NAVIGATION_GEOMETRY
        mask = np.zeros((ROUTE_CHANNEL_COUNT, geometry.height_px, geometry.width_px), np.float32)
        corridor = [
            self.lane_polygons[lane_id]
            for lane_id in sorted(self.roadblock_lanes(route_xyz))
            if lane_id in self.lane_polygons
        ]
        if corridor:
            mask[0] = rasterize_polygons(corridor, reference_pose, geometry)
        goal = global_to_ego(self.destination_xy[None], reference_pose)[0]
        mask[1], destination_valid = destination_heatmap(goal, geometry)
        return mask, bool(corridor) or destination_valid


class SceneMaps:
    """Scene id to ``SceneMap``, loaded once per scene from the USDZ under ``root``."""

    def __init__(self, root: Path = SCENE_ROOT) -> None:
        self.root = root
        self._maps: dict[str, SceneMap | None] = {}

    def get(self, scene_id: str) -> SceneMap | None:
        if scene_id not in self._maps:
            self._maps[scene_id] = self._load(scene_id)
        return self._maps[scene_id]

    def _load(self, scene_id: str) -> SceneMap | None:
        from alpasim_utils.artifact import Artifact

        for path in sorted(self.root.rglob("*.usdz")):
            artifact = Artifact(str(path))
            if artifact.scene_id != scene_id:
                continue
            if artifact.map is None:
                break
            logger.info("AutoE2E driver: HD map of %s from %s", scene_id, path)
            return SceneMap.from_vector_map(artifact.map, artifact.rig.trajectory.positions[-1])
        logger.warning("AutoE2E driver: no HD map for scene %r under %s", scene_id, self.root)
        return None


__all__ = ["SceneMap", "SceneMaps"]
