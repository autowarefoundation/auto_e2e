"""Leak-resistant L2D navigation, built the way KITScenes builds its own.

One episode gets one ``NavigationMap`` cut out of the regional OSM lane
graph, one route matched from the recorded GPS trace onto that lane graph,
and sample rasters rendered by the shared native rasterizer on the 500 ms
anchor grid and warped to each frame. The dataset's own map image then adds
the road surface it draws to the drivable and known-area channels. Members
are the same four that KITScenes writes, so Reactive training reads L2D,
nuPlan, and KITScenes navigation identically.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from navigation.artifacts import encode_sample_navigation
from navigation.contracts import Maneuver, MapFrame
from navigation.geometry import MapChannel, RouteChannel
from navigation.osm_adapter import OSMMapAdapter
from navigation.rasterizer import EgoPose, NativeNavigationRasterizer, NavigationRaster
from navigation.supervision import build_route_supervision
from navigation.valhalla import LocalLaneSequenceResolver, ProviderRoute, _wgs84_to_local_enu

from .map_image import MAP_IMAGE_VERSION, bev_layers, register

L2D_NAVIGATION_VERSION = "l2d_osm_navigation_v1"
ANCHOR_PERIOD_NS = 500_000_000
MANEUVER_LOOKAHEAD_M = 100.0
# Lanes within this distance of the episode trace are kept for the episode map.
EPISODE_MAP_MARGIN_M = 250.0
# The route matcher compares every trace point with every lane, so the trace
# is thinned to one point per this many metres.
TRACE_STEP_M = 4.0


def episode_lane_graph(payload: Mapping[str, Any], trace_enu_m: np.ndarray, margin_m: float = EPISODE_MAP_MARGIN_M) -> dict[str, Any]:
    """The part of a regional lane graph within ``margin_m`` of the trace."""
    low = trace_enu_m[:, :2].min(axis=0) - margin_m
    high = trace_enu_m[:, :2].max(axis=0) + margin_m

    def inside(points: Any) -> bool:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return bool(np.any(np.all((values >= low) & (values <= high), axis=1)))

    lanes = [lane for lane in payload["lanes"] if inside(lane["centerline_enu_m"])]
    if not lanes:
        raise ValueError("no OSM lanes near the episode trace")
    kept = {f"{lane['way_id']}:{lane['direction']}:{lane['lane_index']}" for lane in lanes}
    trimmed = []
    for lane in lanes:
        lane = dict(lane)
        lane["successors"] = [item for item in lane.get("successors", []) if item in kept]
        for side in ("left_adjacent", "right_adjacent"):
            if lane.get(side) not in kept:
                lane[side] = None
        trimmed.append(lane)
    semantic = {
        key: [item for item in values if inside(item.get("points_enu_m", item.get("position_enu_m")))]
        for key, values in payload.get("semantic", {}).items()
    }
    return {**payload, "lanes": trimmed, "semantic": semantic}


def _yaw_enu(heading_deg_cw_from_north: np.ndarray) -> np.ndarray:
    return np.unwrap(np.radians(90.0 - np.asarray(heading_deg_cw_from_north, dtype=np.float64)))


def _thinned(points: np.ndarray, step_m: float) -> np.ndarray:
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index, :2] - points[keep[-1], :2]) >= step_m:
            keep.append(index)
    if keep[-1] != len(points) - 1:
        keep.append(len(points) - 1)
    return np.asarray(keep)


@dataclasses.dataclass(frozen=True)
class MapImageFusion:
    """What the dataset map image added to one sample raster."""

    registered: bool
    residual_px: float | None
    pixels_per_meter: float | None
    added_drivable_px: int
    drivable_iou: float | None


class L2DEpisodeNavigation:
    def __init__(
        self,
        *,
        episode_id: str,
        longitudes_deg: np.ndarray,
        latitudes_deg: np.ndarray,
        headings_deg_cw_from_north: np.ndarray,
        timestamps_ns: np.ndarray,
        lane_graph: Mapping[str, Any],
        lane_graph_sha256: str,
        rasterizer: NativeNavigationRasterizer,
    ) -> None:
        self.episode_id = str(episode_id)
        self.timestamps = np.ascontiguousarray(timestamps_ns, dtype=np.int64)
        latitudes = np.asarray(latitudes_deg, dtype=np.float64)
        longitudes = np.asarray(longitudes_deg, dtype=np.float64)
        count = len(self.timestamps)
        if count < 2 or len(latitudes) != count or len(longitudes) != count or len(headings_deg_cw_from_north) != count:
            raise ValueError("L2D navigation trace lengths differ or are too short")
        if not (np.isfinite(latitudes).all() and np.isfinite(longitudes).all()):
            raise ValueError("L2D navigation trace has non-finite GPS")
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError("L2D navigation timestamps are unordered")
        frame = lane_graph["frame"]
        self.frame = MapFrame(
            frame_id=str(frame["frame_id"]),
            origin_latitude_deg=float(frame["origin_latitude_deg"]),
            origin_longitude_deg=float(frame["origin_longitude_deg"]),
            projection=str(frame["projection"]),
        )
        self.positions = _wgs84_to_local_enu(np.column_stack([latitudes, longitudes]), self.frame)
        self.yaws = _yaw_enu(headings_deg_cw_from_north)
        self.lane_graph_sha256 = lane_graph_sha256
        self.adapter = OSMMapAdapter(episode_lane_graph(lane_graph, self.positions), source_sha256=lane_graph_sha256)
        self.navigation_map = self.adapter.extract()
        thinned = _thinned(self.positions, TRACE_STEP_M)
        self.route = LocalLaneSequenceResolver(self.adapter).resolve(
            ProviderRoute(
                request_id=f"l2d:{self.episode_id}",
                map_version=self.adapter.map_version,
                timestamp_ns=int(self.timestamps[0]),
                shape_wgs84=np.column_stack([latitudes[thinned], longitudes[thinned]]),
                maneuvers=(),
                provider="l2d_recorded_trace",
            ),
            revision=1,
        )
        self.rasterizer = rasterizer
        self._anchor_cache: dict[int, NavigationRaster] = {}
        self.episode_navigation_sha256 = hashlib.sha256(
            f"{L2D_NAVIGATION_VERSION}|{self.episode_id}|{lane_graph_sha256}|{self.route.route_id}".encode()
        ).hexdigest()

    def _pose(self, frame_idx: int) -> EgoPose:
        return EgoPose(
            x_enu_m=float(self.positions[frame_idx, 0]),
            y_enu_m=float(self.positions[frame_idx, 1]),
            yaw_rad=float(self.yaws[frame_idx]),
            timestamp_ns=int(self.timestamps[frame_idx]),
        )

    def anchor_index(self, frame_idx: int) -> int:
        """Latest non-future pose on the episode-relative 500 ms grid."""
        first = int(self.timestamps[0])
        anchor = first + ((int(self.timestamps[frame_idx]) - first) // ANCHOR_PERIOD_NS) * ANCHOR_PERIOD_NS
        return max(0, int(np.searchsorted(self.timestamps, anchor, side="right")) - 1)

    def raster_for_frame(self, frame_idx: int) -> NavigationRaster:
        anchor_idx = self.anchor_index(frame_idx)
        anchor = self._anchor_cache.get(anchor_idx)
        if anchor is None:
            anchor = self.rasterizer.render(self.navigation_map, self.route, self._pose(anchor_idx))
            self._anchor_cache[anchor_idx] = anchor
        if anchor_idx == frame_idx:
            return anchor
        return self.rasterizer.warp(anchor, self._pose(frame_idx))

    def fuse_map_image(
        self,
        raster: NavigationRaster,
        frame_idx: int,
        map_image: np.ndarray | None,
        waypoints_lon_lat: np.ndarray | None,
    ) -> tuple[NavigationRaster, MapImageFusion]:
        """Add the road surface the dataset map image draws, where it registers."""
        none = MapImageFusion(False, None, None, 0, None)
        if map_image is None or waypoints_lon_lat is None:
            return raster, none
        waypoints = _wgs84_to_local_enu(np.column_stack([waypoints_lon_lat[:, 1], waypoints_lon_lat[:, 0]]), self.frame)
        registration = register(map_image, waypoints[:, :2] - self.positions[frame_idx, :2])
        if registration is None:
            return raster, none
        forward, left = self.rasterizer.geometry.pixel_center_grids()
        road, footprint = bev_layers(map_image, registration, yaw_rad=float(self.yaws[frame_idx]), forward_m=forward, left_m=left)
        context = np.array(raster.map_context, copy=True)
        osm = context[MapChannel.DRIVABLE_AREA] > 0.0
        added = road & footprint & ~osm
        context[MapChannel.DRIVABLE_AREA][added] = 1.0
        context[MapChannel.KNOWN_MAP_AREA][footprint] = 1.0
        union = int(((osm | road) & footprint).sum())
        iou = float((osm & road & footprint).sum() / union) if union else None
        fused = dataclasses.replace(raster, map_context=context)
        return fused, MapImageFusion(True, registration.residual_px, registration.pixels_per_meter, int(added.sum()), iou)

    def route_semantics(self, frame_idx: int, raster: NavigationRaster) -> dict[str, object]:
        """Route-derived labels without future ego points (as KITScenes)."""
        if not self.route.valid or not self.route.lane_sequence:
            return {"route_maneuver": Maneuver.UNKNOWN.value, "route_intersection": False,
                    "destination_visible": False, "current_route_lane_id": "",
                    "maneuver_lookahead_m": MANEUVER_LOOKAHEAD_M}
        position = self.positions[frame_idx, :2]
        distances = [
            float(np.min(np.linalg.norm(segment.centerline_enu_m[:, :2] - position, axis=1)))
            for segment in self.route.lane_sequence
        ]
        active = int(np.argmin(distances))
        maneuver, travelled = Maneuver.STRAIGHT, 0.0
        for segment in self.route.lane_sequence[active:]:
            if travelled > MANEUVER_LOOKAHEAD_M:
                break
            if segment.maneuver in (Maneuver.LEFT, Maneuver.RIGHT, Maneuver.U_TURN, Maneuver.MERGE, Maneuver.EXIT):
                maneuver = segment.maneuver
                break
            travelled += float(np.linalg.norm(np.diff(segment.centerline_enu_m[:, :2], axis=0), axis=1).sum())
        x_forward, _ = self.rasterizer.geometry.pixel_center_grids()
        ahead = (x_forward >= 0.0) & (x_forward <= MANEUVER_LOOKAHEAD_M)
        return {
            "route_maneuver": maneuver.value,
            "route_intersection": bool(np.any((raster.map_context[MapChannel.INTERSECTION] > 0.0)
                                              & (raster.route_mask[RouteChannel.SELECTED_CORRIDOR] > 0) & ahead)),
            "destination_visible": bool(np.any(raster.route_mask[RouteChannel.DESTINATION] > 0)),
            "current_route_lane_id": self.route.lane_sequence[active].lane_id,
            "maneuver_lookahead_m": MANEUVER_LOOKAHEAD_M,
        }

    def sample_members(
        self,
        frame_idx: int,
        *,
        map_image: np.ndarray | None = None,
        waypoints_lon_lat: np.ndarray | None = None,
    ) -> dict[str, bytes]:
        raster = self.raster_for_frame(frame_idx)
        raster, fusion = self.fuse_map_image(raster, frame_idx, map_image, waypoints_lon_lat)
        quality = self.route.quality
        return encode_sample_navigation(
            raster,
            route_supervision=build_route_supervision(self.route, self._pose(frame_idx), raster, self.rasterizer.geometry),
            extra_metadata={
                "map_source": "osm_lane_graph+l2d_map_image",
                "l2d_navigation_version": L2D_NAVIGATION_VERSION,
                "osm_lane_graph_sha256": self.lane_graph_sha256,
                "episode_navigation_sha256": self.episode_navigation_sha256,
                "map_image_version": MAP_IMAGE_VERSION,
                "map_image_registered": fusion.registered,
                "map_image_residual_px": fusion.residual_px,
                "map_image_pixels_per_meter": fusion.pixels_per_meter,
                "map_image_added_drivable_px": fusion.added_drivable_px,
                "map_image_drivable_iou": fusion.drivable_iou,
                "route_quality_matched_pose_ratio": quality.matched_pose_ratio,
                "route_quality_median_lateral_distance_m": quality.median_lateral_distance_m,
                "route_quality_p95_lateral_distance_m": quality.p95_lateral_distance_m,
                "route_quality_median_heading_error_rad": quality.median_heading_error_rad,
                "route_quality_p95_heading_error_rad": quality.p95_heading_error_rad,
                "route_quality_shortest_path_fill_count": quality.shortest_path_fill_count,
                "route_quality_shortest_path_fill_length_m": quality.shortest_path_fill_length_m,
                "route_quality_adjacent_transition_count": quality.adjacent_transition_count,
                "route_quality_unresolved_discontinuities": quality.unresolved_discontinuities,
                "route_quality_failure_reasons": "|".join(quality.failure_reasons),
                **self.route_semantics(frame_idx, raster),
            },
        )
