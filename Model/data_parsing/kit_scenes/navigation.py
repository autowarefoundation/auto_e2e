"""Leak-resistant KITScenes navigation generation for shard preprocessing."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from navigation.artifacts import (
    encode_array,
    encode_sample_navigation,
    encode_scene_navigation,
)
from navigation.contracts import (
    Maneuver,
    canonical_json_bytes,
    contract_sha256,
)
from navigation.geometry import MapChannel, RouteChannel
from navigation.lanelet2_adapter import Lanelet2MapAdapter, file_sha256
from navigation.lanelet2_matcher import Lanelet2TraceMatcher
from navigation.quality import DEFAULT_NAVIGATION_QUALITY_POLICY
from navigation.rasterizer import (
    EgoPose,
    NativeNavigationRasterizer,
    NavigationRaster,
)
from navigation.supervision import build_route_supervision

ANCHOR_PERIOD_NS = 500_000_000
MANEUVER_LOOKAHEAD_M = 100.0
KITSCENES_NAVIGATION_VERSION = "kitscenes_navigation_v1"


def _polyline_length(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1).sum())


def _point_to_polyline_distance(
    point_xy: np.ndarray,
    points: np.ndarray,
) -> float:
    line = np.asarray(points, dtype=np.float64)[:, :2]
    point = np.asarray(point_xy, dtype=np.float64)[:2]
    starts = line[:-1]
    vectors = line[1:] - starts
    lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
    offsets = point - starts
    parameters = np.divide(
        np.einsum("ij,ij->i", offsets, vectors),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts + parameters[:, None] * vectors
    return float(np.linalg.norm(closest - point, axis=1).min())


@dataclasses.dataclass(frozen=True)
class SceneNavigationArtifacts:
    """Scene-level members stored beside one partition's sample shards."""

    scene_navigation: bytes
    scene_navigation_geometry: bytes
    navigation_quality: bytes

    def members(self) -> dict[str, bytes]:
        return {
            "scene_navigation.json": self.scene_navigation,
            "scene_navigation_geometry.npz": self.scene_navigation_geometry,
            "navigation_quality.json": self.navigation_quality,
        }


class KitScenesSceneNavigation:
    """Build one route per scene and render timestamp-aligned sample rasters."""

    def __init__(
        self,
        *,
        scene_id: str,
        scene_path: str | Path,
        positions_enu_m: np.ndarray,
        yaws_rad: np.ndarray,
        timestamps_ns: np.ndarray,
        source_revision: str,
        rasterizer: NativeNavigationRasterizer | None = None,
    ) -> None:
        self.scene_id = str(scene_id)
        self.scene_path = Path(scene_path)
        self.positions = np.ascontiguousarray(
            positions_enu_m, dtype=np.float64
        )
        self.yaws = np.ascontiguousarray(yaws_rad, dtype=np.float64)
        self.timestamps = np.ascontiguousarray(timestamps_ns, dtype=np.int64)
        self._validate_trace()

        map_path = self.scene_path / "maps" / "map.osm"
        if not map_path.is_file():
            raise FileNotFoundError(f"KITScenes map is missing: {map_path}")
        self.map_sha256 = file_sha256(map_path)
        map_version = (
            f"kitscenes:{self.scene_id}:{self.map_sha256[:16]}"
        )
        from .map import _cached_scene_map

        scene_map = _cached_scene_map(self.scene_path)
        if scene_map is None:
            raise ValueError(
                f"KITScenes scene {self.scene_id!r} has no loadable map"
            )
        self.navigation_map = Lanelet2MapAdapter(
            scene_map,
            map_version=map_version,
            map_sha256=self.map_sha256,
            frame_id=f"kitscenes:{self.scene_id}:local_enu",
            source_revision=source_revision,
        ).extract()
        self.route = Lanelet2TraceMatcher(
            scene_map,
            self.navigation_map,
            map_sha256=self.map_sha256,
            source_revision=source_revision,
        ).match(
            scene_id=self.scene_id,
            positions_enu_m=self.positions,
            yaws_rad=self.yaws,
            timestamps_ns=self.timestamps,
        )
        self.rasterizer = rasterizer or NativeNavigationRasterizer()
        self._scene_navigation_payload = encode_scene_navigation(
            self.navigation_map,
            self.route,
        )
        self._scene_navigation_sha256 = hashlib.sha256(
            self._scene_navigation_payload
        ).hexdigest()
        self._anchor_cache: dict[int, NavigationRaster] = {}

    def _validate_trace(self) -> None:
        count = len(self.positions)
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] not in (2, 3)
            or count == 0
        ):
            raise ValueError("positions_enu_m must have shape [N,2] or [N,3]")
        if self.yaws.shape != (count,) or self.timestamps.shape != (count,):
            raise ValueError("KITScenes navigation trace lengths differ")
        if not np.isfinite(self.positions).all():
            raise ValueError("KITScenes navigation positions are non-finite")
        if not np.isfinite(self.yaws).all():
            raise ValueError("KITScenes navigation yaws are non-finite")
        if np.any(self.timestamps < 0) or np.any(np.diff(self.timestamps) < 0):
            raise ValueError("KITScenes navigation timestamps are unordered")

    def _pose(self, frame_idx: int) -> EgoPose:
        if frame_idx < 0 or frame_idx >= len(self.timestamps):
            raise IndexError(
                f"frame {frame_idx} outside scene trace "
                f"[0,{len(self.timestamps)})"
            )
        return EgoPose(
            x_enu_m=float(self.positions[frame_idx, 0]),
            y_enu_m=float(self.positions[frame_idx, 1]),
            yaw_rad=float(self.yaws[frame_idx]),
            timestamp_ns=int(self.timestamps[frame_idx]),
        )

    def anchor_index(self, frame_idx: int) -> int:
        """Return the latest non-future pose on the scene-relative 500 ms grid."""
        sample_timestamp = int(self.timestamps[frame_idx])
        first_timestamp = int(self.timestamps[0])
        anchor_timestamp = (
            first_timestamp
            + ((sample_timestamp - first_timestamp) // ANCHOR_PERIOD_NS)
            * ANCHOR_PERIOD_NS
        )
        return max(
            0,
            int(
                np.searchsorted(
                    self.timestamps,
                    anchor_timestamp,
                    side="right",
                )
            )
            - 1,
        )

    def raster_for_frame(self, frame_idx: int) -> NavigationRaster:
        anchor_idx = self.anchor_index(frame_idx)
        anchor = self._anchor_cache.get(anchor_idx)
        if anchor is None:
            anchor = self.rasterizer.render(
                self.navigation_map,
                self.route,
                self._pose(anchor_idx),
            )
            self._anchor_cache[anchor_idx] = anchor
        if anchor_idx == frame_idx:
            return anchor
        return self.rasterizer.warp(anchor, self._pose(frame_idx))

    def route_semantics(
        self,
        frame_idx: int,
        raster: NavigationRaster,
    ) -> dict[str, object]:
        """Return route-derived evaluation labels without future ego points."""
        if not self.route.valid or not self.route.lane_sequence:
            return {
                "route_maneuver": Maneuver.UNKNOWN.value,
                "route_intersection": False,
                "destination_visible": False,
                "current_route_lane_id": "",
                "maneuver_lookahead_m": MANEUVER_LOOKAHEAD_M,
            }

        position = self.positions[frame_idx, :2]
        active_index = min(
            range(len(self.route.lane_sequence)),
            key=lambda index: _point_to_polyline_distance(
                position,
                self.route.lane_sequence[index].centerline_enu_m,
            ),
        )
        maneuver = Maneuver.STRAIGHT
        distance = 0.0
        decisive = {
            Maneuver.LEFT,
            Maneuver.RIGHT,
            Maneuver.U_TURN,
            Maneuver.MERGE,
            Maneuver.EXIT,
        }
        for segment in self.route.lane_sequence[active_index:]:
            if distance > MANEUVER_LOOKAHEAD_M:
                break
            if segment.maneuver in decisive:
                maneuver = segment.maneuver
                break
            distance += _polyline_length(segment.centerline_enu_m)

        geometry = self.rasterizer.geometry
        x_forward, _ = geometry.pixel_center_grids()
        route_intersection = bool(
            np.any(
                (raster.map_context[MapChannel.INTERSECTION] > 0.0)
                & (
                    raster.route_mask[RouteChannel.SELECTED_CORRIDOR]
                    > 0
                )
                & (x_forward >= 0.0)
                & (x_forward <= MANEUVER_LOOKAHEAD_M)
            )
        )
        return {
            "route_maneuver": maneuver.value,
            "route_intersection": route_intersection,
            "destination_visible": bool(
                np.any(
                    raster.route_mask[RouteChannel.DESTINATION] > 0
                )
            ),
            "current_route_lane_id": (
                self.route.lane_sequence[active_index].lane_id
            ),
            "maneuver_lookahead_m": MANEUVER_LOOKAHEAD_M,
        }

    def sample_members(self, frame_idx: int) -> dict[str, bytes]:
        raster = self.raster_for_frame(frame_idx)
        quality = self.route.quality
        return encode_sample_navigation(
            raster,
            route_supervision=build_route_supervision(
                self.route,
                self._pose(frame_idx),
                raster,
                self.rasterizer.geometry,
            ),
            extra_metadata={
                "scene_navigation_sha256": (
                    self._scene_navigation_sha256
                ),
                "route_quality_matched_pose_ratio": (
                    quality.matched_pose_ratio
                ),
                "route_quality_median_lateral_distance_m": (
                    quality.median_lateral_distance_m
                ),
                "route_quality_p95_lateral_distance_m": (
                    quality.p95_lateral_distance_m
                ),
                "route_quality_median_heading_error_rad": (
                    quality.median_heading_error_rad
                ),
                "route_quality_p95_heading_error_rad": (
                    quality.p95_heading_error_rad
                ),
                "route_quality_shortest_path_fill_count": (
                    quality.shortest_path_fill_count
                ),
                "route_quality_shortest_path_fill_length_m": (
                    quality.shortest_path_fill_length_m
                ),
                "route_quality_adjacent_transition_count": (
                    quality.adjacent_transition_count
                ),
                "route_quality_unresolved_discontinuities": (
                    quality.unresolved_discontinuities
                ),
                "route_quality_failure_reasons": "|".join(
                    quality.failure_reasons
                ),
                **self.route_semantics(frame_idx, raster),
            },
        )

    def artifacts(self) -> SceneNavigationArtifacts:
        geometry = self.rasterizer.geometry
        geometry_values = np.asarray(
            [
                geometry.height_px,
                geometry.width_px,
                geometry.meters_per_pixel,
                geometry.x_min_m,
                geometry.x_max_m,
                geometry.y_min_m,
                geometry.y_max_m,
                geometry.ego_anchor_row,
                geometry.ego_anchor_col,
                geometry.route_corridor_width_m,
                geometry.destination_marker_radius_m,
                geometry.route_rear_clip_m,
            ],
            dtype=np.float64,
        )
        quality_payload = canonical_json_bytes(
            {
                "schema_version": KITSCENES_NAVIGATION_VERSION,
                "scene_id": self.scene_id,
                "geometry_id": geometry.geometry_id,
                "map_sha256": self.map_sha256,
                "map_contract_sha256": contract_sha256(
                    self.navigation_map
                ),
                "route_contract_sha256": contract_sha256(self.route),
                "route_valid": self.route.valid,
                "route_confidence": self.route.confidence,
                "quality": self.route.quality,
                "quality_policy": (
                    DEFAULT_NAVIGATION_QUALITY_POLICY.contract()
                ),
                "estimated_destination": self.route.estimated_destination,
                "destination_source": self.route.destination.source,
                "anchor_period_ns": ANCHOR_PERIOD_NS,
                "sample_count": len(self.timestamps),
            }
        )
        return SceneNavigationArtifacts(
            scene_navigation=self._scene_navigation_payload,
            scene_navigation_geometry=encode_array(geometry_values),
            navigation_quality=quality_payload,
        )

    @property
    def scene_navigation_sha256(self) -> str:
        return self._scene_navigation_sha256


def build_scene_navigation(
    *,
    scene_id: str,
    scene_path: str | Path,
    positions_enu_m: np.ndarray,
    yaws_rad: np.ndarray,
    timestamps_ns: np.ndarray,
    source_revision: str,
    rasterizer: NativeNavigationRasterizer | None = None,
) -> KitScenesSceneNavigation:
    """Construct the complete deterministic navigation state for one scene."""
    return KitScenesSceneNavigation(
        scene_id=scene_id,
        scene_path=scene_path,
        positions_enu_m=positions_enu_m,
        yaws_rad=yaws_rad,
        timestamps_ns=timestamps_ns,
        source_revision=source_revision,
        rasterizer=rasterizer,
    )
