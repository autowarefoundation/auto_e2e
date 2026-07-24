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
    canonical_json_bytes,
    contract_sha256,
)
from navigation.lanelet2_adapter import Lanelet2MapAdapter, file_sha256
from navigation.lanelet2_matcher import Lanelet2TraceMatcher
from navigation.rasterizer import (
    EgoPose,
    NativeNavigationRasterizer,
    NavigationRaster,
)

ANCHOR_PERIOD_NS = 500_000_000
KITSCENES_NAVIGATION_VERSION = "kitscenes_navigation_v1"


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

    def sample_members(self, frame_idx: int) -> dict[str, bytes]:
        return encode_sample_navigation(
            self.raster_for_frame(frame_idx),
            extra_metadata={
                "scene_navigation_sha256": (
                    self._scene_navigation_sha256
                ),
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
