"""Thread-safe runtime scheduling for Reactive navigation rasters."""

from __future__ import annotations

import dataclasses
import threading
from typing import Protocol

import numpy as np

from .contracts import NavigationMap, NavigationRoute
from .geometry import (
    LOCALIZATION_HZ,
    MAP_CHANNEL_COUNT,
    RASTER_DECIMATION,
    ROUTE_CHANNEL_COUNT,
)
from .rasterizer import EgoPose, NavigationRaster


class NavigationRasterizer(Protocol):
    geometry: object
    renderer_version: str

    def render(
        self,
        navigation_map: NavigationMap | None,
        route: NavigationRoute | None,
        pose: EgoPose,
    ) -> NavigationRaster: ...

    def warp(
        self,
        raster: NavigationRaster,
        sample_pose: EgoPose,
    ) -> NavigationRaster: ...


@dataclasses.dataclass(frozen=True)
class NavigationRuntimeConfig:
    localization_hz: int = LOCALIZATION_HZ
    raster_decimation: int = RASTER_DECIMATION
    maximum_raster_age_ns: int = 750_000_000
    maximum_future_skew_ns: int = 0

    def __post_init__(self) -> None:
        if self.localization_hz <= 0 or self.raster_decimation <= 0:
            raise ValueError("runtime frequencies must be positive")
        if self.maximum_raster_age_ns <= 0:
            raise ValueError("maximum_raster_age_ns must be positive")
        if self.maximum_future_skew_ns < 0:
            raise ValueError("maximum_future_skew_ns must be non-negative")


@dataclasses.dataclass
class NavigationRuntimeDiagnostics:
    localization_updates: int = 0
    render_attempts: int = 0
    render_successes: int = 0
    render_failures: int = 0
    warp_failures: int = 0
    stale_rasters: int = 0
    future_rasters: int = 0


class AtomicRouteStore:
    """Publish only complete valid route revisions for one map version."""

    def __init__(self, map_version: str) -> None:
        if not map_version:
            raise ValueError("map_version must not be empty")
        self.map_version = map_version
        self._lock = threading.Lock()
        self._route: NavigationRoute | None = None

    def snapshot(self) -> NavigationRoute | None:
        with self._lock:
            return self._route

    def commit(self, route: NavigationRoute) -> bool:
        """Atomically replace the route, retaining the old route on rejection."""
        if not route.valid or route.map_version != self.map_version:
            return False
        with self._lock:
            current = self._route
            if current is not None and route.revision <= current.revision:
                return False
            self._route = route
            return True


class RuntimeNavigationScheduler:
    """Render at 2 Hz and warp the latest complete raster at model timestamps."""

    def __init__(
        self,
        navigation_map: NavigationMap,
        route_store: AtomicRouteStore,
        rasterizer: NavigationRasterizer,
        *,
        config: NavigationRuntimeConfig | None = None,
    ) -> None:
        if route_store.map_version != navigation_map.map_version:
            raise ValueError("route store and navigation map versions differ")
        self.navigation_map = navigation_map
        self.route_store = route_store
        self.rasterizer = rasterizer
        self.config = config or NavigationRuntimeConfig()
        self.diagnostics = NavigationRuntimeDiagnostics()
        self._raster_lock = threading.Lock()
        self._latest_raster: NavigationRaster | None = None

    def latest_raster(self) -> NavigationRaster | None:
        with self._raster_lock:
            return self._latest_raster

    def on_localization(self, pose: EgoPose) -> bool:
        """Consume one 20 Hz localization update and render on decimated ticks."""
        tick = self.diagnostics.localization_updates
        self.diagnostics.localization_updates += 1
        if tick % self.config.raster_decimation:
            return False

        self.diagnostics.render_attempts += 1
        route = self.route_store.snapshot()
        try:
            candidate = self.rasterizer.render(
                self.navigation_map,
                route,
                pose,
            )
        except Exception:
            self.diagnostics.render_failures += 1
            return False

        with self._raster_lock:
            self._latest_raster = candidate
        self.diagnostics.render_successes += 1
        return True

    def for_camera(self, pose: EgoPose) -> NavigationRaster:
        """Return the latest raster warped into one camera/model timestamp."""
        raster = self.latest_raster()
        if raster is None:
            return self._invalid_raster(pose, None)

        age_ns = pose.timestamp_ns - raster.render_pose.timestamp_ns
        if age_ns < -self.config.maximum_future_skew_ns:
            self.diagnostics.future_rasters += 1
            return self._invalid_raster(pose, raster)
        if age_ns > self.config.maximum_raster_age_ns:
            self.diagnostics.stale_rasters += 1
            return self._invalid_raster(pose, raster)
        try:
            return self.rasterizer.warp(raster, pose)
        except Exception:
            self.diagnostics.warp_failures += 1
            return self._invalid_raster(pose, raster)

    def _invalid_raster(
        self,
        pose: EgoPose,
        previous: NavigationRaster | None,
    ) -> NavigationRaster:
        geometry = self.rasterizer.geometry
        height = int(getattr(geometry, "height_px"))
        width = int(getattr(geometry, "width_px"))
        render_pose = previous.render_pose if previous is not None else pose
        return NavigationRaster(
            map_context=np.zeros(
                (MAP_CHANNEL_COUNT, height, width),
                dtype=np.float32,
            ),
            route_mask=np.zeros(
                (ROUTE_CHANNEL_COUNT, height, width),
                dtype=np.uint8,
            ),
            map_valid=False,
            route_valid=False,
            geometry_id=str(getattr(geometry, "geometry_id")),
            render_pose=render_pose,
            sample_pose=pose,
            renderer_version=self.rasterizer.renderer_version,
            map_version=self.navigation_map.map_version,
            route_id=previous.route_id if previous is not None else "",
            route_revision=(
                previous.route_revision if previous is not None else 0
            ),
            route_confidence=0.0,
            input_vector_sha256=(
                previous.input_vector_sha256 if previous is not None else ""
            ),
        )
