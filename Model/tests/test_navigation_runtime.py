"""Runtime navigation scheduling and route revision tests."""

from __future__ import annotations

import types

import numpy as np

from navigation.contracts import (
    Destination,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
)
from navigation.rasterizer import EgoPose, NavigationRaster
from navigation.runtime import AtomicRouteStore, RuntimeNavigationScheduler


def _map() -> NavigationMap:
    return NavigationMap(
        map_version="map-v1",
        provider="fixture",
        frame=MapFrame("fixture", 49.0, 8.0, "local ENU"),
        bounds_enu_m=(-100.0, -100.0, 100.0, 100.0),
    )


def _route(revision: int, *, valid: bool = True) -> NavigationRoute:
    segment = RouteLaneSegment(
        lane_id="lane-1",
        provider_segment_id="provider-lane-1",
        centerline_enu_m=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
    )
    return NavigationRoute(
        route_id="route-1",
        revision=revision,
        provider="fixture",
        timestamp_ns=revision,
        valid_from_ns=revision,
        map_version="map-v1",
        frame=_map().frame,
        lane_sequence=(segment,),
        destination=Destination(np.asarray([20.0, 0.0]), "fixture"),
        confidence=1.0,
        valid=valid,
        quality=RouteQuality(1.0, 0.0, 0.0, 0.0, 0.0),
        estimated_destination=False,
        provenance=RouteProvenance(
            "source",
            "matcher",
            "matcher-config",
            "map-sha",
            "trace-sha",
        ),
    )


class _Rasterizer:
    renderer_version = "fixture-v1"

    def __init__(self):
        self.geometry = types.SimpleNamespace(
            geometry_id="fixture-geometry",
            height_px=8,
            width_px=8,
        )
        self.render_calls = []
        self.warp_calls = []
        self.fail_render = False

    def render(self, navigation_map, route, pose):
        self.render_calls.append((route, pose))
        if self.fail_render:
            raise RuntimeError("render failed")
        route_valid = route is not None and route.valid
        return NavigationRaster(
            map_context=np.zeros((14, 8, 8), dtype=np.float32),
            route_mask=np.full(
                (2, 8, 8),
                int(route_valid),
                dtype=np.uint8,
            ),
            map_valid=True,
            route_valid=route_valid,
            geometry_id=self.geometry.geometry_id,
            render_pose=pose,
            sample_pose=pose,
            renderer_version=self.renderer_version,
            map_version=navigation_map.map_version,
            route_id=route.route_id if route_valid else "",
            route_revision=route.revision if route_valid else 0,
            route_confidence=route.confidence if route_valid else 0.0,
            input_vector_sha256="fixture",
        )

    def warp(self, raster, pose):
        self.warp_calls.append((raster, pose))
        return NavigationRaster(
            map_context=raster.map_context,
            route_mask=raster.route_mask,
            map_valid=raster.map_valid,
            route_valid=raster.route_valid,
            geometry_id=raster.geometry_id,
            render_pose=raster.render_pose,
            sample_pose=pose,
            renderer_version=raster.renderer_version,
            map_version=raster.map_version,
            route_id=raster.route_id,
            route_revision=raster.route_revision,
            route_confidence=raster.route_confidence,
            input_vector_sha256=raster.input_vector_sha256,
        )


def _pose(tick: int) -> EgoPose:
    return EgoPose(
        x_enu_m=tick * 0.1,
        y_enu_m=0.0,
        yaw_rad=0.0,
        timestamp_ns=tick * 50_000_000,
    )


def test_atomic_route_store_retains_previous_complete_revision():
    store = AtomicRouteStore("map-v1")

    assert store.commit(_route(1))
    assert not store.commit(_route(1))
    assert not store.commit(_route(2, valid=False))
    assert store.snapshot().revision == 1
    assert store.commit(_route(2))
    assert store.snapshot().revision == 2


def test_scheduler_renders_every_tenth_localization_and_warps_for_camera():
    navigation_map = _map()
    store = AtomicRouteStore(navigation_map.map_version)
    assert store.commit(_route(1))
    rasterizer = _Rasterizer()
    scheduler = RuntimeNavigationScheduler(
        navigation_map,
        store,
        rasterizer,
    )

    rendered = [
        scheduler.on_localization(_pose(tick))
        for tick in range(20)
    ]

    assert [index for index, value in enumerate(rendered) if value] == [0, 10]
    assert scheduler.diagnostics.render_successes == 2
    camera_raster = scheduler.for_camera(_pose(19))
    assert camera_raster.route_valid
    assert camera_raster.route_revision == 1
    assert camera_raster.render_pose.timestamp_ns == _pose(10).timestamp_ns
    assert camera_raster.sample_pose.timestamp_ns == _pose(19).timestamp_ns


def test_reroute_and_render_failure_keep_previous_complete_raster():
    navigation_map = _map()
    store = AtomicRouteStore(navigation_map.map_version)
    assert store.commit(_route(1))
    rasterizer = _Rasterizer()
    scheduler = RuntimeNavigationScheduler(
        navigation_map,
        store,
        rasterizer,
    )
    assert scheduler.on_localization(_pose(0))

    assert store.commit(_route(2))
    rasterizer.fail_render = True
    for tick in range(1, 11):
        scheduler.on_localization(_pose(tick))

    retained = scheduler.for_camera(_pose(12))
    assert retained.route_valid
    assert retained.route_revision == 1
    assert scheduler.diagnostics.render_failures == 1


def test_stale_or_future_raster_is_explicitly_invalid():
    navigation_map = _map()
    store = AtomicRouteStore(navigation_map.map_version)
    assert store.commit(_route(1))
    rasterizer = _Rasterizer()
    scheduler = RuntimeNavigationScheduler(
        navigation_map,
        store,
        rasterizer,
    )
    assert scheduler.on_localization(_pose(20))

    future = scheduler.for_camera(_pose(19))
    assert not future.map_valid
    assert not future.route_valid
    assert scheduler.diagnostics.future_rasters == 1

    stale = scheduler.for_camera(_pose(36))
    assert not stale.map_valid
    assert not stale.route_valid
    assert scheduler.diagnostics.stale_rasters == 1
