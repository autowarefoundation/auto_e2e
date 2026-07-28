"""Leakage and geometry tests for selected-route training supervision."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("scipy")

from navigation.contracts import (
    Destination,
    MapFrame,
    NavigationRoute,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
)
from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
    RouteChannel,
)
from navigation.rasterizer import EgoPose, NavigationRaster
from navigation.supervision import (
    MAXIMUM_OUTSIDE_DISTANCE_M,
    build_route_supervision,
)


def _route(*, valid: bool = True) -> NavigationRoute:
    return NavigationRoute(
        route_id="route-1",
        revision=1,
        provider="fixture",
        timestamp_ns=0,
        valid_from_ns=0,
        map_version="map-1",
        frame=MapFrame("local-enu", 0.0, 0.0, "enu"),
        lane_sequence=(
            RouteLaneSegment(
                lane_id="lane-1",
                provider_segment_id="lane-1",
                centerline_enu_m=np.asarray(
                    [[-10.0, 0.0], [100.0, 0.0]],
                ),
            ),
        )
        if valid
        else (),
        destination=Destination(
            position_enu_m=np.asarray([100.0, 0.0]),
            source="scene_end",
        ),
        confidence=1.0 if valid else 0.0,
        valid=valid,
        quality=RouteQuality(1.0, 0.0, 0.0, 0.0, 0.0),
        estimated_destination=True,
        provenance=RouteProvenance(
            source_revision="source",
            matcher_version="matcher",
            matcher_config_sha256="config",
            map_sha256="map",
            trace_sha256="trace",
        ),
    )


def _raster(
    *,
    map_valid: bool = True,
    route_valid: bool = True,
) -> NavigationRaster:
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    map_context = np.zeros(
        (14, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    route_mask = np.zeros(
        (2, geometry.height_px, geometry.width_px),
        dtype=np.uint8,
    )
    points = np.column_stack(
        [np.linspace(-10.0, 100.0, 221), np.zeros(221)]
    )
    pixels = np.rint(geometry.ego_to_pixel(points)).astype(np.int64)
    for row, col in pixels:
        map_context[
            MapChannel.DRIVABLE_AREA,
            max(0, row - 5):min(geometry.height_px, row + 6),
            max(0, col - 5):min(geometry.width_px, col + 6),
        ] = 1.0
        route_mask[
            RouteChannel.SELECTED_CORRIDOR,
            max(0, row - 2):min(geometry.height_px, row + 3),
            max(0, col - 2):min(geometry.width_px, col + 3),
        ] = 1
    route_mask[
        RouteChannel.DESTINATION,
        pixels[-1, 0],
        pixels[-1, 1],
    ] = 1
    pose = EgoPose(0.0, 0.0, 0.0, 0)
    return NavigationRaster(
        map_context=map_context,
        route_mask=route_mask,
        map_valid=map_valid,
        route_valid=route_valid,
        geometry_id=geometry.geometry_id,
        render_pose=pose,
        sample_pose=pose,
        renderer_version="fixture",
        map_version="map-1",
        route_id="route-1" if route_valid else "",
        route_revision=1 if route_valid else 0,
        route_confidence=1.0 if route_valid else 0.0,
        input_vector_sha256="fixture",
    )


def test_route_supervision_is_metric_deterministic_and_directional():
    route = _route()
    raster = _raster()
    pose = raster.sample_pose

    first = build_route_supervision(route, pose, raster)
    second = build_route_supervision(route, pose, raster)

    for name, value in first.arrays().items():
        assert np.array_equal(value, second.arrays()[name])

    geometry = DEFAULT_NAVIGATION_GEOMETRY
    center = np.rint(
        geometry.ego_to_pixel(np.asarray([[20.0, 0.0]]))[0]
    ).astype(np.int64)
    outside = np.rint(
        geometry.ego_to_pixel(np.asarray([[20.0, 20.0]]))[0]
    ).astype(np.int64)
    assert first.distance_to_corridor_m[tuple(center)] == 0.0
    assert first.distance_to_corridor_m[tuple(outside)] > 0.0
    assert first.distance_to_corridor_m.max() <= (
        MAXIMUM_OUTSIDE_DISTANCE_M
    )
    assert first.distance_to_drivable_m[tuple(center)] == 0.0
    assert first.distance_to_drivable_m[tuple(outside)] > 0.0
    assert (
        first.distance_to_drivable_m.max()
        <= MAXIMUM_OUTSIDE_DISTANCE_M
    )
    assert first.drivable_available
    assert first.route_heading_valid[tuple(center)] == 1
    direction_norm = np.hypot(
        first.route_heading_sin[tuple(center)],
        first.route_heading_cos[tuple(center)],
    )
    assert direction_norm == pytest.approx(1.0)
    assert first.route_heading_sin[tuple(center)] == pytest.approx(0.0)
    assert first.route_heading_cos[tuple(center)] == pytest.approx(1.0)
    assert first.destination_xy_m == pytest.approx([100.0, 0.0])
    assert first.destination_visible


def test_invalid_route_has_no_supervision_or_future_trajectory_fields():
    supervision = build_route_supervision(
        _route(valid=False),
        _raster(route_valid=False).sample_pose,
        _raster(route_valid=False),
    )

    assert np.count_nonzero(supervision.distance_to_corridor_m) == 0
    assert np.count_nonzero(supervision.distance_to_drivable_m) > 0
    assert supervision.drivable_available
    assert np.count_nonzero(supervision.route_heading_valid) == 0
    assert np.count_nonzero(supervision.destination_xy_m) == 0
    assert not supervision.destination_visible
    assert set(supervision.arrays()) == {
        "distance_to_corridor_m",
        "distance_to_drivable_m",
        "drivable_available",
        "route_heading_sin",
        "route_heading_cos",
        "route_heading_valid",
        "destination_xy_m",
        "destination_visible",
    }


def test_invalid_map_disables_drivable_supervision_explicitly():
    raster = _raster(map_valid=False)

    supervision = build_route_supervision(
        _route(),
        raster.sample_pose,
        raster,
    )

    assert not supervision.drivable_available
    assert np.all(
        supervision.distance_to_drivable_m
        == MAXIMUM_OUTSIDE_DISTANCE_M
    )


def test_valid_map_without_drivable_pixels_is_rejected():
    raster = _raster()
    map_context = raster.map_context.copy()
    map_context[MapChannel.DRIVABLE_AREA] = 0.0
    raster = dataclasses.replace(raster, map_context=map_context)

    with pytest.raises(ValueError, match="no drivable pixels"):
        build_route_supervision(
            _route(),
            raster.sample_pose,
            raster,
        )
