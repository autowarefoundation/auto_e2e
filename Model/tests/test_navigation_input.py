from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from navigation.artifacts import (
    decode_array,
    decode_scene_navigation,
    encode_array,
    encode_sample_navigation,
    encode_scene_navigation,
)
from navigation.contracts import (
    Destination,
    DirectedLaneField,
    MapFrame,
    NavigationMap,
    NavigationRoute,
    PolygonPrimitive,
    PolylinePrimitive,
    RouteLaneSegment,
    RouteProvenance,
    RouteQuality,
    StaticTrafficSignal,
    canonical_json_bytes,
)
from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
)
from navigation.lanelet2_adapter import Lanelet2MapAdapter
from navigation.lanelet2_matcher import Lanelet2TraceMatcher
from navigation.native.build import build as build_native
from navigation.rasterizer import EgoPose, NativeNavigationRasterizer
from navigation.supervision import empty_route_supervision


class _Point:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Line(list):
    def __init__(self, line_id, points, attributes=None):
        super().__init__([_Point(*point) for point in points])
        self.id = line_id
        self.attributes = dict(attributes or {})


class _Lanelet:
    def __init__(
        self,
        lanelet_id,
        start,
        end,
        *,
        y=0.0,
        attributes=None,
        level=None,
    ):
        attrs = {"subtype": "road", **(attributes or {})}
        if level is not None:
            attrs["layer"] = str(level)
        self.id = lanelet_id
        self.attributes = attrs
        self.leftBound = _Line(
            lanelet_id * 10 + 1,
            [(start, y + 2.0), (end, y + 2.0)],
        )
        self.rightBound = _Line(
            lanelet_id * 10 + 2,
            [(start, y - 2.0), (end, y - 2.0)],
        )
        self.centerline = _Line(
            lanelet_id * 10 + 3,
            [(start, y), (end, y)],
        )

    def polygon2d(self):
        left = list(self.leftBound)
        right = list(self.rightBound)
        return [left[0], left[1], right[1], right[0]]


class _Graph:
    def __init__(self, following=None, left=None, right=None):
        self._following = following or {}
        self._left = left or {}
        self._right = right or {}

    def following(self, lanelet):
        return self._following.get(lanelet.id, [])

    def previous(self, lanelet):
        return [
            source
            for following in self._following.values()
            for source in following
            if source.id == lanelet.id
        ]

    def left(self, lanelet):
        return self._left.get(lanelet.id)

    def right(self, lanelet):
        return self._right.get(lanelet.id)

    def shortestPath(self, start, end):
        if any(item.id == end.id for item in self.following(start)):
            return [start, end]
        for middle in self.following(start):
            if any(item.id == end.id for item in self.following(middle)):
                return [start, middle, end]
        return None


@pytest.fixture(scope="session")
def native_library(tmp_path_factory):
    suffix = ".dylib" if __import__("platform").system() == "Darwin" else ".so"
    return build_native(
        tmp_path_factory.mktemp("navigation-native")
        / f"libnavigation_rasterizer{suffix}"
    )


def _frame():
    return MapFrame("test-map", 49.0, 8.0, "local ENU")


def _quality(valid=True):
    return RouteQuality(
        matched_pose_ratio=1.0 if valid else 0.0,
        median_lateral_distance_m=0.0,
        p95_lateral_distance_m=0.0,
        median_heading_error_rad=0.0,
        p95_heading_error_rad=0.0,
        failure_reasons=() if valid else ("no_route",),
    )


def _provenance():
    return RouteProvenance(
        source_revision="source",
        matcher_version="matcher",
        matcher_config_sha256="config",
        map_sha256="map",
        trace_sha256="trace",
    )


def _route(centerline, *, valid=True, destination=(20.0, 0.0), level=0):
    segments = (
        RouteLaneSegment(
            lane_id="lanelet2:1",
            provider_segment_id="1",
            centerline_enu_m=centerline,
            level=level,
        ),
    ) if valid else ()
    return NavigationRoute(
        route_id="route-1",
        revision=1,
        provider="test",
        timestamp_ns=0,
        valid_from_ns=0,
        map_version="map-v1",
        frame=_frame(),
        lane_sequence=segments,
        destination=Destination(np.asarray(destination), "test"),
        confidence=1.0 if valid else 0.0,
        valid=valid,
        quality=_quality(valid),
        estimated_destination=False,
        provenance=_provenance(),
    )


def _semantic_map():
    centerline = np.asarray([[-20.0, 0.0], [20.0, 0.0]])
    road = np.asarray(
        [[-30.0, -4.0], [30.0, -4.0], [30.0, 4.0], [-30.0, 4.0]]
    )
    overlap = np.asarray(
        [[-5.0, -3.0], [5.0, -3.0], [5.0, 3.0], [-5.0, 3.0]]
    )
    return NavigationMap(
        map_version="map-v1",
        provider="test",
        frame=_frame(),
        bounds_enu_m=(-40.0, -20.0, 40.0, 20.0),
        drivable_polygons=(PolygonPrimitive("road", road, level=0),),
        lane_boundaries=(
            PolylinePrimitive(
                "boundary",
                np.asarray([[-20.0, 3.0], [20.0, 3.0]]),
                level=0,
            ),
        ),
        lane_centerlines=(
            PolylinePrimitive("centerline", centerline, level=0),
        ),
        intersection_polygons=(
            PolygonPrimitive("intersection", overlap, level=1),
        ),
        crosswalk_polygons=(
            PolygonPrimitive(
                "crosswalk",
                np.asarray(
                    [[8.0, -3.0], [10.0, -3.0], [10.0, 3.0], [8.0, 3.0]]
                ),
            ),
        ),
        stop_lines=(
            PolylinePrimitive(
                "stop", np.asarray([[6.0, -3.0], [6.0, 3.0]])
            ),
        ),
        static_traffic_signals=(
            StaticTrafficSignal("signal", np.asarray([6.0, 4.0])),
        ),
        directed_lane_fields=(
            DirectedLaneField("lanelet2:1", centerline, level=0),
        ),
        layer_availability={"road_level": True},
        provenance={"fixture": "test"},
    )


def test_geometry_uses_pixel_centers_and_existing_anchor():
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    points = np.asarray([[0.0, 0.0], [17.25, -9.5]])
    pixels = geometry.ego_to_pixel(points)
    np.testing.assert_allclose(
        pixels[0],
        [geometry.ego_anchor_row, geometry.ego_anchor_col],
    )
    np.testing.assert_allclose(geometry.pixel_to_ego(pixels), points)
    assert geometry.matching_pc_range[:2] == (
        geometry.x_min_m,
        geometry.y_min_m,
    )
    contract = geometry.contract()
    assert contract["camera_bev"] == {
        "bev_h": geometry.height_px,
        "bev_w": geometry.width_px,
        "pc_range": list(geometry.matching_pc_range),
    }


def test_contract_has_no_future_target_field():
    route = _route(np.asarray([[0.0, 0.0], [10.0, 0.0]]))
    serialized = canonical_json_bytes(route)
    assert b"future_waypoint" not in serialized
    assert b"trajectory_target" not in serialized
    with pytest.raises(TypeError):
        RouteLaneSegment(
            lane_id="lane",
            provider_segment_id="lane",
            centerline_enu_m=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            future_waypoints=np.zeros((64, 2)),
        )


def test_lanelet_adapter_extracts_provider_independent_primitives():
    lane = _Lanelet(1, 0.0, 20.0, attributes={"turn_direction": "left"})
    crosswalk = _Lanelet(
        2,
        5.0,
        8.0,
        attributes={"subtype": "crosswalk"},
    )
    graph = _Graph()
    scene_map = SimpleNamespace(
        lanelet_map=SimpleNamespace(
            laneletLayer=[crosswalk, lane],
            regulatoryElementLayer=[],
        ),
        routing_graph=graph,
        traffic_rules=SimpleNamespace(canPass=lambda value: value.id == 1),
        get_stop_lines=lambda: [np.asarray([[4.0, -2.0], [4.0, 2.0]])],
        _origin_lat=49.0,
        _origin_lon=8.0,
    )
    result = Lanelet2MapAdapter(
        scene_map,
        map_version="map-v1",
        map_sha256="digest",
        frame_id="scene",
        source_revision="source",
    ).extract()

    assert [field.lane_id for field in result.directed_lane_fields] == [
        "lanelet2:1"
    ]
    assert len(result.drivable_polygons) == 1
    assert len(result.crosswalk_polygons) == 1
    assert len(result.intersection_polygons) == 1
    assert len(result.stop_lines) == 1


def test_trace_matcher_fills_routing_gap_without_serializing_trace():
    first = _Lanelet(1, 0.0, 10.0)
    middle = _Lanelet(2, 10.0, 20.0)
    final = _Lanelet(3, 20.0, 30.0)
    graph = _Graph({1: [middle], 2: [final]})

    def query(center, radius):
        if center[0] < 12.0:
            return [first]
        if center[0] > 18.0:
            return [final]
        return []

    scene_map = SimpleNamespace(
        routing_graph=graph,
        traffic_rules=SimpleNamespace(canPass=lambda _: True),
        get_lanelets_in_roi=query,
    )
    navigation_map = NavigationMap(
        map_version="map-v1",
        provider="test",
        frame=_frame(),
        bounds_enu_m=(0.0, -3.0, 30.0, 3.0),
    )
    positions = np.column_stack(
        [np.concatenate([np.linspace(1.0, 9.0, 10), np.linspace(21.0, 29.0, 10)]),
         np.zeros(20)]
    )
    route = Lanelet2TraceMatcher(
        scene_map,
        navigation_map,
        map_sha256="map",
        source_revision="source",
    ).match(
        scene_id="scene",
        positions_enu_m=positions,
        yaws_rad=np.zeros(20),
        timestamps_ns=np.arange(20, dtype=np.int64) * 100_000_000,
    )

    assert [segment.provider_segment_id for segment in route.lane_sequence] == [
        "1",
        "2",
        "3",
    ]
    assert route.quality.shortest_path_fill_count == 1
    assert route.provenance.trace_sha256
    assert b"positions_enu_m" not in canonical_json_bytes(route)


def _reference_route_mask(centerline, destination):
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    x_grid, y_grid = geometry.pixel_center_grids()
    output = np.zeros((2, geometry.height_px, geometry.width_px), dtype=np.uint8)
    start, end = centerline
    delta = end - start
    fraction = (
        (x_grid - start[0]) * delta[0] + (y_grid - start[1]) * delta[1]
    ) / float(np.dot(delta, delta))
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest_x = start[0] + fraction * delta[0]
    nearest_y = start[1] + fraction * delta[1]
    distance = np.hypot(x_grid - nearest_x, y_grid - nearest_y)
    output[0] = (
        (distance <= geometry.route_corridor_width_m / 2.0)
        & (x_grid >= -geometry.route_rear_clip_m)
    )
    output[1] = (
        np.hypot(x_grid - destination[0], y_grid - destination[1])
        <= geometry.destination_marker_radius_m
    )
    return output


def test_cpp_renderer_matches_python_route_fixture(native_library):
    navigation_map = _semantic_map()
    centerline = np.asarray([[-20.0, 0.0], [20.0, 0.0]])
    route = _route(centerline)
    rasterizer = NativeNavigationRasterizer(library_path=native_library)
    raster = rasterizer.render(
        navigation_map,
        route,
        EgoPose(0.0, 0.0, 0.0, 0),
    )

    expected = _reference_route_mask(centerline, np.asarray([20.0, 0.0]))
    np.testing.assert_array_equal(raster.route_mask, expected)
    assert raster.map_valid and raster.route_valid
    assert np.isin(raster.route_mask, (0, 1)).all()


def test_cpp_renderer_populates_semantics_and_level_ambiguity(native_library):
    rasterizer = NativeNavigationRasterizer(library_path=native_library)
    raster = rasterizer.render(
        _semantic_map(),
        _route(np.asarray([[-20.0, 0.0], [20.0, 0.0]])),
        EgoPose(0.0, 0.0, 0.0, 0),
    )
    expected_nonempty = (
        MapChannel.DRIVABLE_AREA,
        MapChannel.LANE_BOUNDARY,
        MapChannel.LANE_CENTERLINE,
        MapChannel.CROSSWALK,
        MapChannel.STOP_LINE,
        MapChannel.STATIC_TRAFFIC_SIGNAL,
        MapChannel.TRAFFIC_DIRECTION_VALID,
        MapChannel.KNOWN_MAP_AREA,
        MapChannel.ROAD_LEVEL_VALID,
        MapChannel.OVERLAPPING_LEVEL_AMBIGUITY,
    )
    for channel in expected_nonempty:
        assert raster.map_context[channel].sum() > 0, channel
    direction_valid = raster.map_context[MapChannel.TRAFFIC_DIRECTION_VALID] > 0
    np.testing.assert_allclose(
        raster.map_context[MapChannel.TRAFFIC_DIRECTION_SIN][direction_valid],
        0.5,
    )
    np.testing.assert_allclose(
        raster.map_context[MapChannel.TRAFFIC_DIRECTION_COS][direction_valid],
        1.0,
    )
    assert raster.map_context[MapChannel.INTERSECTION].sum() == 0


def test_warp_rotates_direction_vectors_and_exposes_unknown(native_library):
    rasterizer = NativeNavigationRasterizer(library_path=native_library)
    raster = rasterizer.render(
        _semantic_map(),
        _route(np.asarray([[-20.0, 0.0], [20.0, 0.0]])),
        EgoPose(0.0, 0.0, 0.0, 0),
    )
    warped = rasterizer.warp(
        raster,
        EgoPose(0.0, 0.0, math.pi / 2.0, 100_000_000),
    )
    valid = warped.map_context[MapChannel.TRAFFIC_DIRECTION_VALID] > 0
    assert valid.any()
    np.testing.assert_allclose(
        warped.map_context[MapChannel.TRAFFIC_DIRECTION_SIN][valid],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        warped.map_context[MapChannel.TRAFFIC_DIRECTION_COS][valid],
        0.5,
        atol=1e-6,
    )
    translated = rasterizer.warp(
        raster,
        EgoPose(200.0, 0.0, 0.0, 200_000_000),
    )
    assert translated.map_context[MapChannel.KNOWN_MAP_AREA].sum() == 0


def test_invalid_route_has_explicit_validity_and_zero_mask(native_library):
    raster = NativeNavigationRasterizer(library_path=native_library).render(
        _semantic_map(),
        _route(np.asarray([[-20.0, 0.0], [20.0, 0.0]]), valid=False),
        EgoPose(0.0, 0.0, 0.0, 0),
    )
    assert raster.map_valid
    assert not raster.route_valid
    assert raster.route_mask.sum() == 0


def test_artifacts_are_deterministic_and_lossless(native_library):
    rasterizer = NativeNavigationRasterizer(library_path=native_library)
    navigation_map = _semantic_map()
    route = _route(np.asarray([[-20.0, 0.0], [20.0, 0.0]]))
    raster = rasterizer.render(
        navigation_map,
        route,
        EgoPose(0.0, 0.0, 0.0, 0),
    )
    first = encode_array(raster.map_context)
    second = encode_array(raster.map_context)
    assert first == second
    np.testing.assert_array_equal(decode_array(first), raster.map_context)
    supervision = empty_route_supervision(rasterizer.geometry)
    first_members = encode_sample_navigation(
        raster,
        route_supervision=supervision,
    )
    second_members = encode_sample_navigation(
        raster,
        route_supervision=supervision,
    )
    assert first_members == second_members
    scene = encode_scene_navigation(navigation_map, route)
    decoded_map, decoded_route = decode_scene_navigation(scene)
    assert encode_scene_navigation(decoded_map, decoded_route) == scene
