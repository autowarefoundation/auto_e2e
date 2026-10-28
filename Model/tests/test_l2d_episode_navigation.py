"""L2D episodes render KITScenes-format navigation from OSM and the map image."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("scipy")
rasterizer_module = pytest.importorskip("navigation.rasterizer")

from data_parsing.l2d.episode_navigation import L2DEpisodeNavigation, episode_lane_graph  # noqa: E402
from navigation.artifacts import decode_sample_navigation  # noqa: E402
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY, MapChannel, RouteChannel  # noqa: E402

from test_l2d_map_image import _image  # noqa: E402
from test_osm_lane_graph import DLAT, DLON, ORIGIN_LAT, ORIGIN_LON, _graph  # noqa: E402


@pytest.fixture(scope="module")
def rasterizer():
    try:
        return rasterizer_module.NativeNavigationRasterizer(AUTOE2E_NAVIGATION_GEOMETRY)
    except rasterizer_module.NativeRasterizerUnavailable:
        pytest.skip("native navigation rasterizer is not built here")


def _eastbound_drive(count=60):
    """10 Hz drive along the eastbound lane, 1.6 m south of the drawn road."""
    x = np.linspace(-90.0, 90.0, count)
    lon = ORIGIN_LON + x * DLON
    lat = np.full(count, ORIGIN_LAT - 1.6 * DLAT)
    return lon, lat, np.full(count, 90.0), np.arange(count, dtype=np.int64) * 100_000_000


def _navigation(rasterizer):
    lon, lat, heading, timestamps = _eastbound_drive()
    return L2DEpisodeNavigation(
        episode_id="7", longitudes_deg=lon, latitudes_deg=lat, headings_deg_cw_from_north=heading,
        timestamps_ns=timestamps, lane_graph=_graph(), lane_graph_sha256="a" * 64, rasterizer=rasterizer,
    )


def test_episode_graph_keeps_nearby_lanes_and_drops_dangling_references():
    graph = _graph()
    trace = np.asarray([[-100.0, -1.6], [-60.0, -1.6]])
    cut = episode_lane_graph(graph, trace, margin_m=20.0)
    kept = {f"{lane['way_id']}:{lane['direction']}:{lane['lane_index']}" for lane in cut["lanes"]}
    assert kept and len(kept) < len(graph["lanes"])
    assert all(successor in kept for lane in cut["lanes"] for successor in lane["successors"])
    with pytest.raises(ValueError, match="no OSM lanes"):
        episode_lane_graph(graph, np.asarray([[5000.0, 5000.0], [5001.0, 5000.0]]), margin_m=10.0)


def test_recorded_trace_matches_a_valid_route_on_the_osm_lanes(rasterizer):
    navigation = _navigation(rasterizer)
    assert navigation.route.valid, navigation.route.quality.failure_reasons
    lanes = [segment.provider_segment_id for segment in navigation.route.lane_sequence]
    assert lanes[0].startswith("10000:forward") and lanes[-1].startswith("10001:forward")


def test_sample_members_are_the_kitscenes_members_with_osm_channels(rasterizer):
    navigation = _navigation(rasterizer)
    members = navigation.sample_members(10)
    assert set(members) == {"map_semantic.npz", "route_mask.npz", "route_supervision.npz", "navigation_meta.json"}
    map_context, route_mask, metadata = decode_sample_navigation(members)
    assert map_context.shape[0] == 14 and route_mask.shape[0] == 2
    for channel in (MapChannel.DRIVABLE_AREA, MapChannel.LANE_BOUNDARY, MapChannel.LANE_CENTERLINE,
                    MapChannel.INTERSECTION, MapChannel.STOP_LINE, MapChannel.STATIC_TRAFFIC_SIGNAL,
                    MapChannel.TRAFFIC_DIRECTION_VALID, MapChannel.KNOWN_MAP_AREA):
        assert map_context[channel].max() > 0.0, channel.name
    assert route_mask[RouteChannel.SELECTED_CORRIDOR].any()
    assert metadata["map_source"] == "osm_lane_graph+l2d_map_image"
    assert metadata["map_image_registered"] is False
    json.dumps(metadata)


def test_registered_map_image_adds_road_surface_osm_leaves_out(rasterizer):
    navigation = _navigation(rasterizer)
    frame = 30
    ego = navigation.positions[frame, :2]
    # Ten waypoints ahead along the drive, 8 m apart (heading east).
    waypoints_east_north = np.asarray([[8.0 * k, 0.0] for k in range(1, 11)])
    waypoint_lon_lat = np.column_stack([
        ORIGIN_LON + (ego[0] + waypoints_east_north[:, 0]) * DLON,
        ORIGIN_LAT + (ego[1] + waypoints_east_north[:, 1]) * DLAT,
    ])
    # The test image draws a road running along local north; rotate it to east.
    image = _image(1.25, -np.pi / 2, waypoints_east_north[:, [1, 0]] * [1, -1])
    without = decode_sample_navigation(navigation.sample_members(frame))[0]
    members = navigation.sample_members(frame, map_image=image, waypoints_lon_lat=waypoint_lon_lat)
    with_image, _, metadata = decode_sample_navigation(members)
    assert metadata["map_image_registered"] is True
    assert metadata["map_image_residual_px"] < 3.0
    assert (with_image[MapChannel.DRIVABLE_AREA] >= without[MapChannel.DRIVABLE_AREA]).all()
    assert metadata["map_image_drivable_iou"] is not None
