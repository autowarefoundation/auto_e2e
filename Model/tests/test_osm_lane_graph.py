"""OSM ways to the canonical lane graph that L2D renders through."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pyproj")

from data_parsing.l2d.osm_lane_graph import (  # noqa: E402
    OSMNode,
    OSMWay,
    build_osm_lane_graph,
    lane_layout,
    utm_epsg,
)
from navigation.osm_adapter import OSMMapAdapter  # noqa: E402

ORIGIN_LON, ORIGIN_LAT = 8.4, 49.0
# About 73 m of longitude and 111 m of latitude per 1e-3 degree here.
DLON, DLAT = 1e-3 / 73.0, 1e-3 / 111.0


def _node(node_id, x_m, y_m, **tags):
    return OSMNode(node_id, ORIGIN_LON + x_m * DLON, ORIGIN_LAT + y_m * DLAT, tags)


def _t_junction():
    """A two-lane main road west-east with a side road meeting it at x=0."""
    nodes = {
        1: _node(1, -100, 0),
        2: _node(2, -40, 0, highway="traffic_signals", **{"traffic_signals:direction": "forward"}),
        3: _node(3, 0, 0),
        4: _node(4, 100, 0),
        5: _node(5, 0, 80, highway="stop"),
        6: _node(6, 0, 30, highway="crossing", crossing="zebra"),
    }
    ways = [
        OSMWay(10, (1, 2, 3, 4), {"highway": "primary", "lanes": "2", "layer": "0"}),
        OSMWay(11, (5, 6, 3), {"highway": "residential"}),
        OSMWay(12, (1, 4), {"highway": "footway"}),
    ]
    return nodes, ways


def _graph():
    nodes, ways = _t_junction()
    return build_osm_lane_graph(
        nodes, ways, origin_longitude_deg=ORIGIN_LON, origin_latitude_deg=ORIGIN_LAT,
        map_version="test-v1", provenance={"source": "unit"},
    )


def test_lane_layout_follows_osm_lane_tags():
    assert lane_layout({"highway": "primary", "lanes": "4"})[:2] == (2, 2)
    assert lane_layout({"highway": "primary", "lanes": "3", "lanes:forward": "2"})[:2] == (2, 1)
    assert lane_layout({"highway": "residential", "oneway": "yes", "lanes": "2"})[:2] == (2, 0)
    assert lane_layout({"highway": "residential", "oneway": "-1"})[:2] == (0, 1)
    assert lane_layout({"highway": "motorway"})[:2] == (1, 0)
    assert lane_layout({"highway": "primary", "lanes": "2", "width": "7"})[2] == pytest.approx(3.5)
    assert utm_epsg(8.4, 49.0) == 25832 and utm_epsg(13.4, 52.5) == 25833


def test_ways_split_at_junctions_into_directed_lanes_with_topology():
    graph = _graph()
    lanes = {f"{lane['way_id']}:{lane['direction']}:{lane['lane_index']}": lane for lane in graph["lanes"]}
    # The footway is dropped; the main road splits at the junction node 3.
    assert {lane["osm_way_id"] for lane in lanes.values()} == {10, 11}
    west, east = "10000:forward:0", "10001:forward:0"
    assert east in lanes[west]["successors"]
    # Right-hand traffic: the eastbound lane lies south of the drawn line.
    assert np.mean(np.asarray(lanes[west]["centerline_enu_m"])[:, 1]) < 0.0
    assert np.mean(np.asarray(lanes["10000:reverse:0"]["centerline_enu_m"])[:, 1]) > 0.0
    # Turning from the side road onto the main road is a successor too.
    side = next(lane_id for lane_id, lane in lanes.items() if lane["osm_way_id"] == 11 and lane["direction"] == "forward")
    assert any(lanes[successor]["osm_way_id"] == 10 for successor in lanes[side]["successors"])
    assert all(lane["level"] == 0 for lane in lanes.values())


def test_point_tags_become_hd_map_semantics():
    semantic = _graph()["semantic"]
    assert [item["id"] for item in semantic["intersections"]] == ["3"]
    assert [item["id"] for item in semantic["traffic_signals"]] == ["2"]
    assert [item["id"] for item in semantic["crosswalks"]] == ["6"]
    stop_ids = {item["id"].split(":")[0] for item in semantic["stop_lines"]}
    assert stop_ids == {"2", "5"}
    # The forward-only signal stops eastbound traffic only, at the node.
    signal_lines = [item for item in semantic["stop_lines"] if item["id"].startswith("2:")]
    assert all(":forward:" in item["id"] for item in signal_lines)
    assert all(abs(np.mean(np.asarray(item["points_enu_m"])[:, 0]) + 40.0) < 1.0 for item in signal_lines)


def test_lane_graph_renders_every_osm_channel_through_the_shared_rasterizer():
    rasterizer_module = pytest.importorskip("navigation.rasterizer")
    from navigation.geometry import MapChannel

    try:
        rasterizer = rasterizer_module.NativeNavigationRasterizer()
    except rasterizer_module.NativeRasterizerUnavailable:
        pytest.skip("native navigation rasterizer is not built here")
    adapter = OSMMapAdapter(_graph(), source_sha256="0" * 64)
    pose = rasterizer_module.EgoPose(x_enu_m=-60.0, y_enu_m=-1.6, yaw_rad=0.0, timestamp_ns=0)
    raster = rasterizer.render(adapter.extract(), None, pose)
    filled = {channel.name for channel in MapChannel if float(np.asarray(raster.map_context[channel]).max()) > 0.0}
    for name in ("DRIVABLE_AREA", "LANE_BOUNDARY", "LANE_CENTERLINE", "INTERSECTION", "CROSSWALK",
                 "STOP_LINE", "STATIC_TRAFFIC_SIGNAL", "TRAFFIC_DIRECTION_VALID", "KNOWN_MAP_AREA",
                 "ROAD_LEVEL_VALID"):
        assert name in filled, name
    assert math.isfinite(float(np.asarray(raster.map_context).sum()))
