"""Clipping drivable OSM ways out of a country-sized extract."""

from __future__ import annotations

import pytest

from data_parsing.l2d.osm_pbf_region import CELL_DEG, clip_cells, file_sha256


def test_clip_cells_cover_the_buffer_around_every_point():
    cells = clip_cells([(8.40, 49.00), (13.40, 52.50)], buffer_m=1000.0)
    assert (int(49.00 // CELL_DEG), int(8.40 // CELL_DEG)) in cells
    assert (int(52.50 // CELL_DEG), int(13.40 // CELL_DEG)) in cells
    # A point 5 km away is not covered by a 1 km buffer.
    assert (int(49.00 // CELL_DEG), int(8.47 // CELL_DEG)) not in cells
    with pytest.raises(ValueError, match="positive"):
        clip_cells([(8.4, 49.0)], buffer_m=0.0)
    with pytest.raises(ValueError, match="finite"):
        clip_cells([(float("nan"), 49.0)], buffer_m=100.0)


def test_file_digest_streams(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 1000)
    assert file_sha256(path, chunk_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


OSM_XML = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" lat="49.0" lon="8.40" version="1"/>
  <node id="2" lat="49.0" lon="8.401" version="1"><tag k="highway" v="traffic_signals"/><tag k="name" v="x"/></node>
  <node id="3" lat="49.0" lon="8.402" version="1"/>
  <node id="7" lat="52.5" lon="13.40" version="1"/>
  <node id="8" lat="52.5" lon="13.401" version="1"/>
  <way id="10" version="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="highway" v="primary"/><tag k="lanes" v="2"/></way>
  <way id="11" version="1"><nd ref="7"/><nd ref="8"/><tag k="highway" v="primary"/></way>
  <way id="12" version="1"><nd ref="1"/><nd ref="3"/><tag k="highway" v="footway"/></way>
</osm>
"""


def test_read_region_keeps_drivable_ways_near_the_tracks_and_their_point_tags(tmp_path):
    pytest.importorskip("osmium")
    from data_parsing.l2d.osm_pbf_region import read_region

    path = tmp_path / "region.osm"
    path.write_text(OSM_XML)
    nodes, ways = read_region(path, clip_cells([(8.401, 49.0)], buffer_m=500.0))
    assert [way.way_id for way in ways] == [10]
    assert ways[0].tags["lanes"] == "2"
    assert sorted(nodes) == [1, 2, 3]
    assert nodes[2].tags == {"highway": "traffic_signals"}
    assert nodes[1].tags == {}


def test_region_round_trips_canonically_and_subregions_keep_whole_ways():
    from data_parsing.l2d.osm_lane_graph import OSMNode, OSMWay
    from data_parsing.l2d.osm_pbf_region import parse_region, region_bytes, subregion

    nodes = {
        1: OSMNode(1, 8.40, 49.0),
        2: OSMNode(2, 8.401, 49.0, {"highway": "traffic_signals"}),
        3: OSMNode(3, 8.402, 49.0),
        7: OSMNode(7, 13.40, 52.5),
        8: OSMNode(8, 13.401, 52.5),
    }
    ways = [OSMWay(11, (7, 8), {"highway": "primary"}), OSMWay(10, (1, 2, 3), {"highway": "primary", "lanes": "2"})]
    payload = region_bytes(nodes, ways, provenance={"pbf_sha256": "0" * 64})
    assert region_bytes(dict(reversed(list(nodes.items()))), list(reversed(ways)), provenance={"pbf_sha256": "0" * 64}) == payload
    parsed_nodes, parsed_ways, provenance = parse_region(payload)
    assert parsed_nodes == nodes and [way.way_id for way in parsed_ways] == [10, 11]
    assert provenance == {"pbf_sha256": "0" * 64}
    near_nodes, near_ways = subregion(parsed_nodes, parsed_ways, clip_cells([(8.40, 49.0)], buffer_m=100.0))
    assert [way.way_id for way in near_ways] == [10]
    # Node 3 lies outside the cells but its way touches them, so it stays.
    assert sorted(near_nodes) == [1, 2, 3]
