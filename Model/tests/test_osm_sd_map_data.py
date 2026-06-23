"""Tests for the OSM SD-map data pipeline (coords, parser, tokenizer, overpass query).

shapely-dependent tests are skipped when shapely is unavailable. The HF download
and live Overpass fetch are not exercised (network); only the pure query/bbox
construction and offline parsing/tokenisation are tested.
"""

import io
import math
import sys

import numpy as np
import pytest

sys.path.append('..')

from data_parsing.osm_sd_map.wgs84_to_local import wgs84_to_ego_local
from data_parsing.osm_sd_map.overpass import build_query, bbox_from_center


class TestWGS84ToLocal:
    def test_ego_at_origin(self):
        out = wgs84_to_ego_local([48.0], [8.0], 48.0, 8.0, 0.0)
        assert np.allclose(out, [[0.0, 0.0]], atol=1e-6)

    def test_north_is_forward_when_heading_north(self):
        # A point due north of ego, heading north -> +x (forward), ~0 y.
        out = wgs84_to_ego_local([48.001], [8.0], 48.0, 8.0, 0.0)
        assert out[0, 0] > 1.0  # forward positive
        assert abs(out[0, 1]) < 1e-3  # no lateral

    def test_west_is_left_when_heading_north(self):
        # A point due west, heading north -> left (+y), ~0 forward.
        out = wgs84_to_ego_local([48.0], [7.999], 48.0, 8.0, 0.0)
        assert out[0, 1] > 1.0  # left positive
        assert abs(out[0, 0]) < 1e-3

    def test_heading_rotates_frame(self):
        # Heading east: a point due north is now to the ego's left (+y) and
        # straight ahead is east, so forward (x) of a northern point is ~0.
        out = wgs84_to_ego_local([48.001], [8.0], 48.0, 8.0, math.pi / 2)
        assert out[0, 1] > 1.0          # north is on the left when facing east
        assert abs(out[0, 0]) < 1e-3    # north is not ahead when facing east


class TestOverpassQuery:
    def test_query_is_exact_tested_string(self):
        q = build_query(1.0, 2.0, 3.0, 4.0)
        assert q == ("(node(1.0,2.0,3.0,4.0); rel(bn)->.x; way(1.0,2.0,3.0,4.0); "
                     "node(w)->.x; rel(bw);); out meta;")

    def test_bbox_order_and_span(self):
        s, w, n, e = bbox_from_center(48.0, 8.0, 800)
        assert s < 48.0 < n and w < 8.0 < e


class TestTraceBboxCache:
    def test_bbox_from_points_covers_all_points(self):
        from data_parsing.osm_sd_map.overpass import bbox_from_points
        lats, lons = [48.0, 48.01, 47.995], [8.0, 8.02, 7.99]
        s, w, n, e = bbox_from_points(lats, lons, margin_m=100)
        assert s < min(lats) and n > max(lats)
        assert w < min(lons) and e > max(lons)

    def test_patch_reach_is_corner_distance(self):
        from data_parsing.osm_sd_map.cache import patch_reach_m
        r = patch_reach_m((-60.0, -30.0, -5.0, 60.0, 30.0, 3.0), buffer_m=0.0)
        assert abs(r - (60.0 ** 2 + 30.0 ** 2) ** 0.5) < 1e-6

    def test_containment_reuse_avoids_fetch(self, tmp_path):
        import gzip
        from unittest import mock
        from data_parsing.osm_sd_map import overpass

        # Pre-populate a large covering bbox cache file.
        big = (47.99, 7.99, 48.02, 8.03)
        with gzip.open(tmp_path / overpass._bbox_cache_name(*big), "wt", encoding="utf-8") as f:
            f.write("<osm>COVER</osm>")

        # A request fully inside the cached bbox must reuse it (no network).
        def _boom(*a, **k):
            raise AssertionError("fetch_raw_osm should not be called on a cache hit")

        with mock.patch.object(overpass, "fetch_raw_osm", _boom):
            xml = overpass.load_or_fetch_osm_bbox(48.0, 8.0, 48.005, 8.01, tmp_path)
        assert xml == "<osm>COVER</osm>"

    def test_fetch_then_reuse(self, tmp_path):
        from unittest import mock
        from data_parsing.osm_sd_map import overpass

        with mock.patch.object(overpass, "fetch_raw_osm", lambda *a, **k: "<osm>NEW</osm>"):
            first = overpass.load_or_fetch_osm_bbox(10.0, 10.0, 10.002, 10.002, tmp_path)
        assert first == "<osm>NEW</osm>"

        # A contained request reuses the just-written file without fetching.
        def _boom(*a, **k):
            raise AssertionError("should reuse covering cache, not fetch")

        with mock.patch.object(overpass, "fetch_raw_osm", _boom):
            second = overpass.load_or_fetch_osm_bbox(10.0005, 10.0005, 10.001, 10.001, tmp_path)
        assert second == "<osm>NEW</osm>"


_OSM_XML = """<?xml version='1.0'?>
<osm version='0.6'>
  <node id='1' lat='48.0000' lon='8.0000'><tag k='highway' v='traffic_signals'/></node>
  <node id='2' lat='48.0002' lon='8.0000'/>
  <node id='3' lat='48.0002' lon='8.0003'/>
  <way id='10'>
    <nd ref='2'/><nd ref='3'/>
    <tag k='highway' v='residential'/><tag k='name' v='Test St'/>
  </way>
  <relation id='100'>
    <member type='way' ref='10' role='street'/>
    <member type='node' ref='1' role='sign'/>
    <tag k='type' v='associatedStreet'/>
  </relation>
</osm>
"""


class _FakeTokenizer:
    """Minimal HF-tokenizer-like callable returning a dict of lists."""

    def __call__(self, strings, padding=True):
        ids = [[1] + [ord(c) % 50 + 2 for c in s[:6]] for s in strings]
        maxlen = max((len(x) for x in ids), default=0)
        ids = [x + [0] * (maxlen - len(x)) for x in ids]
        return {
            "input_ids": ids,
            "token_type_ids": [[0] * len(x) for x in ids],
            "attention_mask": [[1 if t else 0 for t in x] for x in ids],
        }


class TestParserAndTokenize:
    def test_parse_and_extract(self):
        pytest.importorskip("shapely")
        from data_parsing.osm_sd_map.osm_parser import parse
        from data_parsing.osm_sd_map.osm_tokenize import extract_osm_map_data

        osm = parse(io.StringIO(_OSM_XML))
        assert len(osm.nodes) == 3 and len(osm.ways) == 1 and len(osm.relations) == 1

        pc_range = (-60.0, -30.0, -5.0, 60.0, 30.0, 3.0)
        sample = extract_osm_map_data(
            osm, ego_lat=48.0001, ego_lon=8.00015, ego_heading=0.0,
            tokenizer=_FakeTokenizer(), pc_range=pc_range, fixed_num=10, pts_dim=3,
        )
        # geometry tensors
        assert sample["osm_map_ways_pts"].shape[1:] == (10, 3)
        assert sample["osm_map_nodes_pts"].shape[-1] == 3
        # tag token lists exist and are 1D long tensors
        for t in sample["osm_map_ways_tags_input_ids"]:
            assert t.dtype.is_floating_point is False and t.dim() == 1


class _RoundTripTokenizer:
    """Char-level reversible tokenizer (2=CLS, 3=SEP, 0=PAD) so tag text
    survives encode -> decode for a realistic visualisation check."""

    def __call__(self, strings, padding=True):
        ids = [[2] + [ord(c) for c in s] + [3] for s in strings]
        maxlen = max((len(x) for x in ids), default=0)
        ids = [x + [0] * (maxlen - len(x)) for x in ids]
        return {
            "input_ids": ids,
            "token_type_ids": [[0] * len(x) for x in ids],
            "attention_mask": [[1 if t else 0 for t in x] for x in ids],
        }

    def batch_decode(self, id_lists, skip_special_tokens=True):
        return ["".join(chr(i) for i in ids if i not in (0, 2, 3)) for ids in id_lists]


def _realistic_elements():
    """A small but realistic in-patch OSM scene in the ego (forward, left) frame."""
    from shapely.geometry import LineString

    return dict(
        # a traffic signal 8 m ahead, 3 m left
        osm_map_nodes_pts=np.array([[8.0, 3.0]]),
        osm_map_nodes_tags=["highway: traffic_signals, "],
        # a main road ahead and a crossing side road
        osm_map_ways_pts=[
            LineString([(-40.0, -2.0), (0.0, -2.0), (40.0, -2.0)]),
            LineString([(0.0, -28.0), (0.0, 0.0), (0.0, 28.0)]),
        ],
        osm_map_ways_tags=[
            "highway: secondary, name: Main Street, lanes: 2, surface: asphalt, ",
            "highway: residential, name: Side Road, oneway: yes, ",
        ],
        osm_map_relations_tags=["type: associatedStreet, name: Main Street, "],
        osm_map_relations_node_member_indices=[[0]],
        osm_map_relations_way_member_indices=[[0]],
        osm_map_relations_relation_member_indices=[[]],
        osm_map_relations_node_member_tags=[["type: node, role: sign, "]],
        osm_map_relations_way_member_tags=[["type: way, role: street, "]],
        osm_map_relations_relation_member_tags=[[]],
    )


class TestVisualize:
    def test_writes_png_with_realistic_scene(self, tmp_path):
        pytest.importorskip("shapely")
        pytest.importorskip("matplotlib")
        from data_parsing.osm_sd_map.osm_tokenize import tokenize_osm_elements
        from data_parsing.osm_sd_map.visualize import visualize_osm_map_data

        pc_range = (-60.0, -30.0, -5.0, 60.0, 30.0, 3.0)
        tokenizer = _RoundTripTokenizer()
        sample = tokenize_osm_elements(_realistic_elements(), tokenizer, pc_range,
                                       fixed_num=10, pts_dim=3)

        # tags round-trip through the tokenizer so the figure shows real text
        decoded = tokenizer.batch_decode([t.tolist() for t in sample["osm_map_ways_tags_input_ids"]])
        assert any("Main Street" in d for d in decoded)

        out = tmp_path / "osm.png"
        visualize_osm_map_data(sample, pc_range, str(out), tokenizer=tokenizer)
        assert out.exists() and out.stat().st_size > 0
