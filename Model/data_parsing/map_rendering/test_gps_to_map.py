"""Unit tests for the map_rendering package.

These tests must run offline — `osmnx.graph_from_point` is patched out, and
all graphs are constructed locally.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import networkx as nx
import pytest
import torch
from PIL import Image

from data_parsing.map_rendering import (
    fetch_road_network,
    gps_to_tensor,
    map_match_waypoints,
    render_map_tile,
)
from data_parsing.map_rendering.cache import cache_network, load_cached_network


def _toy_graph() -> nx.MultiDiGraph:
    """Three-node graph along a fixed meridian (lon=0), 100 m apart in lat."""
    graph = nx.MultiDiGraph(crs="epsg:4326")
    nodes = [
        (1, 35.0000, 0.0000),
        (2, 35.0010, 0.0000),
        (3, 35.0020, 0.0000),
    ]
    for nid, lat, lon in nodes:
        graph.add_node(nid, x=lon, y=lat)
    graph.add_edge(1, 2, length=111.0)
    graph.add_edge(2, 1, length=111.0)
    graph.add_edge(2, 3, length=111.0)
    graph.add_edge(3, 2, length=111.0)
    return graph


def test_render_returns_pil_image():
    graph = _toy_graph()
    with mock.patch(
        "data_parsing.map_rendering.gps_to_map.ox.plot_graph"
    ) as plot_graph:
        plot_graph.return_value = (None, None)
        img = render_map_tile(
            graph,
            route_nodes=[1, 2, 3],
            raw_gps_points=[(35.0005, 0.0), (35.0015, 0.0)],
            image_size=(640, 360),
            dpi=100,
        )
    assert isinstance(img, Image.Image)
    assert img.size == (640, 360)
    assert img.mode == "RGB"


def test_tensor_output_shape():
    graph = _toy_graph()

    def fake_transform(image: Image.Image) -> torch.Tensor:
        # Mimic a timm transform: resize + to_tensor + normalize -> (3, 224, 224)
        arr = torch.zeros(3, 224, 224)
        return arr

    with mock.patch(
        "data_parsing.map_rendering.gps_to_map.ox.plot_graph"
    ) as plot_graph, mock.patch(
        "data_parsing.map_rendering.gps_to_map.ox.distance.nearest_nodes",
        return_value=[1, 2, 3],
    ):
        plot_graph.return_value = (None, None)
        tensor = gps_to_tensor(
            latitudes=[35.0000, 35.0010, 35.0020],
            longitudes=[0.0, 0.0, 0.0],
            transform=fake_transform,
            graph=graph,
        )

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_map_match_with_simple_graph():
    graph = _toy_graph()
    with mock.patch(
        "data_parsing.map_rendering.gps_to_map.ox.distance.nearest_nodes",
        return_value=[1, 3],
    ):
        matched, route = map_match_waypoints(
            graph,
            latitudes=[35.0000, 35.0020],
            longitudes=[0.0, 0.0],
        )

    assert matched == [1, 3]
    # Stitching 1 -> 3 must traverse node 2.
    assert route == [1, 2, 3]


def test_map_match_empty_input():
    graph = _toy_graph()
    matched, route = map_match_waypoints(graph, [], [])
    assert matched == []
    assert route == []


def test_map_match_length_mismatch():
    graph = _toy_graph()
    with pytest.raises(ValueError):
        map_match_waypoints(graph, [35.0], [0.0, 0.1])


def test_cache_round_trip(tmp_path: Path):
    graph = _toy_graph()
    cache_path = tmp_path / "net.pkl"

    cache_network(graph, cache_path)
    assert cache_path.exists()

    loaded = load_cached_network(cache_path)
    assert loaded is not None
    assert set(loaded.nodes) == set(graph.nodes)
    assert loaded.number_of_edges() == graph.number_of_edges()


def test_load_cached_network_missing(tmp_path: Path):
    assert load_cached_network(tmp_path / "does-not-exist.pkl") is None


def test_fetch_road_network_calls_osmnx():
    sentinel = _toy_graph()
    with mock.patch(
        "data_parsing.map_rendering.gps_to_map.ox.graph_from_point",
        return_value=sentinel,
    ) as fake:
        result = fetch_road_network(35.0, 0.0, radius_m=500)
    fake.assert_called_once()
    assert result is sentinel
