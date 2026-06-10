"""Render GPS waypoints onto an OpenStreetMap-derived BEV map tile.

The output matches the L2D BEV map style (dark background, gray roads, bright
blue route, optional red raw GPS markers) so a downstream timm transform can
treat it identically to the rendered map tile L2D ships.

Network fetches are slow and require internet access; this module is intended
for OFFLINE preprocessing. Pair with `cache.py` for batch use.
"""

from __future__ import annotations

import io
import logging
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

logger = logging.getLogger(__name__)


# L2D BEV map palette
DEFAULT_BG_COLOR = "#111111"
DEFAULT_ROAD_COLOR = "#444444"
DEFAULT_ROUTE_COLOR = "#00CCFF"
DEFAULT_GPS_COLOR = "#FF3333"

DEFAULT_IMAGE_SIZE = (640, 360)  # (W, H), matches L2D
DEFAULT_RADIUS_M = 800
DEFAULT_DPI = 200


def fetch_road_network(
    center_lat: float,
    center_lon: float,
    radius_m: int = DEFAULT_RADIUS_M,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """Download the OSM road network within a radius of a GPS point.

    Hits Overpass API; expect network latency on the order of seconds. Cache
    aggressively via `cache.py` if you call this repeatedly for nearby points.
    """
    return ox.graph_from_point(
        (center_lat, center_lon),
        dist=radius_m,
        network_type=network_type,
    )


def map_match_waypoints(
    graph: nx.MultiDiGraph,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
) -> tuple[list[int], list[int]]:
    """Snap a GPS trace onto graph nodes and stitch them into a connected route.

    Returns:
        matched_nodes: nearest graph node for each input waypoint (same length).
        route_nodes:   the full node sequence after shortest-path stitching
                       between consecutive matches; empty when matching fails.

    Map matching can fail (no edges in radius, disconnected components, GPS
    outside the graph bbox); callers should treat an empty `route_nodes` as
    "render raw GPS only".
    """
    if len(latitudes) != len(longitudes):
        raise ValueError("latitudes and longitudes must be the same length")
    if not latitudes:
        return [], []

    try:
        matched_nodes = list(
            ox.distance.nearest_nodes(graph, list(longitudes), list(latitudes))
        )
    except Exception as exc:  # noqa: BLE001 — osmnx raises a variety of errors
        logger.warning("nearest_nodes failed: %s", exc)
        return [], []

    route: list[int] = []
    for src, dst in zip(matched_nodes[:-1], matched_nodes[1:]):
        if src == dst:
            if not route or route[-1] != src:
                route.append(src)
            continue
        try:
            segment = nx.shortest_path(graph, src, dst, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.debug("no path %s -> %s: %s", src, dst, exc)
            continue
        if route and route[-1] == segment[0]:
            segment = segment[1:]
        route.extend(segment)

    if not route and matched_nodes:
        route = [matched_nodes[0]]

    return matched_nodes, route
