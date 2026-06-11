"""Render GPS waypoints onto an ego-centric BEV map tile.

The output matches the L2D BEV map style (dark background, gray roads, bright
blue route, optional red raw GPS markers) and the L2D ego-centric framing:
the tile is centered on the ego pose and the ego heading points straight up
(forward = +y in image space). Downstream timm transforms can treat the tile
identically to the rendered map tile L2D ships.

Network fetches are slow and require internet access; this module is intended
for OFFLINE preprocessing. Pair with `cache.py` for batch use.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
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

# WGS84 mean Earth radius in meters; used by the equirectangular approximation.
EARTH_RADIUS_M = 6_378_137.0


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


def _node_xy(graph: nx.MultiDiGraph, node_id: int) -> tuple[float, float]:
    data = graph.nodes[node_id]
    return data["x"], data["y"]  # (lon, lat)


def _project_to_ego_local(
    latitudes: Sequence[float] | np.ndarray,
    longitudes: Sequence[float] | np.ndarray,
    ego_lat: float,
    ego_lon: float,
    ego_heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lat, lon) into the ego-local metric frame.

    The equirectangular approximation maps the small render window around
    the ego into a flat (east, north) plane:

        x_east  = (lon - ego_lon) * cos(ego_lat) * R * pi / 180
        y_north = (lat - ego_lat) * R * pi / 180

    The (east, north) plane is then rotated by `-ego_heading` so the ego
    forward direction points along +y in the output. `ego_heading` is in
    radians; with the convention here, `ego_heading=0` leaves the frame
    north-up.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    cos_lat = math.cos(math.radians(ego_lat))
    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    x_east = (lons - ego_lon) * cos_lat * deg_to_m
    y_north = (lats - ego_lat) * deg_to_m
    cos_h = math.cos(-ego_heading)
    sin_h = math.sin(-ego_heading)
    x_local = x_east * cos_h - y_north * sin_h
    y_local = x_east * sin_h + y_north * cos_h
    return x_local, y_local


def render_map_tile(
    graph: nx.MultiDiGraph,
    route_nodes: Sequence[int],
    ego_lat: float,
    ego_lon: float,
    ego_heading: float,
    raw_gps_points: Sequence[tuple[float, float]] | None = None,
    radius_m: int = DEFAULT_RADIUS_M,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    dpi: int = DEFAULT_DPI,
    bg_color: str = DEFAULT_BG_COLOR,
    road_color: str = DEFAULT_ROAD_COLOR,
    route_color: str = DEFAULT_ROUTE_COLOR,
    gps_color: str = DEFAULT_GPS_COLOR,
    show_raw_gps: bool = True,
) -> Image.Image:
    """Render an ego-centric BEV map tile.

    The tile is centered on `(ego_lat, ego_lon)` and rotated so the ego
    heading points up. `raw_gps_points` is a sequence of `(lat, lon)` tuples
    — same convention as the rest of this module's public API.
    """
    width_px, height_px = image_size
    fig_w_in = width_px / dpi
    fig_h_in = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    try:
        # Roads — collect every edge as a metric-frame line segment.
        edge_segments: list[list[tuple[float, float]]] = []
        for u, v in graph.edges():
            ux_lon, uy_lat = _node_xy(graph, u)
            vx_lon, vy_lat = _node_xy(graph, v)
            xs, ys = _project_to_ego_local(
                [uy_lat, vy_lat],
                [ux_lon, vx_lon],
                ego_lat,
                ego_lon,
                ego_heading,
            )
            edge_segments.append([(xs[0], ys[0]), (xs[1], ys[1])])
        if edge_segments:
            ax.add_collection(
                LineCollection(
                    edge_segments,
                    colors=road_color,
                    linewidths=0.8,
                    zorder=2,
                )
            )

        if route_nodes:
            route_lons = [graph.nodes[n]["x"] for n in route_nodes]
            route_lats = [graph.nodes[n]["y"] for n in route_nodes]
            xs, ys = _project_to_ego_local(
                route_lats, route_lons, ego_lat, ego_lon, ego_heading
            )
            if len(route_nodes) >= 2:
                ax.plot(xs, ys, color=route_color, linewidth=2.0, zorder=3)
            else:
                ax.scatter(xs, ys, color=route_color, s=12, zorder=3)

        if show_raw_gps and raw_gps_points:
            lats, lons = zip(*raw_gps_points)
            xs, ys = _project_to_ego_local(
                list(lats), list(lons), ego_lat, ego_lon, ego_heading
            )
            ax.scatter(xs, ys, color=gps_color, s=6, zorder=4)

        ax.set_xlim(-radius_m, radius_m)
        ax.set_ylim(-radius_m, radius_m)
        ax.set_aspect("equal")  # metric frame — no lat/lon distortion
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg_color, dpi=dpi, pad_inches=0)
    finally:
        plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.size != image_size:
        img = img.resize(image_size, Image.BILINEAR)
    return img


def gps_to_tensor(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    transform,
    ego_heading: float,
    ego_lat: float | None = None,
    ego_lon: float | None = None,
    radius_m: int = DEFAULT_RADIUS_M,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    dpi: int = DEFAULT_DPI,
    show_raw_gps: bool = True,
    graph: nx.MultiDiGraph | None = None,
) -> torch.Tensor:
    """End-to-end: GPS coords in, model-ready `(3, H, W)` tensor out.

    The tile is rendered ego-centric: `(ego_lat, ego_lon)` defaults to the
    last GPS sample, and `ego_heading` (radians) rotates the tile so the ego
    forward direction is up. `transform` is the timm backbone transform —
    same one used for camera tiles — which handles resize + normalization.
    Pass `graph` to skip the Overpass fetch when the network has been cached
    for the area.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    if lats.shape != lons.shape:
        raise ValueError("latitudes and longitudes must have the same shape")
    if lats.size == 0:
        raise ValueError("at least one GPS point is required")

    if ego_lat is None:
        ego_lat = float(lats[-1])
    if ego_lon is None:
        ego_lon = float(lons[-1])

    if graph is None:
        graph = fetch_road_network(ego_lat, ego_lon, radius_m=radius_m)

    _, route = map_match_waypoints(graph, lats.tolist(), lons.tolist())
    raw_points = list(zip(lats.tolist(), lons.tolist()))

    image = render_map_tile(
        graph,
        route_nodes=route,
        ego_lat=ego_lat,
        ego_lon=ego_lon,
        ego_heading=ego_heading,
        raw_gps_points=raw_points,
        radius_m=radius_m,
        image_size=image_size,
        dpi=dpi,
        show_raw_gps=show_raw_gps,
    )

    tensor = transform(image)
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"transform must return a (3, H, W) tensor, got shape {tuple(tensor.shape)}"
        )
    return tensor
