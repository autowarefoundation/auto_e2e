"""Read the drivable OSM ways near L2D tracks out of a country-sized PBF.

A whole-Germany lane graph does not fit in memory, so the PBF is streamed
once and only drivable ways with a node inside a buffer around the episode
tracks are kept, together with the tags of their nodes. The buffer is far
wider than the 120 m by 60 m raster so map matching keeps its connectivity.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

from .osm_lane_graph import DRIVABLE_HIGHWAYS, OSMNode, OSMWay

CELL_DEG = 0.01
# Node tags that become HD-map semantics; everything else is dropped.
_POINT_TAG_KEYS = ("highway", "crossing", "traffic_signals:direction", "direction")


def file_sha256(path: str | Path, chunk_size: int = 16 << 20) -> str:
    """Stream the digest so a multi-gigabyte PBF is never held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clip_cells(points_lon_lat: Iterable[tuple[float, float]], buffer_m: float) -> frozenset[tuple[int, int]]:
    """Grid cells within ``buffer_m`` of any track point."""
    if buffer_m <= 0.0:
        raise ValueError("buffer_m must be positive")
    cells: set[tuple[int, int]] = set()
    for longitude, latitude in points_lon_lat:
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            continue
        lat_cells = math.ceil(buffer_m / (111_320.0 * CELL_DEG))
        lon_cells = math.ceil(buffer_m / (111_320.0 * max(math.cos(math.radians(latitude)), 0.1) * CELL_DEG))
        base_lat, base_lon = math.floor(latitude / CELL_DEG), math.floor(longitude / CELL_DEG)
        for d_lat in range(-lat_cells, lat_cells + 1):
            for d_lon in range(-lon_cells, lon_cells + 1):
                cells.add((base_lat + d_lat, base_lon + d_lon))
    if not cells:
        raise ValueError("no finite track points to clip around")
    return frozenset(cells)


def _cell(longitude: float, latitude: float) -> tuple[int, int]:
    return math.floor(latitude / CELL_DEG), math.floor(longitude / CELL_DEG)


def read_region(pbf_path: str | Path, cells: frozenset[tuple[int, int]]) -> tuple[dict[int, OSMNode], list[OSMWay]]:
    """Drivable ways touching ``cells`` and every node they reference."""
    import osmium

    path = str(pbf_path)
    ways: list[OSMWay] = []
    locations: dict[int, tuple[float, float]] = {}
    for item in osmium.FileProcessor(path, osmium.osm.NODE | osmium.osm.WAY).with_locations():
        if not item.is_way():
            continue
        highway = item.tags.get("highway")
        if highway not in DRIVABLE_HIGHWAYS:
            continue
        refs = [(node.ref, node.lon, node.lat) for node in item.nodes if node.location.valid()]
        if len(refs) < 2 or not any(_cell(lon, lat) in cells for _, lon, lat in refs):
            continue
        ways.append(OSMWay(int(item.id), tuple(int(ref) for ref, _, _ in refs), {tag.k: tag.v for tag in item.tags}))
        for ref, lon, lat in refs:
            locations[int(ref)] = (float(lon), float(lat))
    tags: dict[int, dict[str, str]] = {}
    for item in osmium.FileProcessor(path, osmium.osm.NODE):
        if item.id in locations and len(item.tags):
            kept = {key: item.tags[key] for key in _POINT_TAG_KEYS if key in item.tags}
            if kept:
                tags[int(item.id)] = kept
    nodes = {
        node_id: OSMNode(node_id, lon, lat, tags.get(node_id, {}))
        for node_id, (lon, lat) in locations.items()
    }
    return nodes, sorted(ways, key=lambda way: way.way_id)
