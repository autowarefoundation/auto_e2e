"""Read the drivable OSM ways near L2D tracks out of a country-sized PBF.

A whole-Germany lane graph does not fit in memory, so the PBF is streamed
and only drivable ways with a node inside a buffer around the episode tracks
are kept, together with the tags of their nodes. The buffer is far
wider than the 120 m by 60 m raster so map matching keeps its connectivity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
    """Drivable ways touching ``cells`` and every node they reference.

    Three plain passes instead of one pass with a node location index: the
    index for a whole-country PBF needs several gigabytes, while the node ids
    near the tracks and the ids their ways reference stay small.
    """
    import osmium

    path = str(pbf_path)
    near: set[int] = set()
    for item in osmium.FileProcessor(path, osmium.osm.NODE):
        assert isinstance(item, osmium.osm.Node)
        if _cell(item.lon, item.lat) in cells:
            near.add(int(item.id))
    ways: list[OSMWay] = []
    referenced: set[int] = set()
    for item in osmium.FileProcessor(path, osmium.osm.WAY):
        assert isinstance(item, osmium.osm.Way)
        if item.tags.get("highway") not in DRIVABLE_HIGHWAYS:
            continue
        refs = tuple(int(node.ref) for node in item.nodes)
        if len(refs) < 2 or near.isdisjoint(refs):
            continue
        ways.append(OSMWay(int(item.id), refs, {tag.k: tag.v for tag in item.tags}))
        referenced.update(refs)
    nodes: dict[int, OSMNode] = {}
    for item in osmium.FileProcessor(path, osmium.osm.NODE):
        assert isinstance(item, osmium.osm.Node)
        if item.id not in referenced:
            continue
        kept = {key: item.tags[key] for key in _POINT_TAG_KEYS if key in item.tags}
        nodes[int(item.id)] = OSMNode(int(item.id), float(item.lon), float(item.lat), kept)
    # A way cut by the extract boundary keeps only the nodes the extract has.
    complete = []
    for way in ways:
        refs = tuple(ref for ref in way.node_ids if ref in nodes)
        if len(refs) >= 2:
            complete.append(OSMWay(way.way_id, refs, way.tags))
    return nodes, sorted(complete, key=lambda way: way.way_id)


OSM_REGION_SCHEMA_VERSION = "l2d_osm_region_v1"


def region_bytes(
    nodes: Mapping[int, OSMNode],
    ways: Iterable[OSMWay],
    *,
    provenance: Mapping[str, Any],
) -> bytes:
    """Canonical JSON of a region, so the same PBF and tracks give the same digest."""
    payload = {
        "schema_version": OSM_REGION_SCHEMA_VERSION,
        "provenance": dict(provenance),
        "nodes": [
            [node.node_id, node.longitude_deg, node.latitude_deg, dict(sorted(node.tags.items()))]
            for node in sorted(nodes.values(), key=lambda node: node.node_id)
        ],
        "ways": [
            [way.way_id, list(way.node_ids), dict(sorted(way.tags.items()))]
            for way in sorted(ways, key=lambda way: way.way_id)
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_region(payload: bytes) -> tuple[dict[int, OSMNode], list[OSMWay], dict[str, Any]]:
    region = json.loads(payload)
    if region.get("schema_version") != OSM_REGION_SCHEMA_VERSION:
        raise ValueError("unsupported L2D OSM region schema")
    nodes = {
        int(node_id): OSMNode(int(node_id), float(lon), float(lat), dict(tags))
        for node_id, lon, lat, tags in region["nodes"]
    }
    ways = [OSMWay(int(way_id), tuple(int(ref) for ref in refs), dict(tags)) for way_id, refs, tags in region["ways"]]
    return nodes, ways, dict(region["provenance"])


def subregion(
    nodes: Mapping[int, OSMNode],
    ways: Iterable[OSMWay],
    cells: frozenset[tuple[int, int]],
) -> tuple[dict[int, OSMNode], list[OSMWay]]:
    """The ways of a region that touch ``cells``, with the nodes they use."""
    kept = [
        way for way in ways
        if any(_cell(nodes[ref].longitude_deg, nodes[ref].latitude_deg) in cells for ref in way.node_ids)
    ]
    used = {ref for way in kept for ref in way.node_ids}
    return {ref: nodes[ref] for ref in used}, kept
