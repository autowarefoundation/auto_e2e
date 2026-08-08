"""Deterministic offline OSM PBF to L2D canonical graph conversion."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from navigation.contracts import canonical_json_bytes

from .navigation import L2D_OSM_GRAPH_SCHEMA_VERSION

L2D_OSM_GRAPH_ADAPTER_VERSION = "l2d_osm_pbf_adapter_v1"

_DRIVABLE_HIGHWAYS = frozenset({
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "secondary",
    "secondary_link",
    "service",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "unclassified",
})


@dataclasses.dataclass(frozen=True)
class OSMWayRecord:
    """Minimal deterministic road-way representation."""

    way_id: str
    node_ids: tuple[str, ...]
    highway: str
    oneway: str = "no"
    lanes: str | int | None = None
    width_m: str | float | None = None


def _haversine_m(
    first_lon_lat: tuple[float, float],
    second_lon_lat: tuple[float, float],
) -> float:
    longitude_1, latitude_1 = map(math.radians, first_lon_lat)
    longitude_2, latitude_2 = map(math.radians, second_lon_lat)
    delta_lon = longitude_2 - longitude_1
    delta_lat = latitude_2 - latitude_1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6_378_137.0 * math.asin(min(1.0, math.sqrt(value)))


def _parse_width_m(value: str | float | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    multiplier = 1.0
    if text.endswith(" ft"):
        multiplier = 0.3048
        text = text[:-3].strip()
    elif text.endswith("m"):
        text = text[:-1].strip()
    try:
        width = float(text) * multiplier
    except ValueError:
        return None
    if not math.isfinite(width) or width <= 0.0:
        return None
    return min(width, 30.0)


def _directions(oneway: str) -> tuple[str, ...]:
    normalized = str(oneway).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return ("forward",)
    if normalized == "-1":
        return ("reverse",)
    return ("forward", "reverse")


def encode_l2d_osm_graph_snapshot(
    nodes_lon_lat: Mapping[str, tuple[float, float]],
    ways: Sequence[OSMWayRecord],
    *,
    source_revision: str,
    source_date: str,
    source_artifact_sha256: str,
    attribution: str,
) -> bytes:
    """Encode a graph snapshot independent of PBF record ordering."""
    if not source_revision or not source_date or not attribution:
        raise ValueError("OSM provenance fields must not be empty")
    if (
        len(source_artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_artifact_sha256
        )
    ):
        raise ValueError("OSM source artifact digest must be lowercase SHA-256")

    normalized_nodes: dict[str, tuple[float, float]] = {}
    for raw_node_id, raw_lon_lat in nodes_lon_lat.items():
        node_id = str(raw_node_id)
        longitude, latitude = map(float, raw_lon_lat)
        if (
            not node_id
            or not math.isfinite(longitude)
            or not math.isfinite(latitude)
            or not -180.0 <= longitude <= 180.0
            or not -90.0 <= latitude <= 90.0
        ):
            raise ValueError("OSM node coordinate is invalid")
        normalized_nodes[node_id] = (longitude, latitude)

    used_nodes: set[str] = set()
    edges: list[dict[str, Any]] = []
    for way in sorted(ways, key=lambda item: item.way_id):
        if way.highway not in _DRIVABLE_HIGHWAYS:
            continue
        node_ids = tuple(str(node_id) for node_id in way.node_ids)
        if len(node_ids) < 2:
            continue
        missing = set(node_ids) - set(normalized_nodes)
        if missing:
            raise ValueError(
                f"OSM way {way.way_id!r} references missing nodes"
            )
        width_m = _parse_width_m(way.width_m)
        for segment_index, (first, second) in enumerate(
            zip(node_ids[:-1], node_ids[1:])
        ):
            if first == second:
                continue
            first_lon_lat = normalized_nodes[first]
            second_lon_lat = normalized_nodes[second]
            length_m = _haversine_m(first_lon_lat, second_lon_lat)
            if not math.isfinite(length_m) or length_m <= 0.0:
                continue
            used_nodes.update((first, second))
            for direction in _directions(way.oneway):
                source, destination = (
                    (first, second)
                    if direction == "forward"
                    else (second, first)
                )
                geometry = (
                    [first_lon_lat, second_lon_lat]
                    if direction == "forward"
                    else [second_lon_lat, first_lon_lat]
                )
                edge: dict[str, Any] = {
                    "destination": destination,
                    "geometry_lon_lat": geometry,
                    "highway": way.highway,
                    "key": (
                        f"{way.way_id}:{segment_index}:"
                        f"{direction[0]}"
                    ),
                    "length_m": length_m,
                    "oneway": way.oneway,
                    "source": source,
                }
                if way.lanes is not None:
                    edge["lanes"] = way.lanes
                if width_m is not None:
                    edge["width_m"] = width_m
                edges.append(edge)

    if not edges:
        raise ValueError("OSM source contains no supported drivable ways")
    edges.sort(
        key=lambda edge: (
            edge["source"],
            edge["destination"],
            edge["key"],
        )
    )
    payload = {
        "adapter_version": L2D_OSM_GRAPH_ADAPTER_VERSION,
        "attribution": attribution,
        "edges": edges,
        "nodes": [
            {
                "id": node_id,
                "latitude_deg": normalized_nodes[node_id][1],
                "longitude_deg": normalized_nodes[node_id][0],
            }
            for node_id in sorted(used_nodes)
        ],
        "schema_version": L2D_OSM_GRAPH_SCHEMA_VERSION,
        "source_artifact_sha256": source_artifact_sha256,
        "source_date": source_date,
        "source_revision": source_revision,
    }
    return canonical_json_bytes(payload)


def build_l2d_osm_graph_snapshot(
    pbf_path: str | Path,
    output_path: str | Path,
    *,
    source_revision: str,
    source_date: str,
    attribution: str = "OpenStreetMap contributors",
) -> str:
    """Build one immutable canonical graph from a local `.osm.pbf` file."""
    try:
        import osmium
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "offline OSM PBF conversion requires the pinned osmium package"
        ) from exc

    source = Path(pbf_path)
    if source.suffixes[-2:] != [".osm", ".pbf"]:
        raise ValueError("OSM source must use the .osm.pbf suffix")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    nodes: dict[str, tuple[float, float]] = {}
    ways: list[OSMWayRecord] = []

    class Handler(osmium.SimpleHandler):
        def way(self, way: Any) -> None:
            highway = way.tags.get("highway")
            if highway not in _DRIVABLE_HIGHWAYS:
                return
            node_ids = []
            for node in way.nodes:
                if not node.location.valid():
                    raise ValueError(
                        f"OSM way {way.id} has an invalid node location"
                    )
                node_id = str(node.ref)
                nodes[node_id] = (
                    float(node.location.lon),
                    float(node.location.lat),
                )
                node_ids.append(node_id)
            ways.append(OSMWayRecord(
                way_id=str(way.id),
                node_ids=tuple(node_ids),
                highway=str(highway),
                oneway=str(way.tags.get("oneway", "no")),
                lanes=way.tags.get("lanes"),
                width_m=way.tags.get("width"),
            ))

    Handler().apply_file(str(source), locations=True, idx="flex_mem")
    payload = encode_l2d_osm_graph_snapshot(
        nodes,
        ways,
        source_revision=source_revision,
        source_date=source_date,
        source_artifact_sha256=source_sha256,
        attribution=attribution,
    )
    destination = Path(output_path)
    if destination.exists() and destination.read_bytes() != payload:
        raise FileExistsError(
            "refusing to replace a different immutable OSM graph snapshot"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
