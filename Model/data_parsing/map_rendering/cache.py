"""Caching helpers for the map rendering pipeline.

Network fetches via osmnx are slow (seconds each, internet required). This
module persists fetched graphs to disk and renders/persists tiles for an
entire dataset in one batch so the DataLoader only ever reads PNGs.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx

from .gps_to_map import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RADIUS_M,
    EARTH_RADIUS_M,
    fetch_road_network,
    map_match_waypoints,
    render_map_tile,
)

logger = logging.getLogger(__name__)


def cache_network(graph: nx.MultiDiGraph, filepath: str | Path) -> None:
    """Pickle a road-network graph to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)


def load_cached_network(filepath: str | Path) -> nx.MultiDiGraph | None:
    """Load a pickled road-network graph, or `None` if the file is missing."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError) as exc:
        logger.warning("failed to load cached network %s: %s", path, exc)
        return None


def render_and_cache_tiles(
    dataset_gps_data: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    output_dir: str | Path,
    radius_m: int = DEFAULT_RADIUS_M,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    network_cache_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Pre-render and persist a BEV map tile for every clip in a dataset.

    Args:
        dataset_gps_data: mapping of `clip_id -> (latitudes, longitudes)`.
        output_dir: where rendered PNG tiles are written (`{clip_id}.png`).
        radius_m: render radius around each clip's centroid.
        image_size: output `(W, H)`.
        network_cache_dir: if given, fetched graphs and the shared raw-OSM XML
            are persisted here (keyed by the clip's trajectory bbox) so
            neighbouring clips and the OSM vector branch reuse downloads.
        skip_existing: do not re-render clips whose PNG already exists.

    Returns:
        List of paths to the rendered tile files (including pre-existing ones).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    net_cache = Path(network_cache_dir) if network_cache_dir else None
    if net_cache is not None:
        net_cache.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    for clip_id, (lats, lons) in dataset_gps_data.items():
        tile_path = out / f"{clip_id}.png"
        if skip_existing and tile_path.exists():
            rendered.append(tile_path)
            continue

        if not lats:
            logger.warning("clip %s has no GPS samples; skipping", clip_id)
            continue

        ego_lat = float(lats[-1])
        ego_lon = float(lons[-1])
        ego_heading = _heading_from_trace(lats, lons, ego_lat)

        # Fetch sized to the whole trajectory + the render radius, so the route
        # and the ±radius draw window are always covered (a centroid radius can
        # miss long clips).
        graph = _load_or_fetch_network(
            list(lats), list(lons), radius_m, net_cache
        )
        if graph is None:
            logger.warning("clip %s: failed to obtain road network; skipping", clip_id)
            continue

        _, route = map_match_waypoints(graph, list(lats), list(lons))
        raw_points = list(zip(lats, lons))
        try:
            image = render_map_tile(
                graph,
                route_nodes=route,
                ego_lat=ego_lat,
                ego_lon=ego_lon,
                ego_heading=ego_heading,
                raw_gps_points=raw_points,
                radius_m=radius_m,
                image_size=image_size,
            )
        except Exception as exc:  # noqa: BLE001 — matplotlib/osmnx errors vary
            logger.warning(
                "clip %s: render failed (%s); skipping",
                clip_id,
                exc,
                exc_info=True,
            )
            continue

        image.save(tile_path)
        rendered.append(tile_path)

    return rendered


def _heading_from_trace(
    lats: Sequence[float], lons: Sequence[float], ref_lat: float
) -> float:
    """Estimate ego heading (radians) from the last segment of the GPS trace.

    Uses atan2(east, north) so 0 rad ≡ north and the value matches the
    `ego_heading` convention in `render_map_tile`. Falls back to 0 when the
    trace has fewer than two distinct samples.
    """
    if len(lats) < 2:
        return 0.0
    cos_lat = math.cos(math.radians(ref_lat))
    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    dx = (lons[-1] - lons[-2]) * cos_lat * deg_to_m
    dy = (lats[-1] - lats[-2]) * deg_to_m
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dx, dy)


def _graph_from_shared_osm(
    lats: Sequence[float],
    lons: Sequence[float],
    margin_m: int,
    cache_dir: Path | None,
) -> nx.MultiDiGraph | None:
    """Build the road graph from the shared raw-OSM cache (trace bbox + margin).

    The SD-map (OSM vector) branch and this rendered-tile branch share a single
    on-disk artifact: the raw Overpass OSM XML (see
    ``data_parsing.osm_sd_map.overpass``). Both fetch the bounding box of the
    clip's trajectory expanded by their own margin (here the render radius); the
    shared cache reuses any already-fetched XML that covers the request, so one
    download serves both. Returns ``None`` (caller falls back to a direct fetch)
    if the shared path is unavailable for any reason.
    """
    try:
        import os
        import tempfile

        import osmnx as ox

        from ..osm_sd_map.overpass import load_or_fetch_osm_for_trace
    except Exception as exc:  # noqa: BLE001 — optional path; never break rendering
        logger.debug("shared OSM path unavailable (%s); using direct fetch", exc)
        return None

    xml = load_or_fetch_osm_for_trace(lats, lons, margin_m, cache_dir)
    if xml is None:
        return None

    tmp = tempfile.NamedTemporaryFile("w", suffix=".osm", delete=False, encoding="utf-8")
    try:
        tmp.write(xml)
        tmp.close()
        return ox.graph_from_xml(tmp.name, bidirectional=False, simplify=True, retain_all=True)
    except Exception as exc:  # noqa: BLE001 — osmnx version/parse differences
        logger.warning("graph_from_xml failed (%s); using direct fetch", exc)
        return None
    finally:
        os.unlink(tmp.name)


def _trace_centroid_radius(
    lats: Sequence[float], lons: Sequence[float], margin_m: int
) -> tuple[float, float, int]:
    """Centroid + a radius that covers the whole trace plus ``margin_m``."""
    clat = sum(lats) / len(lats)
    clon = sum(lons) / len(lons)
    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    cos_lat = math.cos(math.radians(clat))
    max_d = 0.0
    for la, lo in zip(lats, lons):
        dx = (lo - clon) * cos_lat * deg_to_m
        dy = (la - clat) * deg_to_m
        max_d = max(max_d, math.hypot(dx, dy))
    return clat, clon, int(max_d + margin_m)


def _graph_cache_path(
    lats: Sequence[float], lons: Sequence[float], margin_m: int, cache_dir: Path
) -> Path | None:
    """Pickle path for the built graph, keyed by the trace bbox."""
    try:
        from ..osm_sd_map.overpass import bbox_from_points
    except Exception:  # noqa: BLE001
        return None
    s, w, n, e = bbox_from_points(lats, lons, margin_m)
    return cache_dir / f"graph_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}.pkl"


def _load_or_fetch_network(
    lats: Sequence[float],
    lons: Sequence[float],
    margin_m: int,
    cache_dir: Path | None,
) -> nx.MultiDiGraph | None:
    """Return a cached graph if available, otherwise build/fetch and cache it.

    The fetch is sized to the trajectory bounding box plus ``margin_m`` (the
    render radius), so long clips are fully covered (a fixed centroid radius can
    miss them). The graph is built from the shared raw-OSM cache when possible
    (reused by the OSM vector branch via bbox containment); otherwise it falls
    back to a direct osmnx fetch sized to cover the trace. The built graph is
    memoized as a pickle keyed by the trace bbox.
    """
    cache_path = _graph_cache_path(lats, lons, margin_m, cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cached = load_cached_network(cache_path)
        if cached is not None:
            return cached

    graph = _graph_from_shared_osm(lats, lons, margin_m, cache_dir)
    if graph is None:
        clat, clon, radius = _trace_centroid_radius(lats, lons, margin_m)
        try:
            graph = fetch_road_network(clat, clon, radius_m=radius)
        except Exception as exc:  # noqa: BLE001 — network/Overpass failures
            logger.warning(
                "fetch_road_network(%.4f, %.4f) failed: %s", clat, clon, exc
            )
            return None

    if cache_path is not None:
        cache_network(graph, cache_path)
    return graph
