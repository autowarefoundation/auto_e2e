"""Caching helpers for the map rendering pipeline.

Network fetches via osmnx are slow (seconds each, internet required). This
module persists fetched graphs to disk and renders/persists tiles for an
entire dataset in one batch so the DataLoader only ever reads PNGs.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .gps_to_map import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RADIUS_M,
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
        network_cache_dir: if given, fetched graphs are persisted here keyed by
            centroid so neighbouring clips share a cached download.
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

        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        graph = _load_or_fetch_network(
            center_lat, center_lon, radius_m, net_cache
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
                raw_gps_points=raw_points,
                image_size=image_size,
            )
        except Exception as exc:  # noqa: BLE001 — matplotlib/osmnx errors vary
            logger.warning("clip %s: render failed (%s); skipping", clip_id, exc)
            continue

        image.save(tile_path)
        rendered.append(tile_path)

    return rendered


def _load_or_fetch_network(
    center_lat: float,
    center_lon: float,
    radius_m: int,
    cache_dir: Path | None,
) -> nx.MultiDiGraph | None:
    """Return a cached graph if available, otherwise fetch and cache it.

    Centroids are quantized to ~100 m so nearby clips reuse the same download.
    """
    if cache_dir is not None:
        key = f"{round(center_lat, 3)}_{round(center_lon, 3)}_{radius_m}.pkl"
        cache_path = cache_dir / key
        cached = load_cached_network(cache_path)
        if cached is not None:
            return cached
    else:
        cache_path = None

    try:
        graph = fetch_road_network(center_lat, center_lon, radius_m=radius_m)
    except Exception as exc:  # noqa: BLE001 — network/Overpass failures
        logger.warning(
            "fetch_road_network(%.4f, %.4f) failed: %s",
            center_lat,
            center_lon,
            exc,
        )
        return None

    if cache_path is not None:
        cache_network(graph, cache_path)
    return graph
