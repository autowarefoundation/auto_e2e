"""Overpass fetch + shared raw-OSM cache.

The Overpass query here is taken verbatim from SDTagNet's
``nlp_pretraining/get_av2_sd_maps.ipynb`` / ``get_nusc_sd_maps.ipynb`` — it is
tuned to pull every node/way/relation (with members) inside the bbox plus the
relations that reference them, *without* dragging in unrelated elements:

    (node(S,W,N,E); rel(bn)->.x; way(S,W,N,E); node(w)->.x; rel(bw);); out meta;

The cached artifact is the **raw OSM XML**, which is the single shared source
for both map branches:
  * the SD-map (SDTagNet) branch parses it with ``osm_parser.parse``;
  * the rendered-tile branch (``map_rendering``) builds an osmnx graph from it
    via ``ox.graph_from_xml`` — see ``map_rendering.cache``.

XML is cached gzip-compressed, keyed by quantized centroid (same ~100 m
quantization the rendered-tile network cache already uses), so neighbouring
clips reuse one Overpass download.
"""

from __future__ import annotations

import gzip
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Mirrors the headers used in the SDTagNet retrieval notebooks. Set a unique
# User-Agent for your own large-scale use to be polite to the public endpoint.
DEFAULT_HEADERS = {
    "Connection": "keep-alive",
    "Accept": "*/*",
    "User-Agent": "auto_e2e-sd-map",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://overpass-turbo.eu",
    "Referer": "https://overpass-turbo.eu/",
}

EARTH_RADIUS_M = 6_378_137.0


def build_query(south, west, north, east):
    """The exact tested SDTagNet Overpass query for a bbox (S, W, N, E)."""
    return (
        "(node({0},{1},{2},{3}); rel(bn)->.x; way({0},{1},{2},{3}); "
        "node(w)->.x; rel(bw);); out meta;"
    ).format(south, west, north, east)


def bbox_from_center(center_lat, center_lon, radius_m):
    """Axis-aligned (S, W, N, E) bbox covering a radius around a point."""
    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    dlat = radius_m / deg_to_m
    dlon = radius_m / (deg_to_m * math.cos(math.radians(center_lat)))
    return (center_lat - dlat, center_lon - dlon,
            center_lat + dlat, center_lon + dlon)


def bbox_from_points(latitudes, longitudes, margin_m):
    """(S, W, N, E) bbox covering all points plus a metric margin on each side.

    This is the trajectory-aware fetch extent: it tightly bounds an episode's
    driven path rather than a fixed radius around its centroid, so it both
    covers long episodes (which a centroid radius can miss) and stays small for
    short ones. ``margin_m`` is the per-side reach the consumer needs beyond the
    trajectory (the SD-map ``pc_range`` patch reach, or the renderer's draw
    radius).
    """
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(longitudes), max(longitudes)
    center_lat = (lat_min + lat_max) / 2.0
    deg_to_m = EARTH_RADIUS_M * math.pi / 180.0
    dlat = margin_m / deg_to_m
    dlon = margin_m / (deg_to_m * math.cos(math.radians(center_lat)))
    return (lat_min - dlat, lon_min - dlon, lat_max + dlat, lon_max + dlon)


def fetch_raw_osm(south, west, north, east, url=OVERPASS_URL, headers=None, timeout=180):
    """POST the tested query to Overpass and return the raw OSM XML text."""
    import requests

    query = build_query(south, west, north, east)
    response = requests.post(
        url, headers=headers or DEFAULT_HEADERS, data={"data": query}, timeout=timeout
    )
    response.raise_for_status()
    return response.text


_BBOX_PREFIX = "osm_bbox_"
_BBOX_SUFFIX = ".xml.gz"


def _bbox_cache_name(south, west, north, east):
    return f"{_BBOX_PREFIX}{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}{_BBOX_SUFFIX}"


def _parse_bbox_name(name):
    if not (name.startswith(_BBOX_PREFIX) and name.endswith(_BBOX_SUFFIX)):
        return None
    parts = name[len(_BBOX_PREFIX):-len(_BBOX_SUFFIX)].split("_")
    if len(parts) != 4:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def _covers(cached, requested, eps=1e-9):
    """True if cached (S,W,N,E) bbox fully contains the requested one."""
    cs, cw, cn, ce = cached
    rs, rw, rn, re = requested
    return cs <= rs + eps and cw <= rw + eps and cn >= rn - eps and ce >= re - eps


def _read_gz(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        logger.warning("failed to read cached OSM %s: %s", path, exc)
        return None


def load_or_fetch_osm_bbox(south, west, north, east, cache_dir,
                           url=OVERPASS_URL, headers=None, timeout=180,
                           reuse_covering=True):
    """Return raw OSM XML for a bbox, using a gzip cache keyed by the bbox.

    The cache is shared by both map branches. Because they request different
    margins, a request is also served by any *already cached* XML whose bbox
    fully contains it (``reuse_covering``) — so e.g. the SD-map branch reuses
    the renderer's larger fetch for the same area instead of downloading again.

    Returns the XML text, or ``None`` if the fetch fails.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    req = (south, west, north, east)

    if cache_dir is not None and cache_dir.exists():
        exact = cache_dir / _bbox_cache_name(*req)
        if exact.exists():
            xml = _read_gz(exact)
            if xml is not None:
                return xml
        if reuse_covering:
            for p in cache_dir.glob(f"{_BBOX_PREFIX}*{_BBOX_SUFFIX}"):
                cached = _parse_bbox_name(p.name)
                if cached is not None and _covers(cached, req):
                    xml = _read_gz(p)
                    if xml is not None:
                        return xml

    try:
        xml = fetch_raw_osm(south, west, north, east, url=url, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — network/Overpass failures vary
        logger.warning("Overpass fetch bbox (%.4f, %.4f, %.4f, %.4f) failed: %s",
                       south, west, north, east, exc)
        return None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_dir / _bbox_cache_name(*req), "wt", encoding="utf-8") as f:
            f.write(xml)
    return xml


def load_or_fetch_osm(center_lat, center_lon, radius_m, cache_dir,
                      url=OVERPASS_URL, headers=None, timeout=180):
    """Centroid+radius convenience wrapper over :func:`load_or_fetch_osm_bbox`."""
    south, west, north, east = bbox_from_center(center_lat, center_lon, radius_m)
    return load_or_fetch_osm_bbox(south, west, north, east, cache_dir,
                                  url=url, headers=headers, timeout=timeout)


def load_or_fetch_osm_for_trace(latitudes, longitudes, margin_m, cache_dir,
                                url=OVERPASS_URL, headers=None, timeout=180):
    """Fetch raw OSM covering a whole trajectory (bbox of points + margin)."""
    bbox = bbox_from_points(latitudes, longitudes, margin_m)
    return load_or_fetch_osm_bbox(*bbox, cache_dir, url=url, headers=headers, timeout=timeout)
