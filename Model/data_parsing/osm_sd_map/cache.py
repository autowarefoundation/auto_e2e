"""Offline per-episode tokenised OSM cache.

Mirrors ``map_rendering.cache`` philosophy: do the slow work (Overpass fetch,
OSM parse, tokenisation) once, offline, and persist per-episode shards so the
DataLoader only reads files. One ``.pt`` file per episode holds a
``{frame_index: osm_map_data}`` mapping.

The expensive ``osm_parser.parse`` runs once per episode (shared raw OSM cache);
``build_node_way_lists`` + patch extraction + tokenisation run per frame because
each frame has its own ego pose.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import torch

from .osm_parser import parse
from .osm_tokenize import extract_osm_map_data
from .overpass import load_or_fetch_osm_for_trace

logger = logging.getLogger(__name__)

# Default fetch margin = the rendered-tile branch's render radius
# (map_rendering.gps_to_map.DEFAULT_RADIUS_M). Defaulting to it means the vector
# and raster branches request the same-sized area around a clip, so their shared
# raw-OSM cache collapses to a single download (the vector branch crops to
# pc_range at tokenisation time regardless). For a vector-only setup with no
# rendered tiles, pass `fetch_margin_m=patch_reach_m(pc_range)` to fetch the
# minimal area instead. Keep in sync with map_rendering's DEFAULT_RADIUS_M.
DEFAULT_FETCH_MARGIN_M = 800


def patch_reach_m(pc_range, buffer_m=50.0):
    """Max distance from an ego pose to a corner of its pc_range patch, + buffer.

    The minimal per-side margin the SD-map fetch needs beyond the trajectory so
    every frame's pc_range crop is covered. Use it as ``fetch_margin_m`` for a
    vector-only setup; the default (``DEFAULT_FETCH_MARGIN_M``) is larger so the
    fetch is shared with the rendered-tile branch.
    """
    max_x = max(abs(pc_range[0]), abs(pc_range[3]))
    max_y = max(abs(pc_range[1]), abs(pc_range[4]))
    return (max_x ** 2 + max_y ** 2) ** 0.5 + buffer_m


def build_episode_osm_cache(
    episode_id,
    frames,
    tokenizer,
    pc_range,
    out_dir,
    raw_osm_cache_dir=None,
    fetch_margin_m=None,
    fixed_num=10,
    pts_dim=3,
    remove_not_relevant_keys=True,
    skip_existing=True,
):
    """Build and persist the tokenised OSM cache for one episode.

    Args:
        episode_id: identifier; the shard is ``{out_dir}/{episode_id}.pt``.
        frames: iterable of ``(frame_index, ego_lat, ego_lon, ego_heading)``.
        tokenizer: HuggingFace tokenizer matching the NLP encoder.
        pc_range: ego-metric crop ``(x_min,y_min,z_min,x_max,y_max,z_max)``.
        out_dir: directory for per-episode shards.
        raw_osm_cache_dir: shared raw-OSM XML cache (see ``overpass``); also used
            by the rendered-tile branch (containment reuse).
        fetch_margin_m: per-side margin beyond the trajectory bbox for the
            Overpass fetch. Defaults to ``DEFAULT_FETCH_MARGIN_M`` (the raster
            branch's render radius) so the fetch is shared with the rendered-tile
            branch. Pass ``patch_reach_m(pc_range)`` for a minimal vector-only
            fetch. Either way it's sized to the trajectory (covers long episodes;
            no fixed centroid radius blind spot).

    Returns:
        Path to the written shard, or ``None`` if the OSM fetch failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"{episode_id}.pt"
    if skip_existing and shard_path.exists():
        return shard_path

    frames = list(frames)
    if not frames:
        logger.warning("episode %s has no frames; skipping", episode_id)
        return None

    if fetch_margin_m is None:
        fetch_margin_m = DEFAULT_FETCH_MARGIN_M

    # One fetch per episode, sized to the trajectory bbox + patch margin.
    lats = [f[1] for f in frames]
    lons = [f[2] for f in frames]
    xml = load_or_fetch_osm_for_trace(lats, lons, fetch_margin_m, raw_osm_cache_dir)
    if xml is None:
        logger.warning("episode %s: no OSM available; skipping", episode_id)
        return None

    osm_elements = parse(io.StringIO(xml))

    shard = {}
    for frame_index, ego_lat, ego_lon, ego_heading in frames:
        shard[frame_index] = extract_osm_map_data(
            osm_elements, ego_lat, ego_lon, ego_heading, tokenizer, pc_range,
            fixed_num=fixed_num, pts_dim=pts_dim,
            remove_not_relevant_keys=remove_not_relevant_keys,
        )

    torch.save(shard, shard_path)
    return shard_path


def load_episode_osm_cache(out_dir, episode_id):
    """Load a per-episode shard ``{frame_index: osm_map_data}`` (or ``None``)."""
    shard_path = Path(out_dir) / f"{episode_id}.pt"
    if not shard_path.exists():
        return None
    return torch.load(shard_path, weights_only=False)
