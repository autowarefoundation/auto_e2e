"""How an L2D raw snapshot is cut into units, and how packs read it back.

L2D is a LeRobot v3 dataset: one parquet file and one video file per camera
hold many episodes each, and ``meta/`` describes all of them. A unit is
therefore one upstream file (plus one unit for all of ``meta/`` and one for
the pinned OpenStreetMap extract), and the snapshot provenance records which
units each episode needs. A pack partition is a group of episodes and reads
the union of their units.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

L2D_META_UNIT = "meta"
L2D_OSM_UNIT = "osm"


def _safe_parts(path: str) -> list[str]:
    parts = path.split("/")
    if not path or any(part in {"", ".", ".."} or "\\" in part for part in parts):
        raise ValueError(f"unsafe L2D repository path: {path!r}")
    return parts


def file_unit_id(path: str) -> str:
    """One upstream file's unit id: its repository path with dots for slashes."""
    parts = _safe_parts(path)
    if parts[0] not in {"data", "videos"}:
        raise ValueError(f"not an L2D data or video file path: {path!r}")
    return ".".join(parts)


def plan_units(
    episodes: Iterable[int],
    *,
    meta_files: Sequence[str],
    data_file_of: Callable[[int], str],
    video_files_of: Callable[[int], Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(unit_files, episode_units)`` for the requested episodes.

    Every episode needs the meta unit, its parquet file, and one video file
    per camera; files shared by several episodes become one unit.
    """
    if not meta_files or any(_safe_parts(path)[0] != L2D_META_UNIT for path in meta_files):
        raise ValueError("L2D meta files must all live under meta/")
    unit_files: dict[str, list[str]] = {L2D_META_UNIT: sorted(meta_files)}
    episode_units: dict[str, list[str]] = {}
    ordered = sorted({int(episode) for episode in episodes})
    if not ordered:
        raise ValueError("an L2D snapshot needs at least one episode")
    for episode in ordered:
        files = [data_file_of(episode), *video_files_of(episode)]
        units = [L2D_META_UNIT]
        for path in files:
            unit = file_unit_id(path)
            # Dots for slashes is not injective ("a/b.c" and "a.b/c"); two
            # paths on one id would silently read the wrong file.
            if unit_files.setdefault(unit, [path]) != [path]:
                raise ValueError(f"L2D files {unit_files[unit][0]!r} and {path!r} share unit id {unit!r}")
            units.append(unit)
        episode_units[str(episode)] = sorted(set(units))
    return unit_files, episode_units


def partition_plan(
    provenance: Mapping[str, Any],
    partition_size: int,
) -> tuple[list[list[str]], list[list[str]]]:
    """``(partition units, partition episodes)`` in episode order."""
    if partition_size < 1:
        raise ValueError("partition_size must be at least 1")
    episode_units = provenance.get("episode_units")
    if not isinstance(episode_units, Mapping) or not episode_units:
        raise ValueError("L2D snapshot provenance lists no episode units")
    episodes = sorted(episode_units, key=int)
    units, groups = [], []
    for start in range(0, len(episodes), partition_size):
        group = episodes[start:start + partition_size]
        groups.append(group)
        units.append(sorted({unit for episode in group for unit in episode_units[episode]}))
    return units, groups


# OSM extracts are fetched only from Geofabrik, by a dated name that never
# changes and with a SHA-256 the caller pins; anything else is refused before
# a request is made.
OSM_EXTRACT_HOSTS = frozenset({"download.geofabrik.de"})
OSM_EXTRACT_MAX_BYTES = 10 * 1024**3


def validate_osm_extract(url: str, sha256: str) -> str:
    """The extract's file name, once the URL and pinned digest are acceptable."""
    import re
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in OSM_EXTRACT_HOSTS:
        raise ValueError(f"OSM extracts come only from https://{'|'.join(sorted(OSM_EXTRACT_HOSTS))}")
    if parts.username or parts.password or parts.port or parts.query or parts.fragment:
        raise ValueError("OSM extract URL must not carry credentials, a port, a query, or a fragment")
    name = parts.path.rsplit("/", 1)[-1]
    if re.fullmatch(r"[a-z0-9-]+-[0-9]{6}\.osm\.pbf", name) is None:
        raise ValueError("OSM extract must be a dated file (<region>-YYMMDD.osm.pbf)")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("OSM extract needs its SHA-256 pinned as 64 lowercase hex digits")
    return name
