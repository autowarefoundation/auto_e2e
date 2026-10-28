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
L2D_OSM_DIRECTORY = "osm"


def file_unit_id(path: str) -> str:
    """One upstream file's unit id: its repository path with dots for slashes."""
    parts = path.split("/")
    if not path or any(part in {"", ".", ".."} for part in parts) or parts[0] == L2D_META_UNIT:
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
    if not meta_files or any(not path.startswith(f"{L2D_META_UNIT}/") for path in meta_files):
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
            unit_files.setdefault(unit, [path])
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
