"""Caching helpers for the map rendering pipeline.

Network fetches via osmnx are slow (seconds each, internet required). This
module persists fetched graphs to disk and renders/persists tiles for an
entire dataset in one batch so the DataLoader only ever reads PNGs.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx

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
