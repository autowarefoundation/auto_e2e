"""Validation-split helpers for open-loop evaluation (#66 §4).

L2D ships all episodes in a single "train" partition, so we define our own
val split. The recommended design is a **geographic holdout** (reserve whole
cities) to avoid the geographic/temporal leakage that inflates nuScenes
planning numbers (Lilja et al., CVPR 2024); a simple episode-range split is an
acceptable early-experiment fallback (per the proposal).
"""

from __future__ import annotations

from collections.abc import Sequence


def episode_range_split(num_episodes: int,
                        val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    """Reserve the last ``val_fraction`` of episode indices for validation.

    Simple, leakage-prone (adjacent frames/locations) — for early experiments
    only; prefer :func:`geographic_holdout_split`.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    n_val = max(1, int(round(num_episodes * val_fraction)))
    cut = num_episodes - n_val
    return list(range(cut)), list(range(cut, num_episodes))


def geographic_holdout_split(
    episode_cities: Sequence[str],
    holdout_cities: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Hold out whole cities for validation (recommended, leakage-safe).

    Args:
        episode_cities: city label per episode (index-aligned).
        holdout_cities: cities to reserve for validation.
    Returns:
        ``(train_indices, val_indices)``.
    """
    holdout = set(holdout_cities)
    train: list[int] = []
    val: list[int] = []
    for i, city in enumerate(episode_cities):
        (val if city in holdout else train).append(i)
    return train, val
