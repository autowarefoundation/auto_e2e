"""Validation-split helpers for open-loop evaluation (#66 §4).

L2D ships all episodes in a single "train" partition, so we define our own
val split. The recommended design is a **geographic holdout** (reserve whole
cities) to avoid the geographic/temporal leakage that inflates nuScenes
planning numbers (Lilja et al., CVPR 2024); a simple episode-range split is an
acceptable early-experiment fallback (per the proposal).

Both helpers operate on plain indices / labels supplied by the caller — they do
**not** read L2D metadata themselves. Pulling per-episode city labels from the
dataset (for :func:`geographic_holdout_split`) is a separate metadata step and a
follow-up; these functions just turn that information into train/val indices.
"""

from __future__ import annotations

from collections.abc import Sequence


def episode_range_split(num_episodes: int,
                        val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    """Reserve the last ``val_fraction`` of episode indices for validation.

    Simple, leakage-prone (adjacent frames/locations) — for early experiments
    only; prefer :func:`geographic_holdout_split`. Guarantees a non-empty train
    and val split.

    Raises:
        ValueError: if ``num_episodes < 2`` (can't form two non-empty splits) or
            ``val_fraction`` is not in ``(0, 1)``.
    """
    if num_episodes < 2:
        raise ValueError(
            f"need >= 2 episodes to form non-empty train/val splits, got {num_episodes}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    # at least 1 val, and at least 1 train (cap val at num_episodes - 1)
    n_val = min(max(1, int(round(num_episodes * val_fraction))), num_episodes - 1)
    cut = num_episodes - n_val
    return list(range(cut)), list(range(cut, num_episodes))


def geographic_holdout_split(
    episode_cities: Sequence[str],
    holdout_cities: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Hold out whole cities for validation (recommended, leakage-safe).

    This is a **helper that requires episode-level city labels** supplied by the
    caller (`episode_cities`, index-aligned with the dataset). Extracting those
    labels from L2D metadata is a separate step (follow-up), not done here.

    Args:
        episode_cities: city label per episode (index-aligned).
        holdout_cities: cities to reserve for validation.
    Returns:
        ``(train_indices, val_indices)``.
    Raises:
        ValueError: if the holdout leaves the train or val split empty (e.g. none
            of ``holdout_cities`` appear, or they cover every episode).
    """
    holdout = set(holdout_cities)
    train: list[int] = []
    val: list[int] = []
    for i, city in enumerate(episode_cities):
        (val if city in holdout else train).append(i)
    if not val:
        raise ValueError(
            f"holdout_cities {sorted(holdout)} match no episodes — val split is empty")
    if not train:
        raise ValueError("holdout_cities cover every episode — train split is empty")
    return train, val


def long_tail_split(
    sample_scenarios: Sequence[Sequence[str]],
    long_tail_classes: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Stratify evaluation samples into long-tail vs nominal subsets (#98).

    Average open-loop metrics are dominated by ego status and nominal driving
    (Li et al., CVPR 2024 — "Is Ego Status All You Need?"), so the reasoning
    band must be measured on the long-tail subset, reported as the delta over
    an ego-status-only baseline (e.g. :func:`hold_last_action_baseline`).

    Like the other helpers here, this operates on caller-supplied labels — one
    sequence of active scenario class names per evaluation sample (from teacher
    labels, L2D metadata, or KITScenes annotations); it does not read datasets.

    Args:
        sample_scenarios: per-sample active scenario classes (index-aligned).
        long_tail_classes: class names that define the long tail (e.g. the
            ``edge_case`` taxonomy group, or KITScenes' rare categories).
    Returns:
        ``(long_tail_indices, nominal_indices)`` — a sample lands in the
        long-tail subset when ANY of its classes is in ``long_tail_classes``.
    Raises:
        ValueError: if ``long_tail_classes`` is empty.
    """
    if not long_tail_classes:
        raise ValueError("long_tail_classes must be non-empty")
    rare = set(long_tail_classes)
    long_tail: list[int] = []
    nominal: list[int] = []
    for i, classes in enumerate(sample_scenarios):
        (long_tail if rare.intersection(classes) else nominal).append(i)
    return long_tail, nominal
