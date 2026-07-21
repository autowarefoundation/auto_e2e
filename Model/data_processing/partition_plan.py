"""Deterministic partitioning of a dataset's groups for the map_task fan-out (#121 §3.3).

The episode-range fan-out (Design/pipeline_parallelization_design.md §3.2, option B)
splits a dataset's GROUPS (L2D episodes / NVIDIA clips) into partitions; each
partition is ingested → labeled → packed by its own pod. This module owns the
split itself, kept as a PURE function (no Flyte, no network) so it is unit-testable
and so the plan is reproducible: the same ``(group_ids, target_cost/partition_size)``
always yields the SAME partitions, which is what makes Flyte cache hits stable
across re-runs (§3.4a).

Two split modes:
  * COUNT (smoke / default): a fixed ``partition_size`` groups per partition.
  * COST (production): accumulate consecutive groups until an estimated cost
    threshold is hit, so uneven group sizes don't create lopsided pods. L2D has
    ~190 frames/episode but very unevenly, and a fixed 10-ep unit over ~100k
    episodes would spawn ~10k partitions × 3 stages — crushing the Flyte control
    plane. Cost mode bounds per-partition work instead of per-partition count.

A ``max_partitions`` guard makes a runaway fan-out (e.g. episodes=0 = all, at a
tiny target_cost) fail LOUDLY rather than silently launching thousands of pods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Sequence, TypeVar

# Group id is an episode index (int, L2D) or a clip uuid (str, NVIDIA). Kept
# generic so one planner serves both datasets.
G = TypeVar("G", int, str)

# A fan-out larger than this is almost certainly a mistake (a mis-set target_cost
# or an un-guarded episodes=0). Callers may raise it explicitly via
# allow_large_fanout, but never silently. Chosen well above any real smoke/scale
# run (10ep→full L2D at a sane cost target is hundreds, not thousands).
DEFAULT_MAX_PARTITIONS = 512


@dataclass(frozen=True)
class PartitionSpec(Generic[G]):
    """One unit of the fan-out: the groups a single pod-chain owns.

    ``index`` is the stable 0-based ordinal (for labeling map_task items / logs);
    ``group_ids`` are the GLOBAL ids (episode indices / clip uuids), so the
    per-sample ``sample_uid`` a pod emits is independent of the partition
    boundary (§3.1). ``est_cost`` is the accumulated estimate (0.0 in COUNT mode).
    """

    index: int
    group_ids: tuple[G, ...]
    est_cost: float = 0.0

    def __len__(self) -> int:
        return len(self.group_ids)


@dataclass(frozen=True)
class PartitionPlan(Generic[G]):
    """The full plan + a human-readable summary for ``log()`` (§3.3)."""

    partitions: tuple[PartitionSpec[G], ...]
    mode: str  # "count" | "cost"
    total_groups: int
    total_est_cost: float = 0.0
    _summary: str = field(default="", repr=False)

    def __len__(self) -> int:
        return len(self.partitions)

    def summary(self) -> str:
        sizes = [len(p) for p in self.partitions]
        base = (f"plan: {len(self.partitions)} partitions ({self.mode} mode) over "
                f"{self.total_groups} groups; sizes min={min(sizes)} "
                f"max={max(sizes)} total={sum(sizes)}")
        if self.mode == "cost":
            base += f"; est_cost total={self.total_est_cost:.1f}"
        return base


def plan_partitions(
    group_ids: Sequence[G],
    *,
    partition_size: int = 10,
    group_costs: Sequence[float] | None = None,
    target_cost: float | None = None,
    max_partitions: int = DEFAULT_MAX_PARTITIONS,
    allow_large_fanout: bool = False,
) -> PartitionPlan[G]:
    """Split ``group_ids`` into deterministic, gap-free, non-overlapping partitions.

    COUNT mode (default, or whenever ``group_costs``/``target_cost`` are absent):
    ``partition_size`` consecutive groups per partition — the smoke-test unit.

    COST mode (both ``group_costs`` and ``target_cost`` given): accumulate
    consecutive groups until the running cost would exceed ``target_cost``, then
    close the partition. A single group heavier than ``target_cost`` still forms
    its own partition (never dropped, never split — a group is the atomic unit).

    Invariants (pinned by tests):
      * every group appears in exactly ONE partition, in input order (coverage +
        no overlap), so the union is the input and nothing is duplicated/lost;
      * deterministic: same inputs → identical partitions (cache stability, §3.4a);
      * ``len(partitions) <= max_partitions`` unless ``allow_large_fanout`` — else
        ValueError (no silent thousand-pod fan-out).

    Raises:
        ValueError: empty input, non-positive partition_size, non-positive
            target_cost, mismatched group_costs length, or the guard trips.
    """
    n = len(group_ids)
    if n == 0:
        raise ValueError("plan_partitions: no groups to partition (empty group_ids)")

    cost_mode = group_costs is not None and target_cost is not None
    if cost_mode:
        assert group_costs is not None and target_cost is not None  # for type-checkers
        if len(group_costs) != n:
            raise ValueError(
                f"group_costs length {len(group_costs)} != group_ids length {n}")
        if target_cost <= 0:
            raise ValueError(f"target_cost must be > 0, got {target_cost}")
        partitions = _partition_by_cost(group_ids, group_costs, target_cost)
        mode = "cost"
    else:
        if partition_size <= 0:
            raise ValueError(f"partition_size must be > 0, got {partition_size}")
        partitions = _partition_by_count(group_ids, partition_size)
        mode = "count"

    if len(partitions) > max_partitions and not allow_large_fanout:
        raise ValueError(
            f"plan_partitions would create {len(partitions)} partitions "
            f"(> max_partitions={max_partitions}). This usually means an "
            f"un-guarded episodes=0 (all) or a too-small target_cost. Raise "
            f"target_cost / partition_size, or pass allow_large_fanout=True to "
            f"deliberately fan out this wide.")

    total_cost = sum(p.est_cost for p in partitions)
    return PartitionPlan(
        partitions=tuple(partitions), mode=mode, total_groups=n,
        total_est_cost=total_cost)


def _partition_by_count(group_ids: Sequence[G], size: int) -> list[PartitionSpec[G]]:
    """Fixed ``size`` consecutive groups per partition (COUNT mode)."""
    out: list[PartitionSpec[G]] = []
    for i in range(0, len(group_ids), size):
        out.append(PartitionSpec(index=len(out),
                                 group_ids=tuple(group_ids[i:i + size])))
    return out


def _partition_by_cost(
    group_ids: Sequence[G], group_costs: Sequence[float], target: float,
) -> list[PartitionSpec[G]]:
    """Accumulate consecutive groups until running cost >= target (COST mode).

    A group is atomic: if adding it would exceed ``target`` but the current
    partition is non-empty, the group starts a NEW partition; a single group
    heavier than ``target`` forms its own (over-target) partition rather than
    being split or dropped.
    """
    out: list[PartitionSpec[G]] = []
    cur_ids: list[G] = []
    cur_cost = 0.0
    for gid, cost in zip(group_ids, group_costs):
        if cur_ids and cur_cost + cost > target:
            out.append(PartitionSpec(index=len(out),
                                     group_ids=tuple(cur_ids), est_cost=cur_cost))
            cur_ids, cur_cost = [], 0.0
        cur_ids.append(gid)
        cur_cost += cost
    if cur_ids:
        out.append(PartitionSpec(index=len(out),
                                 group_ids=tuple(cur_ids), est_cost=cur_cost))
    return out
