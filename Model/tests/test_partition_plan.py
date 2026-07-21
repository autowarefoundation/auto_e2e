"""Deterministic fan-out partitioning (#121 Phase 2, §3.3).

plan_partitions is the linchpin of the episode-range fan-out: it decides which
groups each pod-chain owns. These tests pin the invariants the fan-out relies on,
none of which need Flyte or video:
  * COVERAGE + NO OVERLAP: every group in exactly one partition, in input order;
  * DETERMINISM: same inputs → identical partitions (Flyte cache stability, §3.4a);
  * COST mode bounds per-partition work, treating a group as atomic;
  * the max_partitions guard fails loud (no silent thousand-pod fan-out).
"""

from __future__ import annotations

import pytest

from data_processing.partition_plan import PartitionSpec, plan_partitions


def _all_ids(plan):
    out = []
    for p in plan.partitions:
        out.extend(p.group_ids)
    return out


# --- COUNT mode ----------------------------------------------------------------

def test_count_mode_covers_all_in_order_no_overlap():
    ids = list(range(25))
    plan = plan_partitions(ids, partition_size=10)
    assert plan.mode == "count"
    assert len(plan) == 3                       # 10 + 10 + 5
    assert [len(p) for p in plan.partitions] == [10, 10, 5]
    # union == input, in order, no duplicates, nothing dropped
    assert _all_ids(plan) == ids
    # partition indices are the stable 0-based ordinal
    assert [p.index for p in plan.partitions] == [0, 1, 2]


def test_count_mode_exact_multiple():
    plan = plan_partitions(list(range(20)), partition_size=10)
    assert [len(p) for p in plan.partitions] == [10, 10]


def test_count_mode_size_larger_than_input_is_one_partition():
    plan = plan_partitions([5, 6, 7], partition_size=10)
    assert len(plan) == 1
    assert plan.partitions[0].group_ids == (5, 6, 7)


def test_works_with_string_group_ids_nvidia_clips():
    uuids = [f"clip-{i}" for i in range(7)]
    plan = plan_partitions(uuids, partition_size=3)
    assert [len(p) for p in plan.partitions] == [3, 3, 1]
    assert _all_ids(plan) == uuids


def test_global_ids_preserved_not_renumbered():
    # A partition carries the GLOBAL episode ids (10,11,...), not 0-based — this
    # is what keeps sample_uid partition-independent (§3.1).
    plan = plan_partitions([10, 11, 12, 13], partition_size=2)
    assert plan.partitions[0].group_ids == (10, 11)
    assert plan.partitions[1].group_ids == (12, 13)


# --- determinism ---------------------------------------------------------------

def test_determinism_same_inputs_same_plan():
    ids = list(range(37))
    a = plan_partitions(ids, partition_size=8)
    b = plan_partitions(ids, partition_size=8)
    assert [p.group_ids for p in a.partitions] == [p.group_ids for p in b.partitions]


# --- COST mode -----------------------------------------------------------------

def test_cost_mode_bounds_running_cost():
    ids = list(range(6))
    costs = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]   # target 10 → 2 groups (8) then close
    plan = plan_partitions(ids, group_costs=costs, target_cost=10.0)
    assert plan.mode == "cost"
    assert [len(p) for p in plan.partitions] == [2, 2, 2]
    assert _all_ids(plan) == ids
    # est_cost accumulated per partition
    assert plan.partitions[0].est_cost == pytest.approx(8.0)
    assert plan.total_est_cost == pytest.approx(24.0)


def test_cost_mode_group_heavier_than_target_is_own_partition():
    # A single group over target must NOT be split or dropped — it forms its own
    # (over-target) partition.
    ids = [0, 1, 2]
    costs = [3.0, 100.0, 3.0]
    plan = plan_partitions(ids, group_costs=costs, target_cost=10.0)
    assert [p.group_ids for p in plan.partitions] == [(0,), (1,), (2,)]
    assert _all_ids(plan) == ids


def test_cost_mode_accumulates_uneven_groups():
    ids = [0, 1, 2, 3, 4]
    costs = [2.0, 2.0, 2.0, 9.0, 1.0]        # target 10
    plan = plan_partitions(ids, group_costs=costs, target_cost=10.0)
    # 2+2+2=6, +9 would be 15>10 → close [0,1,2]; [3] then +1=10 not >10 → [3,4]
    assert [p.group_ids for p in plan.partitions] == [(0, 1, 2), (3, 4)]


# --- guards / validation -------------------------------------------------------

def test_max_partitions_guard_trips_loud():
    with pytest.raises(ValueError, match="max_partitions"):
        plan_partitions(list(range(100)), partition_size=1, max_partitions=10)


def test_allow_large_fanout_overrides_guard():
    plan = plan_partitions(list(range(100)), partition_size=1,
                           max_partitions=10, allow_large_fanout=True)
    assert len(plan) == 100


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        plan_partitions([], partition_size=10)


def test_bad_partition_size_raises():
    with pytest.raises(ValueError, match="partition_size"):
        plan_partitions([1, 2, 3], partition_size=0)


def test_cost_mode_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        plan_partitions([1, 2, 3], group_costs=[1.0, 2.0], target_cost=5.0)


def test_cost_mode_bad_target_raises():
    with pytest.raises(ValueError, match="target_cost"):
        plan_partitions([1, 2, 3], group_costs=[1.0, 2.0, 3.0], target_cost=0.0)


def test_spec_len_reports_group_count():
    spec = PartitionSpec(index=0, group_ids=(1, 2, 3))
    assert len(spec) == 3
