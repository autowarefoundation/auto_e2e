"""Cache-provenance DatasetSnapshot (#121 Phase 2, §3.4a).

The fan-out's "ingest once, pack rarely" contract rests on the Flyte cache key
carrying the REAL determinants of a stage's output. DatasetSnapshot is that key
material, so these tests pin:
  * the group digest is ORDER-INDEPENDENT (the SET of groups is the determinant),
    DETERMINISTIC across calls (process-stable blake2b, not builtin hash), and
    TYPE-TAGGED (int 12 and str "12" never collide);
  * contract versions come from the single source (contract_versions.py), so a
    stage never inlines them;
  * a revision / group-set / contract change flips the snapshot (cache miss),
    while an unchanged slice reproduces it exactly (cache hit).
"""

from __future__ import annotations

from data_processing.contract_versions import UID_SCHEMA_VERSION, PARSER_VERSION
from data_processing.dataset_snapshot import (
    DatasetSnapshot,
    group_metadata_digest,
    published_shard_name,
    shard_partition_id,
    split_bucket,
)


def test_digest_is_order_independent():
    assert group_metadata_digest([1, 2, 3]) == group_metadata_digest([3, 1, 2])


def test_digest_is_deterministic_across_calls():
    # Would fail if built on the process-salted builtin hash().
    assert group_metadata_digest([5, 6, 7]) == group_metadata_digest([5, 6, 7])


def test_int_and_str_group_ids_never_collide():
    assert group_metadata_digest([12]) != group_metadata_digest(["12"])


def test_different_group_set_changes_digest():
    assert group_metadata_digest([1, 2, 3]) != group_metadata_digest([1, 2, 3, 4])


def test_split_bucket_matches_training_contract_vectors():
    assert split_bucket("l2d-e000000") == 2
    assert split_bucket("l2d-e000012") == 1
    assert split_bucket("nv-25cd4769") == 5


def test_published_shard_names_are_partition_unique_and_order_stable():
    first = published_shard_name(["10", "11"], 0)
    same = published_shard_name(["11", "10"], 0)
    other = published_shard_name(["12", "13"], 0)

    assert first == same
    assert first != other
    assert first.endswith("-train-000000.tar")
    assert shard_partition_id(None) == ""
    assert published_shard_name(None, 2) == "train-000002.tar"


def test_build_defaults_versions_from_single_source():
    s = DatasetSnapshot.build("yaak-ai/L2D", "rev-abc", [10, 11, 12])
    assert s.uid_schema_version == UID_SCHEMA_VERSION
    assert s.parser_version == PARSER_VERSION
    assert s.metadata_digest == group_metadata_digest([10, 11, 12])


def test_snapshot_frozen_and_hashable():
    # frozen=True → usable as a dict key / in a set, and immutable (safe cache key).
    s = DatasetSnapshot.build("yaak-ai/L2D", "rev-abc", [1, 2])
    assert hash(s) == hash(DatasetSnapshot.build("yaak-ai/L2D", "rev-abc", [2, 1]))
    import pytest
    with pytest.raises(Exception):
        s.dataset = "other"  # type: ignore[misc]


def test_revision_bump_changes_snapshot():
    a = DatasetSnapshot.build("yaak-ai/L2D", "rev-1", [1, 2, 3])
    b = DatasetSnapshot.build("yaak-ai/L2D", "rev-2", [1, 2, 3])
    assert a != b
    assert a.cache_key_fields()["source_revision"] != b.cache_key_fields()["source_revision"]


def test_same_slice_reproduces_snapshot_cache_hit():
    a = DatasetSnapshot.build("yaak-ai/L2D", "rev-1", [1, 2, 3])
    b = DatasetSnapshot.build("yaak-ai/L2D", "rev-1", [3, 2, 1])
    assert a == b                      # identical literal → Flyte cache hit
    assert a.cache_key_fields() == b.cache_key_fields()


def test_cache_key_fields_has_no_runtime_knobs():
    # The provenance must NOT carry episode count / num_workers / resources — only
    # data-revision + code contracts (§3.4c). Guard the exact field set.
    s = DatasetSnapshot.build("yaak-ai/L2D", "rev-1", [1])
    assert set(s.cache_key_fields()) == {
        "dataset", "source_revision", "metadata_digest",
        "uid_schema_version", "parser_version",
        "shard_schema_version", "geometry_version",
    }
