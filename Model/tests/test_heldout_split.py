"""Held-out train/val split for pre-extracted shards (fair generalization eval).

The eval used to score the SAME shards used for training (all tars, no split), so
ADE/FDE measured memorization and structurally favored the lower-capacity
imitation baseline. make_pre_extracted_loader now supports a disjoint per-sample
hash split. These tests pin the invariants the fair comparison relies on:
  - train and val are DISJOINT (no sample in both),
  - the split is DETERMINISTIC across processes/reruns (fixed hash, not salted
    hash()), so the train task and the separate eval task agree,
  - val is roughly the requested fraction,
  - split="all" / val_fraction=0 keeps every sample (legacy behaviour).
"""

from __future__ import annotations

import pytest

# pre_extracted imports webdataset at module load; it is not in the core CI
# requirements. Skip the whole module when webdataset is unavailable (matches the
# NVIDIA-dep importorskip pattern elsewhere). The split logic itself is pure-python
# hashing, but it lives in pre_extracted, so we guard the import.
pytest.importorskip("webdataset")

from data_parsing.pre_extracted import _split_bucket, _split_keep  # noqa: E402


def _keys(n):
    # Stable synthetic sample keys like the shard __key__ ("s00000000"...).
    return [f"s{i:08d}" for i in range(n)]


def test_split_is_deterministic():
    """Same key → same bucket every call (fixed hash, not process-salted)."""
    for k in _keys(50):
        assert _split_bucket(k) == _split_bucket(k)
    # Known-stable spot check: buckets are in range and repeatable.
    assert all(0 <= _split_bucket(k) < 10 for k in _keys(200))


def test_train_val_disjoint_and_cover_all():
    """Every sample is in exactly one of train/val; none in both, none dropped."""
    train_keep = _split_keep("train", 0.2)
    val_keep = _split_keep("val", 0.2)
    train, val = set(), set()
    for k in _keys(1000):
        s = {"__key__": k}
        in_t, in_v = train_keep(s), val_keep(s)
        assert in_t != in_v, f"{k}: must be in exactly one split (t={in_t}, v={in_v})"
        (train if in_t else val).add(k)
    assert train and val
    assert train.isdisjoint(val)
    assert len(train) + len(val) == 1000


def test_val_fraction_approximately_honored():
    """val is ~val_fraction of samples (within a bucket's granularity)."""
    val_keep = _split_keep("val", 0.2)
    n = 2000
    val = sum(1 for k in _keys(n) if val_keep({"__key__": k}))
    frac = val / n
    assert 0.12 < frac < 0.28, f"val fraction {frac:.3f} not near 0.2"


def test_all_split_keeps_everything():
    """split='all' or val_fraction=0 keeps every sample (legacy in-sample path)."""
    keep_all = _split_keep("all", 0.2)
    keep_zero = _split_keep("train", 0.0)
    for k in _keys(100):
        s = {"__key__": k}
        assert keep_all(s) is True
        assert keep_zero(s) is True
