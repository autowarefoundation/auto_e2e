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

import json


def _sample(episode, frame):
    """A raw shard sample dict with meta.json carrying split_group_uid (episode)."""
    uid = f"l2d-v1-e{episode:06d}-f{frame:06d}"
    grp = f"l2d-e{episode:06d}"
    return {"__key__": uid,
            "meta.json": json.dumps({"sample_uid": uid, "split_group_uid": grp}).encode()}


def test_split_is_deterministic():
    """Same key → same bucket every call (fixed hash, not process-salted)."""
    ks = [f"l2d-e{i:06d}" for i in range(200)]
    for k in ks[:50]:
        assert _split_bucket(k) == _split_bucket(k)
    assert all(0 <= _split_bucket(k) < 10 for k in ks)


def test_frames_of_same_episode_never_straddle_train_val():
    """THE key #121 invariant: all frames of one episode land on ONE side, so
    correlated neighbours don't leak across train/val."""
    train_keep = _split_keep("train", 0.2)
    val_keep = _split_keep("val", 0.2)
    for ep in range(50):
        sides = {("train" if train_keep(_sample(ep, f)) else
                  "val" if val_keep(_sample(ep, f)) else "?")
                 for f in range(20)}  # 20 frames of the same episode
        assert len(sides) == 1, f"episode {ep} frames split across {sides}"


def test_train_val_group_disjoint_and_cover_all():
    """Every sample in exactly one split; train/val episode groups are disjoint."""
    train_keep = _split_keep("train", 0.2)
    val_keep = _split_keep("val", 0.2)
    train_g, val_g, n = set(), set(), 0
    for ep in range(200):
        for f in range(5):
            s = _sample(ep, f)
            n += 1
            in_t, in_v = train_keep(s), val_keep(s)
            assert in_t != in_v
            (train_g if in_t else val_g).add(ep)
    assert train_g and val_g
    assert train_g.isdisjoint(val_g)          # episode never in both
    assert len(train_g) + len(val_g) == 200   # every episode placed


def test_val_fraction_approximately_honored():
    """val is ~val_fraction of EPISODES (split granularity is the group)."""
    val_keep = _split_keep("val", 0.2)
    n = 500
    val = sum(1 for ep in range(n) if val_keep(_sample(ep, 0)))
    frac = val / n
    assert 0.12 < frac < 0.28, f"val fraction {frac:.3f} not near 0.2"


def test_all_split_keeps_everything():
    """split='all' or val_fraction=0 keeps every sample (legacy in-sample path)."""
    keep_all = _split_keep("all", 0.2)
    keep_zero = _split_keep("train", 0.0)
    for ep in range(100):
        s = _sample(ep, 0)
        assert keep_all(s) is True
        assert keep_zero(s) is True


def test_legacy_shard_without_split_group_falls_back_to_key():
    """A shard whose meta.json predates split_group_uid still splits (by __key__),
    so old shards don't crash the loader."""
    train_keep = _split_keep("train", 0.2)
    val_keep = _split_keep("val", 0.2)
    legacy = {"__key__": "s00000001", "meta.json": json.dumps({"idx": 1}).encode()}
    assert train_keep(legacy) != val_keep(legacy)  # placed on exactly one side
