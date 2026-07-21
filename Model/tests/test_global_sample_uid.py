"""Global, partition-independent sample_uid (#121 Phase 1, §3.1).

The whole episode-range fan-out rests on ONE invariant: the same physical frame
gets the same `sample_uid` no matter which episode/clip subset a given pod loaded.
Positional `f"s{si:08d}"` broke this (it renumbers per subset); the new uid is
built from (episode_index, frame_index) / (clip_uuid, sample_idx).

These tests exercise the uid/split logic without decoding video: they drive the
parser's `sample_uid`/`split_group_uid` against a stubbed `_samples` /
`_episode_ranges` that simulates two different episode subsets sharing an episode.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("data_processing.contract_versions")
from data_processing.contract_versions import UID_SCHEMA_VERSION


# --- L2D: uid from (episode_index, frame_index), stable across subsets ----------

class _FakeL2D:
    """Minimal stand-in exposing exactly what sample_uid/split_group_uid read."""
    from data_parsing.l2d.dataset import L2DDataset as _cls
    sample_uid = _cls.sample_uid
    split_group_uid = _cls.split_group_uid

    def __init__(self, samples, ranges):
        self._samples = samples          # list[(ep_idx, row)]
        self._episode_ranges = ranges    # {ep_idx: (start, end)}


def test_l2d_uid_stable_across_episode_subsets():
    # Subset A loaded episodes {5,6}: their local rows start at different offsets
    # than subset B, which loaded {6,7}. Episode 6 is shared.
    # ep6 occupies rows [100,200) in A and [0,100) in B — DIFFERENT positional si,
    # but frame_index = row - ep_start is the same physical frame.
    a = _FakeL2D(samples=[(5, 10), (6, 150)], ranges={5: (0, 100), 6: (100, 200)})
    b = _FakeL2D(samples=[(6, 50), (7, 250)], ranges={6: (0, 100), 7: (100, 300)})
    # ep6 frame at within-episode offset 50: A row=150 (150-100), B row=50 (50-0).
    assert a.sample_uid(1) == b.sample_uid(0), \
        "same physical frame (ep6, frame 50) must get the same uid across subsets"
    assert a.sample_uid(1) == f"l2d-{UID_SCHEMA_VERSION}-e000006-f000050"


def test_l2d_split_group_is_episode_level():
    a = _FakeL2D(samples=[(6, 100), (6, 199)], ranges={6: (0, 200)})
    # Two different frames of the SAME episode share one split group (no
    # train/val straddle within an episode).
    assert a.split_group_uid(0) == a.split_group_uid(1) == "l2d-e000006"


def test_l2d_uid_charset_safe_for_webdataset_key():
    a = _FakeL2D(samples=[(6, 150)], ranges={6: (100, 200)})
    uid = a.sample_uid(0)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", uid), f"uid {uid!r} not __key__-safe"
    assert "." not in uid and "/" not in uid


# --- NVIDIA: uid from (clip_uuid, sample_idx) -----------------------------------
# NVIDIA parser imports pandas/physical_ai_av (not in core/CI env), so build the
# stub lazily inside the tests behind importorskip.

def _fake_nvidia(samples):
    pytest.importorskip("pandas")
    pytest.importorskip("physical_ai_av")
    from data_parsing.nvidia_physical_ai.dataset import NvidiaAVDataset

    class _FakeNvidia:
        sample_uid = NvidiaAVDataset.sample_uid
        split_group_uid = NvidiaAVDataset.split_group_uid

        def __init__(self, s):
            self._samples = s

    return _FakeNvidia(samples)


def test_nvidia_uid_stable_across_clip_subsets():
    uuid = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    # Same (clip, sample_idx) at different LIST positions in two subsets.
    a = _fake_nvidia([("aaaa", 0, 0), (uuid, 7, 999)])
    b = _fake_nvidia([(uuid, 7, 999)])
    assert a.sample_uid(1) == b.sample_uid(0)
    assert a.sample_uid(1) == f"nv-{UID_SCHEMA_VERSION}-{uuid}-f000007"


def test_nvidia_split_group_is_clip_level():
    uuid = "fd1d1b6b-59bf-4292-8295-5028aa6aa5e3"
    a = _fake_nvidia([(uuid, 3, 0), (uuid, 90, 0)])
    assert a.split_group_uid(0) == a.split_group_uid(1) == f"nv-{uuid}"
    uid = a.sample_uid(0)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", uid) and "." not in uid
