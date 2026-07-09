"""Tests for the S3-backed per-sample reasoning-label cache (#98/#113).

No network: a stub S3 client backs the store in a dict. Covers get/put round
trip, get_or_compute (compute only on miss), prefix isolation by
(dataset/teacher/prompt_version), and the disabled (no-bucket) path.
"""

from __future__ import annotations

from data_processing.reasoning_label_generation.label_cache import (
    LabelCache,
    cache_prefix,
)
from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest


class _StubS3:
    """In-memory S3 stand-in implementing the get_object/put_object surface."""

    def __init__(self):
        self.store = {}

    def put_object(self, Bucket, Key, Body):
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        import io
        if (Bucket, Key) not in self.store:
            raise KeyError("NoSuchKey")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}


def _record(sid="s0"):
    return MockTeacher().label(TeacherRequest(sample_id=sid, dataset_name="l2d"))


def test_prefix_isolates_by_key_components():
    a = cache_prefix("yaak-ai/L2D", "mock", "v2")
    b = cache_prefix("yaak-ai/L2D", "openai_compatible", "v2")
    c = cache_prefix("yaak-ai/L2D", "mock", "v3")
    assert a != b and a != c and b != c
    assert "dataset=yaak-ai_L2D" in a  # slash sanitized


def test_put_get_roundtrip():
    s3 = _StubS3()
    cache = LabelCache("bkt", "l2d", "mock", "v2", s3_client=s3)
    rec = _record("s5")
    cache.put("s5", rec)
    got = cache.get("s5")
    assert got is not None
    assert got.sample_id == rec.sample_id
    assert got.horizons[0].cause == rec.horizons[0].cause


def test_get_or_compute_only_computes_on_miss():
    s3 = _StubS3()
    cache = LabelCache("bkt", "l2d", "mock", "v2", s3_client=s3)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return _record("s7")

    r1 = cache.get_or_compute("s7", compute)   # miss -> compute + store
    r2 = cache.get_or_compute("s7", compute)   # hit -> no compute
    assert calls["n"] == 1
    assert r1.sample_id == r2.sample_id == "s7"
    assert cache.hits == 1 and cache.misses == 1


def test_disabled_cache_never_touches_s3():
    # No bucket -> always compute, never call the (absent) client.
    cache = LabelCache(None, "l2d", "mock", "v2", s3_client=None)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return _record("s0")

    cache.get_or_compute("s0", compute)
    cache.get_or_compute("s0", compute)
    assert calls["n"] == 2  # computed every time, no caching


def test_different_prefix_is_separate_cache():
    s3 = _StubS3()
    mock_cache = LabelCache("bkt", "l2d", "mock", "v2", s3_client=s3)
    cosmos_cache = LabelCache("bkt", "l2d", "openai_compatible", "v2", s3_client=s3)
    mock_cache.put("s1", _record("s1"))
    # A different teacher's cache does not see the mock entry.
    assert cosmos_cache.get("s1") is None
    assert mock_cache.get("s1") is not None
