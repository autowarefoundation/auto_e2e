"""Tests for the whole-record JSONL JOIN interchange (#98/#117).

The teacher-as-cached-stage refactor writes each label as one full record per
line (write_records_jsonl) so data_processing can JOIN labels into shards by
sample_id (load_records_by_sample_id). No torch/GPU/network.
"""

from __future__ import annotations

import os
import tempfile

from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest
from data_processing.reasoning_label_generation.targets import (
    load_records_by_sample_id,
    write_records_jsonl,
)


def _record(sid):
    return MockTeacher().label(TeacherRequest(sample_id=sid, dataset_name="l2d"))


def test_records_jsonl_roundtrip_by_sample_id():
    records = [_record(f"s{i:08d}") for i in range(3)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "records.jsonl")
        write_records_jsonl(records, path)
        by_id = load_records_by_sample_id(path)

    assert set(by_id) == {"s00000000", "s00000001", "s00000002"}
    # Whole record survives the round trip: horizons + provenance intact.
    r0 = by_id["s00000000"]
    assert len(r0.horizons) == len(records[0].horizons)
    assert r0.horizons[0].cause == records[0].horizons[0].cause
    assert r0.teacher_provider == records[0].teacher_provider
    assert r0.prompt_version == records[0].prompt_version


def test_load_is_last_write_wins_and_skips_blank_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "records.jsonl")
        # Two lines for the same sample_id + a blank line: last wins, blank skipped.
        write_records_jsonl([_record("s00000000"), _record("s00000000")], path)
        with open(path, "a") as f:
            f.write("\n")
        by_id = load_records_by_sample_id(path)

    assert list(by_id) == ["s00000000"]


def test_join_lookup_miss_returns_none():
    # A sample_id with no label (data_processing packs it imitation-only).
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "records.jsonl")
        write_records_jsonl([_record("s00000000")], path)
        by_id = load_records_by_sample_id(path)

    assert by_id.get("s99999999") is None
