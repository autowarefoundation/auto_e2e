"""Tests for the whole-record JSONL JOIN interchange (#98/#117).

The teacher-as-cached-stage refactor writes each label as one full record per
line (write_records_jsonl) so data_processing can JOIN labels into shards by
sample_id (load_records_by_sample_id). No torch/GPU/network.
"""

from __future__ import annotations

import os
import tempfile

import pytest

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


def test_load_rejects_duplicate_ids_and_skips_blank_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "records.jsonl")
        # Two lines for one immutable JOIN key are ambiguous and must fail.
        write_records_jsonl([_record("s00000000"), _record("s00000000")], path)
        with open(path, "a") as f:
            f.write("\n")
        with pytest.raises(ValueError, match="duplicate reasoning sample_id"):
            load_records_by_sample_id(path)

        # Blank lines remain harmless when the records are otherwise unique.
        write_records_jsonl([_record("s00000000")], path)
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


def test_roundtrip_preserves_v2_optional_fields_and_abstain():
    """Audit G-D: whole-record JSON roundtrip keeps v2 optional fields + abstain."""
    from data_processing.reasoning_label_generation.schema import (
        ReasoningHorizonLabel, ReasoningLabelRecord, SCHEMA_VERSION)
    from data_processing.reasoning_label_generation.targets import (
        record_to_json, record_from_json)

    h = ReasoningHorizonLabel(
        horizon_sec=2.0, relation_to_ego="same_lane_ahead",
        hazard_event=["cut_in_risk"], cause=["lead_vehicle"],
        longitudinal_response="slow_down", confidence=0.8, provenance="teacher_gt",
        global_scene_context=["urban"], time_to_conflict=1.5,
        time_to_collision=3.0, time_to_stop_line=None)
    rec = ReasoningLabelRecord(
        schema_version=SCHEMA_VERSION, sample_id="s00000042", timestamp=1.0,
        dataset_name="l2d", teacher_provider="openai_compatible",
        teacher_model="cosmos", prompt_version="v3", request_mode="clip_horizons",
        horizons=[h])
    back = record_from_json(record_to_json(rec))
    assert back.horizons[0].global_scene_context == ["urban"]
    assert back.horizons[0].time_to_conflict == 1.5
    assert back.horizons[0].time_to_stop_line is None
    assert back.horizons[0].hazard_event == ["cut_in_risk"]

    ab = ReasoningLabelRecord.abstain(
        sample_id="s00000043", dataset_name="l2d",
        teacher_provider="openai_compatible", teacher_model="cosmos",
        prompt_version="v3", request_mode="clip_horizons",
        teacher_error="unparseable")
    ab_back = record_from_json(record_to_json(ab))
    assert ab_back.abstained is True and ab_back.teacher_error == "unparseable"
    assert ab_back.horizons == []
