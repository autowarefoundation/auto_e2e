"""Tests for the offline teacher pipeline (issue #98, R2/R3/R9).

No GPU / network / real model / Kubernetes. Covers:
    * mock teacher produces valid, deterministic 5-horizon records;
    * OpenAI-compatible teacher with a stub transport parses a clip response;
    * strict=True raises on transport failure / empty / unparseable response;
    * strict=False abstains and marks teacher_error (R9);
    * cached teacher round-trips a JSONL artifact by sample_id;
    * run_labeling writes JSONL and validates locally (no Flyte/K8s);
    * build_teacher resolves providers by name.
"""

from __future__ import annotations

import json

import pytest

from data_processing.reasoning_label_generation.cached_teacher import CachedTeacher
from data_processing.reasoning_label_generation.flyte_tasks import run_labeling
from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.openai_compatible import (
    OpenAICompatibleTeacher,
)
from data_processing.reasoning_label_generation.parquet_writer import write_jsonl
from data_processing.reasoning_label_generation.schema import NUM_HORIZONS
from data_processing.reasoning_label_generation.teacher_client import (
    TeacherRequest,
    build_teacher,
)
from data_processing.reasoning_label_generation.validators import validate_record
from model_components.reasoning.reasoning_taxonomy import DEFAULT_TAXONOMY


def _req(sample_id="scene_1_t0"):
    return TeacherRequest(sample_id=sample_id, dataset_name="l2d")


def test_mock_teacher_valid_and_deterministic():
    t = MockTeacher()
    r1 = t.label(_req())
    r2 = t.label(_req())
    validate_record(r1)
    assert len(r1.horizons) == NUM_HORIZONS
    # Deterministic: same sample_id → identical labels.
    assert r1.horizons[0].cause == r2.horizons[0].cause
    assert r1.horizons[0].relation_to_ego == r2.horizons[0].relation_to_ego


def _valid_clip_json() -> str:
    tax = DEFAULT_TAXONOMY
    horizons = []
    for sec in (0, 1, 2, 3, 4):
        horizons.append({
            "horizon_sec": sec,
            "relation_to_ego": tax.labels("relation_to_ego")[0],
            "hazard_event": [tax.labels("hazard_event")[0]],
            "cause": [tax.labels("cause")[0]],
            "longitudinal_response": tax.labels("longitudinal_response")[0],
            "lateral_response": tax.labels("lateral_response")[0],
            "tactical_response": tax.labels("tactical_response")[0],
            "rule_response": tax.labels("rule_response")[0],
            "confidence": 0.7,
            "evidence": "stub",
        })
    return json.dumps({"horizons": horizons})


def _stub_transport(text):
    def _t(url, payload, headers):
        return {"choices": [{"message": {"content": text}}]}
    return _t


def test_openai_compatible_parses_stub_response():
    t = OpenAICompatibleTeacher(transport=_stub_transport(_valid_clip_json()))
    rec = t.label(_req())
    validate_record(rec)
    assert not rec.abstained
    assert rec.teacher_provider == "openai_compatible"
    assert len(rec.horizons) == NUM_HORIZONS


def test_openai_strict_raises_on_transport_error():
    def _boom(url, payload, headers):
        raise RuntimeError("connection refused")

    t = OpenAICompatibleTeacher(transport=_boom, strict=True)
    with pytest.raises(RuntimeError, match="teacher endpoint call failed"):
        t.label(_req())


def test_openai_strict_raises_on_empty():
    t = OpenAICompatibleTeacher(transport=_stub_transport(""), strict=True)
    with pytest.raises(RuntimeError, match="empty response"):
        t.label(_req())


def test_openai_strict_raises_on_unparseable():
    t = OpenAICompatibleTeacher(transport=_stub_transport("not json at all"), strict=True)
    with pytest.raises(RuntimeError, match="could not be parsed"):
        t.label(_req())


def test_openai_nonstrict_abstains_and_marks_error():
    def _boom(url, payload, headers):
        raise RuntimeError("503")

    t = OpenAICompatibleTeacher(transport=_boom, strict=False)
    rec = t.label(_req())
    assert rec.abstained is True
    assert rec.teacher_error and "503" in rec.teacher_error
    validate_record(rec)  # abstained records validate


def test_cached_teacher_roundtrip(tmp_path):
    src = MockTeacher()
    records = [src.label(_req(f"s_{i}")) for i in range(3)]
    artifact = str(tmp_path / "labels.jsonl")
    write_jsonl(records, artifact)

    cached = CachedTeacher(label_artifact=artifact)
    got = cached.label(_req("s_1"))
    validate_record(got)
    assert got.sample_id == "s_1"
    assert len(got.horizons) == NUM_HORIZONS
    # Matches what the source produced.
    assert got.horizons[0].cause == records[1].horizons[0].cause


def test_cached_teacher_missing_strict_raises(tmp_path):
    artifact = str(tmp_path / "labels.jsonl")
    write_jsonl([MockTeacher().label(_req("s_0"))], artifact)
    cached = CachedTeacher(label_artifact=artifact, strict=True)
    with pytest.raises(KeyError, match="no cached label"):
        cached.label(_req("absent"))


def test_run_labeling_writes_jsonl_local(tmp_path):
    reqs = [_req(f"s_{i}") for i in range(4)]
    out = str(tmp_path / "out.jsonl")
    records = run_labeling(reqs, provider="mock", jsonl_path=out)
    assert len(records) == 4
    lines = [l for l in open(out) if l.strip()]
    assert len(lines) == 4 * NUM_HORIZONS  # one row per (sample, horizon)


def test_build_teacher_resolves_providers():
    assert isinstance(build_teacher("mock"), MockTeacher)
    with pytest.raises(ValueError, match="Unknown teacher provider"):
        build_teacher("nope")
