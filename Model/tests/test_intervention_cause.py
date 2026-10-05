"""Tests for counterfactual cause selection from intervention deltas (#122)."""

from __future__ import annotations

import dataclasses

from data_processing.reasoning_label_generation.intervention_cause import (
    InterventionCauseCandidate,
    aggregate_candidate_deltas,
    horizon_labels_from_intervention_deltas,
    select_causes_by_intervention,
)
from data_processing.reasoning_label_generation.schema import HORIZON_SECONDS
from data_processing.reasoning_label_generation.targets import _SOURCE_WEIGHT
from evaluation.cause_label_unit_protocol import (
    Factor,
    geometry_teacher_causes,
    in_conflict,
    oracle_delta,
    run_cause_label_unit_protocol,
    _straight_path,
)


def test_select_filters_and_ranks():
    cands = [
        InterventionCauseCandidate("pedestrian", 0.05),
        InterventionCauseCandidate("red_light", 0.40),
        InterventionCauseCandidate("parked_car", 0.001),  # below floor
    ]
    got = select_causes_by_intervention(cands, min_delta=0.01, relative_to_max=0.5)
    assert got == ["red_light"]  # pedestrian is < 0.5 * 0.40


def test_top_k():
    cands = aggregate_candidate_deltas({"a": 0.9, "b": 0.8, "c": 0.7})
    assert select_causes_by_intervention(cands, min_delta=0.0, top_k=2, relative_to_max=None) == [
        "a",
        "b",
    ]


def test_horizon_labels_provenance():
    per_h = {
        0.0: aggregate_candidate_deltas({"pedestrian": 0.5, "noise": 0.0}, horizon_sec=0.0),
        1.0: [],
    }
    labels = horizon_labels_from_intervention_deltas(per_h, min_delta=0.01, relative_to_max=None)
    assert len(labels) == len(HORIZON_SECONDS)
    assert labels[0].cause == ["pedestrian"]
    assert labels[0].provenance == "counterfactual_weak"
    assert labels[0].confidence > 0
    assert labels[1].cause == []
    assert labels[1].confidence == 0.0


def test_gt_provenance_is_opt_in():
    per_h = {0.0: aggregate_candidate_deltas({"pedestrian_crossing": 0.5})}
    labels = horizon_labels_from_intervention_deltas(
        per_h, provenance="counterfactual_gt", relative_to_max=None,
    )
    assert labels[0].provenance == "counterfactual_gt"


def test_counterfactual_weak_source_weight():
    assert _SOURCE_WEIGHT["counterfactual_weak"] == 0.3


def test_factor_has_no_typed_deltas():
    """Old SCENES table typed in_conflict/oracle_delta/blind_delta on one line."""
    names = {f.name for f in dataclasses.fields(Factor)}
    assert names == {"name", "x", "y", "radius_m"}


def test_oracle_delta_is_computed_from_geometry():
    """Moving a factor onto the path must raise the proxy; the old table could not."""
    path = _straight_path()
    far = Factor("pedestrian_crossing", 12.0, 8.0, 0.4)
    near = Factor("pedestrian_crossing", 12.0, 0.2, 0.4)
    assert not in_conflict(far, path)
    assert in_conflict(near, path)
    assert oracle_delta(near, path) > oracle_delta(far, path)


def test_geometry_teacher_is_not_taxonomy_hash():
    """MockTeacher hashed scene id → slippery_road for ped_in_crosswalk."""
    from evaluation.cause_label_unit_protocol import SCENES

    ped = next(s for s in SCENES if s.scene_id == "ped_in_crosswalk")
    got = geometry_teacher_causes(ped)
    assert "pedestrian_crossing" in got
    assert "slippery_road" not in got


def test_protocol_is_synthetic_not_issue_122_anchor():
    report = run_cause_label_unit_protocol()
    assert report["claim"] == "synthetic_unit_protocol"
    assert "kit_lidar_object_gt" in report["not_claimed"]
    assert "issue_122_external_truth_validation" in report["not_claimed"]
    assert report["provenance_emitted"] == "counterfactual_weak"
    assert report["oracle_vs_geometry"]["f1"] > report["blind_vs_geometry"]["f1"]


def test_protocol_can_fail_on_weaker_true_conflict():
    """relative_to_max drops the far-in-corridor lead; typed tables never FN'd."""
    report = run_cause_label_unit_protocol()
    scene = next(
        s for s in report["oracle_scenes"] if s["scene_id"] == "slippery_plus_lead"
    )
    assert "slippery_road" in scene["truth"]
    assert "slow_lead_vehicle" in scene["truth"]
    assert "slow_lead_vehicle" not in scene["pred"]
    assert scene["fn"] >= 1.0
    assert report["oracle_vs_geometry"]["f1"] < 1.0


def test_unit_protocol_does_not_import_mock_teacher():
    import ast
    from pathlib import Path

    import evaluation.cause_label_unit_protocol as protocol

    tree = ast.parse(Path(protocol.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("mock_teacher" in name for name in imported)
