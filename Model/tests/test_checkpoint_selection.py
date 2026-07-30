"""Tests for scene-balanced composite checkpoint selection."""

from __future__ import annotations

import math

import pytest

from evaluation.checkpoint_selection import (
    SELECTOR_MIN_DELTA,
    SELECTOR_POLICY_VERSION,
    UTILITY_SCALES,
    aggregate_validation_records,
    build_selector_calibration_report,
    freeze_component_availability,
    score_checkpoint,
    score_is_better,
    validate_frozen_availability,
)


def _record(
    sample_uid: str,
    group_uid: str,
    value: float,
    **overrides,
):
    return {
        "sample_uid": sample_uid,
        "split_group_uid": group_uid,
        "ade_3s_m": value,
        "fde_3s_m": value * 2.0,
        "comfort_excess": value / 100.0,
        "offroad_excess": value / 200.0,
        "route_gap": None,
        "wrong_branch_excess": None,
        "destination_error_m": None,
        "diagnostic_predicted_offroad_rate": value / 100.0,
        "diagnostic_target_offroad_rate": 0.0,
        "diagnostic_predicted_route_compliance": 0.9,
        "diagnostic_target_route_compliance": 1.0,
        "diagnostic_raster_tolerance_m": 0.5,
        **overrides,
    }


def test_selector_policy_uses_three_second_displacement_contract():
    assert SELECTOR_POLICY_VERSION == "rollout_composite_selector_v3"
    assert UTILITY_SCALES["ade_3s_m"] == 2.5
    assert UTILITY_SCALES["fde_3s_m"] == 3.0
    assert "fde_6_4s_m" not in UTILITY_SCALES


def test_scene_balanced_aggregate_is_not_sample_weighted():
    records = [
        _record("a-1", "scene-a", 1.0),
        _record("b-1", "scene-b", 3.0),
        _record("b-2", "scene-b", 5.0),
    ]

    aggregate = aggregate_validation_records(records)
    ade = aggregate["metrics"]["ade_3s_m"]

    assert ade["natural"] == pytest.approx(3.0)
    assert ade["scene_balanced"] == pytest.approx(2.5)
    assert ade["eligible_sample_count"] == 3
    assert ade["eligible_scene_count"] == 2
    assert ade["scene_distribution"]["p50"] == pytest.approx(2.5)


def test_duplicate_exposure_does_not_change_scene_balanced_mean():
    base = [
        _record("a-1", "scene-a", 1.0),
        _record("b-1", "scene-b", 3.0),
        _record("b-2", "scene-b", 5.0),
    ]
    repeated = base + [
        _record("b-3", "scene-b", 3.0),
        _record("b-4", "scene-b", 5.0),
    ]

    first = aggregate_validation_records(base)
    second = aggregate_validation_records(repeated)

    assert (
        first["metrics"]["ade_3s_m"]["scene_balanced"]
        == second["metrics"]["ade_3s_m"]["scene_balanced"]
    )
    assert (
        first["metrics"]["ade_3s_m"]["natural"]
        != second["metrics"]["ade_3s_m"]["natural"]
    )


def test_aggregation_rejects_missing_scene_and_duplicate_sample():
    missing_scene = _record("sample", "scene", 1.0)
    missing_scene["split_group_uid"] = ""
    with pytest.raises(ValueError, match="split_group_uid"):
        aggregate_validation_records([missing_scene])

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_validation_records([
            _record("sample", "scene-a", 1.0),
            _record("sample", "scene-b", 2.0),
        ])


def test_availability_uses_frozen_coverage_thresholds():
    records = [
        _record(
            f"sample-{index}",
            f"scene-{index % 4}",
            1.0,
            route_gap=0.1,
            wrong_branch_excess=(0.0 if index < 20 else None),
            destination_error_m=(2.0 if index < 10 else None),
        )
        for index in range(50)
    ]

    availability = freeze_component_availability(
        aggregate_validation_records(records)
    )

    assert availability["map_safety"]
    assert availability["navigation"]
    assert availability["wrong_branch"]
    assert availability["destination"]
    assert availability["calibration"] == {
        "target_offroad_rate": 0.0,
        "target_route_compliance": 1.0,
        "raster_tolerance_m": 0.5,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "route_gap": 0.1,
                "diagnostic_target_route_compliance": 0.0,
            },
            "target compliance is saturated",
        ),
        (
            {
                "diagnostic_target_offroad_rate": 1.0,
            },
            "target off-road rate is saturated",
        ),
    ],
)
def test_availability_rejects_saturated_map_diagnostics(
    overrides,
    message,
):
    records = [
        _record(
            f"sample-{index}",
            f"scene-{index % 4}",
            1.0,
            **overrides,
        )
        for index in range(50)
    ]

    with pytest.raises(ValueError, match=message):
        freeze_component_availability(
            aggregate_validation_records(records)
        )


def test_score_renormalizes_unavailable_map_and_navigation():
    records = [
        _record(
            f"sample-{index}",
            f"scene-{index % 2}",
            1.0,
            offroad_excess=None,
        )
        for index in range(4)
    ]
    aggregate = aggregate_validation_records(records)
    availability = freeze_component_availability(aggregate)

    result = score_checkpoint(aggregate, availability)

    assert set(result["components"]) == {"trajectory", "comfort"}
    assert sum(result["effective_weights"].values()) == pytest.approx(1.0)
    assert result["utility_scales"] == UTILITY_SCALES
    assert result["effective_weights"]["trajectory"] == pytest.approx(
        0.50 / 0.65
    )
    assert math.isfinite(result["score"])


def test_lower_errors_produce_better_composite_score():
    worse_records = [
        _record(
            f"sample-{index}",
            f"scene-{index % 4}",
            2.0,
            route_gap=0.1,
            wrong_branch_excess=(0.1 if index < 20 else None),
            destination_error_m=2.0,
        )
        for index in range(50)
    ]
    better_records = [
        {
            **record,
            "ade_3s_m": float(record["ade_3s_m"]) * 0.5,
            "fde_3s_m": float(record["fde_3s_m"]) * 0.5,
            "comfort_excess": float(record["comfort_excess"]) * 0.5,
            "offroad_excess": float(record["offroad_excess"]) * 0.5,
            "route_gap": float(record["route_gap"]) * 0.5,
            "wrong_branch_excess": (
                None
                if record["wrong_branch_excess"] is None
                else float(record["wrong_branch_excess"]) * 0.5
            ),
            "destination_error_m": (
                float(record["destination_error_m"]) * 0.5
            ),
        }
        for record in worse_records
    ]
    worse_aggregate = aggregate_validation_records(worse_records)
    availability = freeze_component_availability(worse_aggregate)
    better_aggregate = aggregate_validation_records(better_records)

    worse = score_checkpoint(worse_aggregate, availability)
    better = score_checkpoint(better_aggregate, availability)

    assert better["score"] > worse["score"]
    assert score_is_better(
        better["score"],
        worse["score"],
    )
    assert not score_is_better(
        worse["score"] + SELECTOR_MIN_DELTA,
        worse["score"],
    )


def test_large_excess_remains_rankable_without_hard_clipping():
    worse_aggregate = aggregate_validation_records([
        _record(
            f"worse-{index}",
            f"scene-{index % 2}",
            1.0,
            comfort_excess=0.6,
            offroad_excess=0.8,
        )
        for index in range(4)
    ])
    better_aggregate = aggregate_validation_records([
        _record(
            f"better-{index}",
            f"scene-{index % 2}",
            1.0,
            comfort_excess=0.3,
            offroad_excess=0.4,
        )
        for index in range(4)
    ])
    availability = freeze_component_availability(worse_aggregate)

    worse = score_checkpoint(worse_aggregate, availability)
    better = score_checkpoint(better_aggregate, availability)

    assert 0.0 < worse["components"]["comfort"] < 1.0
    assert 0.0 < worse["components"]["map_safety"] < 1.0
    assert better["components"]["comfort"] > worse["components"]["comfort"]
    assert (
        better["components"]["map_safety"]
        > worse["components"]["map_safety"]
    )
    assert better["score"] > worse["score"]


def test_calibration_reports_saturation_and_weight_rank_sensitivity():
    selections = [
        {
            "policy_version": SELECTOR_POLICY_VERSION,
            "components": {
                "trajectory": trajectory,
                "comfort": 1.0,
                "map_safety": map_safety,
                "navigation": navigation,
            },
        }
        for trajectory, map_safety, navigation in (
            (0.3, 0.8, 0.5),
            (0.5, 0.6, 0.7),
            (0.8, 0.4, 0.6),
        )
    ]

    report = build_selector_calibration_report(selections)

    assert report["checkpoint_count"] == 3
    assert report["rank_evidence_sufficient"]
    assert report["almost_always_saturated_components"] == ["comfort"]
    assert len(report["weight_sensitivity"]) == 8
    assert all(
        scenario["spearman_rank_correlation"] is not None
        for scenario in report["weight_sensitivity"]
    )


def test_calibration_rejects_component_availability_drift():
    with pytest.raises(ValueError, match="availability changed"):
        build_selector_calibration_report([
            {
                "policy_version": SELECTOR_POLICY_VERSION,
                "components": {"trajectory": 0.5, "comfort": 0.5},
            },
            {
                "policy_version": SELECTOR_POLICY_VERSION,
                "components": {"trajectory": 0.6},
            },
        ])


def test_frozen_availability_tolerates_only_calibration_float_noise():
    expected = {
        "trajectory": True,
        "coverage": {"ade_3s_m": 10},
        "calibration": {"raster_tolerance_m": 0.5},
    }
    observed = {
        **expected,
        "calibration": {"raster_tolerance_m": 0.5 + 1e-10},
    }

    validate_frozen_availability(expected, observed)

    with pytest.raises(ValueError, match="discrete availability"):
        validate_frozen_availability(
            expected,
            {
                **observed,
                "coverage": {"ade_3s_m": 9},
            },
        )
