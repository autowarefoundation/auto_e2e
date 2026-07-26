"""Route-conditioned evaluation metric tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.navigation_metrics import (
    NAVIGATION_COMPARISON_VERSION,
    NAVIGATION_EVALUATION_VERSION,
    ROUTE_QUALITY_FIELDS,
    compare_navigation_records,
    navigation_sample_metrics,
    route_swap_sample_metrics,
    summarize_navigation_metrics,
)
from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
    RouteChannel,
)


def _raster_for_path(path):
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    route = np.zeros(
        (ROUTE_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.uint8,
    )
    pixels = np.rint(geometry.ego_to_pixel(path)).astype(int)
    for row, col in pixels:
        route[
            RouteChannel.SELECTED_CORRIDOR,
            max(0, row - 1):row + 2,
            max(0, col - 1):col + 2,
        ] = 1
    row, col = pixels[-1]
    route[
        RouteChannel.DESTINATION,
        max(0, row - 1):row + 2,
        max(0, col - 1):col + 2,
    ] = 1
    maps = np.zeros(
        (MAP_CHANNEL_COUNT, geometry.height_px, geometry.width_px),
        dtype=np.float32,
    )
    return route, maps


def _comparison_records(
    count,
    *,
    ade,
    fde,
    wrong_branch,
):
    return [
        {
            "sample_uid": f"sample-{index:03d}",
            "ade_m": ade,
            "fde_m": fde,
            "route_valid": True,
            "junction": True,
            "maneuver": "left",
            "wrong_branch": wrong_branch,
        }
        for index in range(count)
    ]


def test_navigation_metrics_detect_wrong_branch_and_destination_error():
    target = np.column_stack(
        [np.arange(1.0, 65.0), np.zeros(64)]
    )
    predicted = target.copy()
    predicted[32:, 1] = np.linspace(0.0, 20.0, 32)
    route, maps = _raster_for_path(target)

    record = navigation_sample_metrics(
        predicted,
        target,
        route,
        maps,
        route_valid=True,
        metadata={
            "route_intersection": True,
            "route_maneuver": "left",
            "route_confidence": 0.9,
            "route_quality_matched_pose_ratio": 0.95,
        },
    )

    assert record["maneuver"] == "left"
    assert record["junction"] is True
    assert 0.0 < record["route_point_compliance"] < 1.0
    assert record["target_route_point_compliance"] == 1.0
    assert record["route_outside_distance_m"] > 0.0
    assert record["wrong_branch"] == 1.0
    assert record["destination_distance_error_m"] > 0.0
    assert record["route_confidence"] == 0.9
    assert record["route_quality_matched_pose_ratio"] == 0.95


def test_invalid_route_keeps_displacement_and_masks_route_metrics():
    trajectory = np.zeros((64, 2), dtype=np.float64)
    route, maps = _raster_for_path(trajectory)

    record = navigation_sample_metrics(
        trajectory,
        trajectory,
        route,
        maps,
        route_valid=False,
        metadata={"route_maneuver": "unknown"},
    )

    assert record["ade_m"] == 0.0
    assert record["fde_m"] == 0.0
    assert math.isnan(record["route_point_compliance"])
    assert math.isnan(record["wrong_branch"])


def test_navigation_summary_has_required_slices_and_is_deterministic():
    records = [
        {
            "ade_m": 1.0,
            "fde_m": 2.0,
            "route_valid": True,
            "junction": True,
            "maneuver": "left",
            "route_point_compliance": 0.8,
            "route_outside_distance_m": 0.5,
            "wrong_branch": 0.0,
            "destination_distance_error_m": 1.5,
            **{field: 0.9 for field in ROUTE_QUALITY_FIELDS},
        },
        {
            "ade_m": 3.0,
            "fde_m": 4.0,
            "route_valid": False,
            "junction": False,
            "maneuver": "straight",
            "route_point_compliance": math.nan,
            "route_outside_distance_m": math.nan,
            "wrong_branch": math.nan,
            "destination_distance_error_m": math.nan,
            **{field: 0.2 for field in ROUTE_QUALITY_FIELDS},
        },
    ]

    first = summarize_navigation_metrics(
        records,
        bootstrap_seed=7,
        bootstrap_resamples=50,
    )
    second = summarize_navigation_metrics(
        records,
        bootstrap_seed=7,
        bootstrap_resamples=50,
    )

    assert first == second
    assert first["schema_version"] == NAVIGATION_EVALUATION_VERSION
    assert first["slices"]["overall"]["sample_count"] == 2
    assert first["slices"]["route_valid"]["sample_count"] == 1
    assert first["slices"]["route_invalid"]["sample_count"] == 1
    assert first["slices"]["junction"]["sample_count"] == 1
    assert first["slices"]["maneuver_left"]["sample_count"] == 1
    assert first["slices"]["maneuver_right"]["sample_count"] == 0
    assert first["slices"]["maneuver_straight"]["sample_count"] == 1
    quality = first["slices"]["overall"]["route_quality"]
    assert quality["route_confidence"]["p50"] == 0.55
    delta = first["route_valid_vs_invalid_delta"]
    assert delta["definition"] == "route_valid_minus_route_invalid"
    assert delta["ade_m"]["mean"] == -2.0
    assert delta["fde_m"]["mean"] == -2.0


def test_route_swap_metrics_measure_reactive_path_change():
    selected = np.column_stack(
        [np.arange(1.0, 65.0), np.zeros(64)]
    )
    swapped = selected.copy()
    swapped[:, 1] = 10.0
    route, _ = _raster_for_path(selected)
    swapped_route, _ = _raster_for_path(swapped)

    metrics = route_swap_sample_metrics(
        selected,
        swapped,
        route,
        swapped_route_mask=swapped_route,
        selected_maneuver="straight",
        swapped_maneuver="left",
    )

    assert metrics["mean_path_delta_m"] == 10.0
    assert metrics["endpoint_delta_m"] == 10.0
    assert metrics["selected_route_compliance"] == 1.0
    assert metrics["swapped_input_compliance"] == 0.0
    assert metrics["swapped_route_compliance"] == 1.0
    assert metrics["selected_compliance_drop"] == 1.0
    assert metrics["lateral_endpoint_delta_m"] == 10.0
    assert metrics["maneuver_direction_consistent"] == 1.0

    report = summarize_navigation_metrics(
        [{
            "ade_m": 0.0,
            "fde_m": 0.0,
            "route_valid": True,
            "junction": True,
            "maneuver": "straight",
            "route_point_compliance": 1.0,
            "route_outside_distance_m": 0.0,
            "wrong_branch": 0.0,
            "destination_distance_error_m": 0.0,
        }],
        route_swap_records=[metrics],
        bootstrap_resamples=10,
    )
    counterfactual = report["route_swap_counterfactual"]
    assert counterfactual["different_maneuver_sample_count"] == 1
    assert (
        counterfactual["maneuver_direction_consistent"]["mean"]
        == 1.0
    )
    assert (
        counterfactual["maneuver_pairs"]["straight_to_left"][
            "direction_consistency"
        ]["mean"]
        == 1.0
    )


def test_paired_comparison_supports_route_gain_within_guardrails():
    baseline = _comparison_records(
        40,
        ade=1.0,
        fde=2.0,
        wrong_branch=1.0,
    )
    conditioned = _comparison_records(
        40,
        ade=1.01,
        fde=2.02,
        wrong_branch=0.0,
    )

    first = compare_navigation_records(
        conditioned,
        baseline,
        bootstrap_seed=11,
        bootstrap_resamples=50,
    )
    second = compare_navigation_records(
        conditioned,
        baseline,
        bootstrap_seed=11,
        bootstrap_resamples=50,
    )

    assert first == second
    assert first["schema_version"] == NAVIGATION_COMPARISON_VERSION
    assert first["decision"]["verdict"] == "supported"
    assert first["primary_metric"]["count"] == 40
    assert first["primary_metric"]["difference_ci95"] == [-1.0, -1.0]
    assert (
        first["aggregate_guardrails"]["ade_m"][
            "relative_regression"
        ]
        == pytest.approx(0.01)
    )


def test_paired_comparison_rejects_aggregate_regression():
    baseline = _comparison_records(
        40,
        ade=1.0,
        fde=2.0,
        wrong_branch=1.0,
    )
    conditioned = _comparison_records(
        40,
        ade=1.03,
        fde=2.0,
        wrong_branch=0.0,
    )

    report = compare_navigation_records(
        conditioned,
        baseline,
        bootstrap_resamples=20,
    )

    assert report["decision"]["verdict"] == "not_supported"
    assert report["decision"]["primary_interval_below_zero"] is True
    assert report["decision"]["aggregate_within_tolerance"] is False


def test_paired_comparison_requires_thirty_primary_samples():
    baseline = _comparison_records(
        29,
        ade=1.0,
        fde=2.0,
        wrong_branch=1.0,
    )
    conditioned = _comparison_records(
        29,
        ade=1.0,
        fde=2.0,
        wrong_branch=0.0,
    )

    report = compare_navigation_records(
        conditioned,
        baseline,
        bootstrap_resamples=20,
    )

    assert report["decision"]["verdict"] == "inconclusive"
    assert report["decision"]["has_enough_primary_evidence"] is False


def test_paired_comparison_rejects_sample_identity_drift():
    baseline = _comparison_records(
        30,
        ade=1.0,
        fde=2.0,
        wrong_branch=1.0,
    )
    conditioned = _comparison_records(
        30,
        ade=1.0,
        fde=2.0,
        wrong_branch=0.0,
    )
    conditioned[-1]["sample_uid"] = "different-sample"

    with pytest.raises(ValueError, match="sample identities differ"):
        compare_navigation_records(
            conditioned,
            baseline,
            bootstrap_resamples=20,
        )
