"""Route-conditioned open-loop metrics for KITScenes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from navigation.geometry import (
    DEFAULT_NAVIGATION_GEOMETRY,
    MapChannel,
    NavigationRasterGeometry,
    RouteChannel,
)


NAVIGATION_EVALUATION_VERSION = "navigation_evaluation_v1"
NAVIGATION_COMPARISON_VERSION = "navigation_comparison_v1"
BOOTSTRAP_SEED = 149
BOOTSTRAP_RESAMPLES = 1_000
PRIMARY_MINIMUM_SAMPLE_COUNT = 30
MAXIMUM_AGGREGATE_REGRESSION = 0.02
ROUTE_QUALITY_FIELDS = (
    "route_confidence",
    "route_quality_matched_pose_ratio",
    "route_quality_median_lateral_distance_m",
    "route_quality_p95_lateral_distance_m",
    "route_quality_median_heading_error_rad",
    "route_quality_p95_heading_error_rad",
    "route_quality_shortest_path_fill_count",
    "route_quality_shortest_path_fill_length_m",
    "route_quality_adjacent_transition_count",
    "route_quality_unresolved_discontinuities",
)


def _mask_values_at_positions(
    mask: np.ndarray,
    positions_xy_m: np.ndarray,
    geometry: NavigationRasterGeometry,
) -> np.ndarray:
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [T,2]")
    pixels = geometry.ego_to_pixel(positions)
    rows = np.rint(pixels[:, 0]).astype(np.int64)
    cols = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < geometry.height_px)
        & (cols >= 0)
        & (cols < geometry.width_px)
    )
    values = np.zeros(len(positions), dtype=bool)
    values[inside] = np.asarray(mask, dtype=bool)[
        rows[inside],
        cols[inside],
    ]
    return values


def _destination_xy(
    destination_mask: np.ndarray,
    geometry: NavigationRasterGeometry,
) -> np.ndarray | None:
    pixels = np.argwhere(np.asarray(destination_mask, dtype=bool))
    if len(pixels) == 0:
        return None
    return geometry.pixel_to_ego(pixels.astype(np.float64)).mean(axis=0)


def _distance_to_mask_m(
    mask: np.ndarray,
    positions_xy_m: np.ndarray,
    geometry: NavigationRasterGeometry,
) -> np.ndarray:
    """Return distance from each point to the nearest occupied raster cell."""
    occupied_pixels = np.argwhere(np.asarray(mask, dtype=bool))
    if len(occupied_pixels) == 0:
        return np.full(len(positions_xy_m), math.nan, dtype=np.float64)
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    occupied_xy = geometry.pixel_to_ego(
        occupied_pixels.astype(np.float64)
    )
    distances = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), 64):
        chunk = positions[start:start + 64]
        differences = chunk[:, None, :] - occupied_xy[None, :, :]
        distances[start:start + len(chunk)] = np.sqrt(
            np.einsum("nki,nki->nk", differences, differences)
        ).min(axis=1)
    distances[
        _mask_values_at_positions(mask, positions, geometry)
    ] = 0.0
    return distances


def navigation_sample_metrics(
    predicted_xy_m: np.ndarray,
    target_xy_m: np.ndarray,
    route_mask: np.ndarray,
    map_context: np.ndarray,
    *,
    route_valid: bool,
    metadata: Mapping[str, Any],
    geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
) -> dict[str, Any]:
    """Compute per-sample displacement and selected-route metrics."""
    predicted = np.asarray(predicted_xy_m, dtype=np.float64)
    target = np.asarray(target_xy_m, dtype=np.float64)
    routes = np.asarray(route_mask)
    maps = np.asarray(map_context)
    if predicted.shape != target.shape or (
        predicted.ndim != 2 or predicted.shape[1] != 2
    ):
        raise ValueError(
            "predicted and target trajectories must share shape [T,2]"
        )
    if routes.shape != (
        2,
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("route mask shape differs from navigation geometry")
    if maps.ndim != 3 or maps.shape[1:] != routes.shape[1:]:
        raise ValueError("map context shape differs from route mask")

    errors = np.linalg.norm(predicted - target, axis=1)
    intersection = bool(metadata.get("route_intersection", False))
    if "route_intersection" not in metadata:
        intersection = bool(
            np.any(
                (maps[MapChannel.INTERSECTION] > 0.0)
                & (routes[RouteChannel.SELECTED_CORRIDOR] > 0)
            )
        )
    record: dict[str, Any] = {
        "ade_m": float(errors.mean()),
        "fde_m": float(errors[-1]),
        "route_valid": bool(route_valid),
        "junction": intersection,
        "maneuver": str(metadata.get("route_maneuver", "unknown")),
        "route_point_compliance": math.nan,
        "target_route_point_compliance": math.nan,
        "route_outside_distance_m": math.nan,
        "wrong_branch": math.nan,
        "destination_distance_error_m": math.nan,
    }
    for field in ROUTE_QUALITY_FIELDS:
        record[field] = float(metadata.get(field, math.nan))
    if not route_valid:
        return record

    corridor = routes[RouteChannel.SELECTED_CORRIDOR] > 0
    predicted_on_route = _mask_values_at_positions(
        corridor,
        predicted,
        geometry,
    )
    target_on_route = _mask_values_at_positions(
        corridor,
        target,
        geometry,
    )
    record["route_point_compliance"] = float(
        predicted_on_route.mean()
    )
    record["target_route_point_compliance"] = float(
        target_on_route.mean()
    )
    record["route_outside_distance_m"] = float(
        _distance_to_mask_m(corridor, predicted, geometry).mean()
    )
    if intersection and bool(target_on_route[-1]):
        record["wrong_branch"] = float(not bool(predicted_on_route[-1]))

    destination = _destination_xy(
        routes[RouteChannel.DESTINATION],
        geometry,
    )
    if destination is not None:
        predicted_distance = float(
            np.linalg.norm(predicted[-1] - destination)
        )
        target_distance = float(np.linalg.norm(target[-1] - destination))
        record["destination_distance_error_m"] = abs(
            predicted_distance - target_distance
        )
    return record


def route_swap_sample_metrics(
    predicted_xy_m: np.ndarray,
    swapped_route_xy_m: np.ndarray,
    selected_route_mask: np.ndarray,
    *,
    swapped_route_mask: np.ndarray | None = None,
    selected_maneuver: str = "unknown",
    swapped_maneuver: str = "unknown",
    geometry: NavigationRasterGeometry = DEFAULT_NAVIGATION_GEOMETRY,
) -> dict[str, Any]:
    """Measure planner response when another scene's route raster is supplied."""
    predicted = np.asarray(predicted_xy_m, dtype=np.float64)
    swapped = np.asarray(swapped_route_xy_m, dtype=np.float64)
    if predicted.shape != swapped.shape or (
        predicted.ndim != 2 or predicted.shape[1] != 2
    ):
        raise ValueError("counterfactual trajectories must share shape [T,2]")
    route_mask = np.asarray(selected_route_mask)
    if route_mask.shape != (
        2,
        geometry.height_px,
        geometry.width_px,
    ):
        raise ValueError("selected route mask shape differs from geometry")
    corridor = route_mask[RouteChannel.SELECTED_CORRIDOR] > 0
    selected_compliance = _mask_values_at_positions(
        corridor,
        predicted,
        geometry,
    ).mean()
    swapped_compliance = _mask_values_at_positions(
        corridor,
        swapped,
        geometry,
    ).mean()
    swapped_route_compliance = math.nan
    if swapped_route_mask is not None:
        swapped_mask = np.asarray(swapped_route_mask)
        if swapped_mask.shape != route_mask.shape:
            raise ValueError("swapped route mask shape differs from geometry")
        swapped_route_compliance = float(
            _mask_values_at_positions(
                swapped_mask[RouteChannel.SELECTED_CORRIDOR] > 0,
                swapped,
                geometry,
            ).mean()
        )
    deltas = np.linalg.norm(predicted - swapped, axis=1)
    maneuver_lateral_sign = {
        "left": 1,
        "straight": 0,
        "right": -1,
    }
    selected_sign = maneuver_lateral_sign.get(selected_maneuver)
    swapped_sign = maneuver_lateral_sign.get(swapped_maneuver)
    expected_direction = (
        None
        if selected_sign is None or swapped_sign is None
        else int(np.sign(swapped_sign - selected_sign))
    )
    lateral_endpoint_delta = float(swapped[-1, 1] - predicted[-1, 1])
    direction_consistent = math.nan
    if expected_direction is not None and expected_direction != 0:
        direction_consistent = float(
            lateral_endpoint_delta * expected_direction > 0.0
        )
    return {
        "mean_path_delta_m": float(deltas.mean()),
        "endpoint_delta_m": float(deltas[-1]),
        "lateral_endpoint_delta_m": lateral_endpoint_delta,
        "selected_route_compliance": float(selected_compliance),
        "swapped_input_compliance": float(swapped_compliance),
        "swapped_route_compliance": swapped_route_compliance,
        "selected_compliance_drop": float(
            selected_compliance - swapped_compliance
        ),
        "selected_maneuver": selected_maneuver,
        "swapped_maneuver": swapped_maneuver,
        "maneuver_direction_consistent": direction_consistent,
    }


def _mean_with_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if len(finite) == 0:
        return {"count": 0, "mean": None, "ci95": None}
    mean = float(finite.mean())
    if len(finite) == 1:
        interval = [mean, mean]
    else:
        indices = rng.integers(
            0,
            len(finite),
            size=(resamples, len(finite)),
        )
        bootstrap_means = finite[indices].mean(axis=1)
        interval = [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ]
    return {
        "count": int(len(finite)),
        "mean": mean,
        "ci95": interval,
    }


def _distribution_with_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    summary = _mean_with_ci(
        finite.tolist(),
        rng=rng,
        resamples=resamples,
    )
    if len(finite) == 0:
        return {
            **summary,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        **summary,
        "min": float(finite.min()),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def _difference_with_ci(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    left = np.asarray(
        [value for value in left_values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    right = np.asarray(
        [value for value in right_values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "mean": None,
        "ci95": None,
    }
    if len(left) == 0 or len(right) == 0:
        return result
    result["mean"] = float(left.mean() - right.mean())
    left_indices = rng.integers(
        0,
        len(left),
        size=(resamples, len(left)),
    )
    right_indices = rng.integers(
        0,
        len(right),
        size=(resamples, len(right)),
    )
    differences = (
        left[left_indices].mean(axis=1)
        - right[right_indices].mean(axis=1)
    )
    result["ci95"] = [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    ]
    return result


def _paired_metric_with_ci(
    conditioned_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    if len(conditioned_values) != len(baseline_values):
        raise ValueError("paired metric inputs must have equal length")
    finite_pairs = [
        (float(conditioned), float(baseline))
        for conditioned, baseline in zip(
            conditioned_values,
            baseline_values,
            strict=True,
        )
        if conditioned is not None
        and baseline is not None
        and math.isfinite(float(conditioned))
        and math.isfinite(float(baseline))
    ]
    if not finite_pairs:
        return {
            "count": 0,
            "conditioned_mean": None,
            "baseline_mean": None,
            "difference_mean": None,
            "difference_ci95": None,
            "relative_regression": None,
            "relative_regression_ci95": None,
        }
    values = np.asarray(finite_pairs, dtype=np.float64)
    conditioned = values[:, 0]
    baseline = values[:, 1]
    differences = conditioned - baseline
    conditioned_mean = float(conditioned.mean())
    baseline_mean = float(baseline.mean())
    if len(values) == 1:
        bootstrap_differences = differences
        bootstrap_relative = (
            differences / baseline
            if baseline[0] != 0.0
            else np.asarray([], dtype=np.float64)
        )
    else:
        indices = rng.integers(
            0,
            len(values),
            size=(resamples, len(values)),
        )
        bootstrap_conditioned = conditioned[indices].mean(axis=1)
        bootstrap_baseline = baseline[indices].mean(axis=1)
        bootstrap_differences = (
            bootstrap_conditioned - bootstrap_baseline
        )
        nonzero = bootstrap_baseline != 0.0
        bootstrap_relative = (
            bootstrap_differences[nonzero]
            / bootstrap_baseline[nonzero]
        )
    relative_regression = (
        None
        if baseline_mean == 0.0
        else float((conditioned_mean - baseline_mean) / baseline_mean)
    )
    relative_interval = (
        None
        if len(bootstrap_relative) == 0
        else [
            float(np.quantile(bootstrap_relative, 0.025)),
            float(np.quantile(bootstrap_relative, 0.975)),
        ]
    )
    return {
        "count": int(len(values)),
        "conditioned_mean": conditioned_mean,
        "baseline_mean": baseline_mean,
        "difference_mean": float(differences.mean()),
        "difference_ci95": [
            float(np.quantile(bootstrap_differences, 0.025)),
            float(np.quantile(bootstrap_differences, 0.975)),
        ],
        "relative_regression": relative_regression,
        "relative_regression_ci95": relative_interval,
    }


def _records_by_sample_uid(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_uid = str(record.get("sample_uid", ""))
        if not sample_uid:
            raise ValueError(f"{label} record has no sample_uid")
        if sample_uid in indexed:
            raise ValueError(
                f"{label} records contain duplicate sample_uid {sample_uid}"
            )
        indexed[sample_uid] = record
    if not indexed:
        raise ValueError(f"{label} records are empty")
    return indexed


def compare_navigation_records(
    conditioned_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    primary_minimum_sample_count: int = PRIMARY_MINIMUM_SAMPLE_COUNT,
    maximum_aggregate_regression: float = MAXIMUM_AGGREGATE_REGRESSION,
) -> dict[str, Any]:
    """Evaluate the frozen paired route-conditioning research gate."""
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if primary_minimum_sample_count <= 0:
        raise ValueError("primary_minimum_sample_count must be positive")
    if maximum_aggregate_regression < 0.0:
        raise ValueError("maximum_aggregate_regression must be non-negative")

    conditioned = _records_by_sample_uid(
        conditioned_records,
        label="conditioned",
    )
    baseline = _records_by_sample_uid(
        baseline_records,
        label="baseline",
    )
    if conditioned.keys() != baseline.keys():
        missing_conditioned = sorted(baseline.keys() - conditioned.keys())
        missing_baseline = sorted(conditioned.keys() - baseline.keys())
        raise ValueError(
            "comparison sample identities differ: "
            f"missing_conditioned={missing_conditioned[:3]} "
            f"missing_baseline={missing_baseline[:3]}"
        )

    sample_uids = sorted(conditioned)
    for sample_uid in sample_uids:
        conditioned_record = conditioned[sample_uid]
        baseline_record = baseline[sample_uid]
        for key in ("route_valid", "junction", "maneuver"):
            if conditioned_record.get(key) != baseline_record.get(key):
                raise ValueError(
                    f"comparison route metadata differs for {sample_uid}: "
                    f"{key}"
                )
        conditioned_wrong_branch = conditioned_record.get("wrong_branch")
        baseline_wrong_branch = baseline_record.get("wrong_branch")
        conditioned_eligible = (
            conditioned_wrong_branch is not None
            and math.isfinite(float(conditioned_wrong_branch))
        )
        baseline_eligible = (
            baseline_wrong_branch is not None
            and math.isfinite(float(baseline_wrong_branch))
        )
        if conditioned_eligible != baseline_eligible:
            raise ValueError(
                "wrong-branch eligibility differs for "
                f"{sample_uid}"
            )

    rng = np.random.default_rng(bootstrap_seed)
    overall = {
        metric: _paired_metric_with_ci(
            [conditioned[uid][metric] for uid in sample_uids],
            [baseline[uid][metric] for uid in sample_uids],
            rng=rng,
            resamples=bootstrap_resamples,
        )
        for metric in ("ade_m", "fde_m")
    }
    primary_uids = [
        uid
        for uid in sample_uids
        if conditioned[uid].get("wrong_branch") is not None
        and math.isfinite(float(conditioned[uid]["wrong_branch"]))
    ]
    primary = _paired_metric_with_ci(
        [conditioned[uid]["wrong_branch"] for uid in primary_uids],
        [baseline[uid]["wrong_branch"] for uid in primary_uids],
        rng=rng,
        resamples=bootstrap_resamples,
    )
    has_enough_evidence = (
        primary["count"] >= primary_minimum_sample_count
    )
    primary_improved = (
        has_enough_evidence
        and primary["difference_ci95"] is not None
        and primary["difference_ci95"][1] < 0.0
    )
    aggregate_within_tolerance = all(
        overall[metric]["relative_regression"] is not None
        and overall[metric]["relative_regression"]
        <= maximum_aggregate_regression
        for metric in ("ade_m", "fde_m")
    )
    if not has_enough_evidence:
        verdict = "inconclusive"
    elif primary_improved and aggregate_within_tolerance:
        verdict = "supported"
    else:
        verdict = "not_supported"

    return {
        "schema_version": NAVIGATION_COMPARISON_VERSION,
        "bootstrap": {
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "method": "paired_sample_bootstrap",
        },
        "sample_count": len(sample_uids),
        "sample_uid_digest": hashlib.sha256(
            "\n".join(sample_uids).encode("ascii")
        ).hexdigest(),
        "primary_metric": {
            "name": "route_valid_junction_wrong_branch_rate",
            "lower_is_better": True,
            "minimum_sample_count": primary_minimum_sample_count,
            "eligible_sample_uid_digest": hashlib.sha256(
                "\n".join(primary_uids).encode("ascii")
            ).hexdigest(),
            **primary,
        },
        "aggregate_guardrails": {
            "maximum_relative_regression": (
                maximum_aggregate_regression
            ),
            **overall,
        },
        "decision": {
            "verdict": verdict,
            "has_enough_primary_evidence": has_enough_evidence,
            "primary_interval_below_zero": primary_improved,
            "aggregate_within_tolerance": aggregate_within_tolerance,
        },
    }


def _slice_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    return {
        "sample_count": len(records),
        "ade_m": _mean_with_ci(
            [float(record["ade_m"]) for record in records],
            rng=rng,
            resamples=resamples,
        ),
        "fde_m": _mean_with_ci(
            [float(record["fde_m"]) for record in records],
            rng=rng,
            resamples=resamples,
        ),
        "route_point_compliance": _mean_with_ci(
            [
                float(record["route_point_compliance"])
                for record in records
            ],
            rng=rng,
            resamples=resamples,
        ),
        "route_outside_distance_m": _mean_with_ci(
            [
                float(record.get("route_outside_distance_m", math.nan))
                for record in records
            ],
            rng=rng,
            resamples=resamples,
        ),
        "wrong_branch_rate": _mean_with_ci(
            [float(record["wrong_branch"]) for record in records],
            rng=rng,
            resamples=resamples,
        ),
        "destination_distance_error_m": _mean_with_ci(
            [
                float(record["destination_distance_error_m"])
                for record in records
            ],
            rng=rng,
            resamples=resamples,
        ),
        "route_quality": {
            field: _distribution_with_ci(
                [
                    float(record.get(field, math.nan))
                    for record in records
                ],
                rng=rng,
                resamples=resamples,
            )
            for field in ROUTE_QUALITY_FIELDS
        },
    }


def summarize_navigation_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    route_swap_records: Sequence[Mapping[str, float]] = (),
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Aggregate required route slices with deterministic bootstrap intervals."""
    if not records:
        raise ValueError("navigation evaluation requires sample records")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    normalized = [dict(record) for record in records]
    rng = np.random.default_rng(bootstrap_seed)
    slices = {
        "overall": normalized,
        "route_valid": [
            record for record in normalized if record["route_valid"]
        ],
        "route_invalid": [
            record for record in normalized if not record["route_valid"]
        ],
        "junction": [
            record for record in normalized if record["junction"]
        ],
        "non_junction": [
            record for record in normalized if not record["junction"]
        ],
    }
    for maneuver in ("left", "right", "straight"):
        slices[f"maneuver_{maneuver}"] = [
            record
            for record in normalized
            if record["maneuver"] == maneuver
        ]

    valid_records = slices["route_valid"]
    invalid_records = slices["route_invalid"]
    valid_invalid_delta = {
        "definition": "route_valid_minus_route_invalid",
        "ade_m": _difference_with_ci(
            [float(record["ade_m"]) for record in valid_records],
            [float(record["ade_m"]) for record in invalid_records],
            rng=rng,
            resamples=bootstrap_resamples,
        ),
        "fde_m": _difference_with_ci(
            [float(record["fde_m"]) for record in valid_records],
            [float(record["fde_m"]) for record in invalid_records],
            rng=rng,
            resamples=bootstrap_resamples,
        ),
    }

    counterfactual: dict[str, Any] = {
        "sample_count": len(route_swap_records),
        "different_maneuver_sample_count": sum(
            record.get("selected_maneuver")
            != record.get("swapped_maneuver")
            for record in route_swap_records
        ),
    }
    for key in (
        "mean_path_delta_m",
        "endpoint_delta_m",
        "lateral_endpoint_delta_m",
        "selected_route_compliance",
        "swapped_input_compliance",
        "swapped_route_compliance",
        "selected_compliance_drop",
        "maneuver_direction_consistent",
    ):
        counterfactual[key] = _mean_with_ci(
            [float(record[key]) for record in route_swap_records],
            rng=rng,
            resamples=bootstrap_resamples,
        )
    maneuver_pairs: dict[str, list[Mapping[str, Any]]] = {}
    for record in route_swap_records:
        pair = (
            f"{record.get('selected_maneuver', 'unknown')}_to_"
            f"{record.get('swapped_maneuver', 'unknown')}"
        )
        maneuver_pairs.setdefault(pair, []).append(record)
    counterfactual["maneuver_pairs"] = {
        pair: {
            "sample_count": len(pair_records),
            "direction_consistency": _mean_with_ci(
                [
                    float(record["maneuver_direction_consistent"])
                    for record in pair_records
                ],
                rng=rng,
                resamples=bootstrap_resamples,
            ),
            "lateral_endpoint_delta_m": _mean_with_ci(
                [
                    float(record["lateral_endpoint_delta_m"])
                    for record in pair_records
                ],
                rng=rng,
                resamples=bootstrap_resamples,
            ),
        }
        for pair, pair_records in sorted(maneuver_pairs.items())
    }

    return {
        "schema_version": NAVIGATION_EVALUATION_VERSION,
        "bootstrap": {
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
        "slices": {
            name: _slice_summary(
                values,
                rng=rng,
                resamples=bootstrap_resamples,
            )
            for name, values in slices.items()
        },
        "route_valid_vs_invalid_delta": valid_invalid_delta,
        "route_swap_counterfactual": counterfactual,
    }
