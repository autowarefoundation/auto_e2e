"""Scene-balanced validation aggregation and composite checkpoint scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np


SELECTOR_POLICY_VERSION = "rollout_composite_selector_v3"
SELECTOR_MIN_DELTA = 0.0005
SELECTOR_CALIBRATION_VERSION = "rollout_selector_calibration_v3"
TOP_LEVEL_WEIGHTS = {
    "trajectory": 0.50,
    "comfort": 0.15,
    "map_safety": 0.15,
    "navigation": 0.20,
}
UTILITY_SCALES = {
    "ade_3s_m": 2.5,
    "fde_3s_m": 3.0,
    "comfort_excess": 0.15,
    "offroad_excess": 0.10,
    "route_gap": 0.15,
    "wrong_branch_excess": 1.0,
    "destination_error_m": 7.5,
}
METRIC_NAMES = (
    "ade_3s_m",
    "fde_3s_m",
    "comfort_excess",
    "offroad_excess",
    "route_gap",
    "wrong_branch_excess",
    "destination_error_m",
)
DIAGNOSTIC_NAMES = (
    "diagnostic_predicted_offroad_rate",
    "diagnostic_target_offroad_rate",
    "diagnostic_predicted_route_compliance",
    "diagnostic_target_route_compliance",
    "diagnostic_raster_tolerance_m",
)
AGGREGATE_NAMES = METRIC_NAMES + DIAGNOSTIC_NAMES
REQUIRED_METRICS = (
    "ade_3s_m",
    "fde_3s_m",
    "comfort_excess",
)


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _aggregate_metrics(
    aggregates: Mapping[str, object],
) -> Mapping[str, Mapping[str, Any]]:
    metrics = aggregates.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("validation aggregates have no metrics")
    return cast(Mapping[str, Mapping[str, Any]], metrics)


def _finite_optional(record: Mapping[str, object], name: str) -> float | None:
    value = record.get(name)
    if value is None:
        return None
    numeric = _as_float(value)
    return numeric if math.isfinite(numeric) else None


def _quantile(values: Sequence[float], quantile: float) -> float:
    return float(
        np.quantile(
            np.asarray(values, dtype=np.float64),
            quantile,
            method="linear",
        )
    )


def aggregate_validation_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate complete sample records naturally and by equal scene weight."""
    if not records:
        raise ValueError("validation records must not be empty")
    sample_uids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for record in records:
        sample_uid = record.get("sample_uid")
        group_uid = record.get("split_group_uid")
        if not isinstance(sample_uid, str) or not sample_uid:
            raise ValueError("validation record has no sample_uid")
        if sample_uid in sample_uids:
            raise ValueError(
                f"duplicate validation sample_uid {sample_uid!r}"
            )
        if not isinstance(group_uid, str) or not group_uid:
            raise ValueError(
                f"validation sample {sample_uid!r} has no split_group_uid"
            )
        values = {
            name: _finite_optional(record, name)
            for name in AGGREGATE_NAMES
        }
        missing_required = [
            name for name in REQUIRED_METRICS
            if values[name] is None
        ]
        if missing_required:
            raise ValueError(
                f"validation sample {sample_uid!r} lacks required metrics "
                f"{missing_required}"
            )
        sample_uids.add(sample_uid)
        normalized.append({
            "sample_uid": sample_uid,
            "split_group_uid": group_uid,
            **values,
        })
    normalized.sort(key=lambda item: str(item["sample_uid"]))

    metrics = {}
    for name in AGGREGATE_NAMES:
        eligible = [
            record for record in normalized
            if record[name] is not None
        ]
        by_scene: dict[str, list[float]] = defaultdict(list)
        for record in eligible:
            by_scene[str(record["split_group_uid"])].append(
                _as_float(record[name])
            )
        scene_means: list[dict[str, object]] = [
            {
                "split_group_uid": group_uid,
                "value": float(np.mean(by_scene[group_uid])),
                "sample_count": len(by_scene[group_uid]),
            }
            for group_uid in sorted(by_scene)
        ]
        if eligible:
            natural = float(np.mean([
                _as_float(record[name]) for record in eligible
            ]))
            scene_values = [
                _as_float(scene["value"]) for scene in scene_means
            ]
            scene_balanced = float(np.mean(scene_values))
            scene_distribution: dict[str, float | int | None] = {
                "count": len(scene_values),
                "mean": scene_balanced,
                "p50": _quantile(scene_values, 0.50),
                "p90": _quantile(scene_values, 0.90),
            }
        else:
            natural = None
            scene_balanced = None
            scene_distribution = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p90": None,
            }
        metrics[name] = {
            "natural": natural,
            "scene_balanced": scene_balanced,
            "eligible_sample_count": len(eligible),
            "eligible_scene_count": len(scene_means),
            "scene_distribution": scene_distribution,
            "scene_means": scene_means,
        }

    return {
        "sample_count": len(normalized),
        "scene_count": len({
            str(record["split_group_uid"]) for record in normalized
        }),
        "metrics": metrics,
    }


def freeze_component_availability(
    aggregates: Mapping[str, object],
    *,
    minimum_route_samples: int = 50,
    minimum_wrong_branch_samples: int = 20,
) -> dict[str, object]:
    """Freeze score components from immutable validation coverage."""
    metrics = _aggregate_metrics(aggregates)
    for name in REQUIRED_METRICS:
        if metrics[name]["eligible_sample_count"] != aggregates["sample_count"]:
            raise ValueError(
                f"required metric {name} has incomplete coverage"
            )
    route_count = int(metrics["route_gap"]["eligible_sample_count"])
    wrong_branch_count = int(
        metrics["wrong_branch_excess"]["eligible_sample_count"]
    )
    destination_count = int(
        metrics["destination_error_m"]["eligible_sample_count"]
    )
    offroad_count = int(
        metrics["offroad_excess"]["eligible_sample_count"]
    )
    map_safety = offroad_count > 0
    navigation = route_count >= minimum_route_samples
    calibration = {}
    if map_safety:
        target_offroad = metrics[
            "diagnostic_target_offroad_rate"
        ]["natural"]
        if target_offroad is None:
            raise ValueError(
                "map selector coverage has no target off-road diagnostic"
            )
        target_offroad = float(target_offroad)
        if target_offroad >= 0.95:
            raise ValueError(
                "map selector target off-road rate is saturated"
            )
        calibration["target_offroad_rate"] = target_offroad
    if navigation:
        target_route_compliance = metrics[
            "diagnostic_target_route_compliance"
        ]["natural"]
        if target_route_compliance is None:
            raise ValueError(
                "navigation selector coverage has no target route diagnostic"
            )
        target_route_compliance = float(target_route_compliance)
        if target_route_compliance <= 0.05:
            raise ValueError(
                "navigation selector target compliance is saturated"
            )
        calibration["target_route_compliance"] = (
            target_route_compliance
        )
    if map_safety or navigation:
        raster_tolerance = metrics[
            "diagnostic_raster_tolerance_m"
        ]["natural"]
        if raster_tolerance is None or float(raster_tolerance) <= 0.0:
            raise ValueError(
                "selector raster tolerance diagnostic is unavailable"
            )
        calibration["raster_tolerance_m"] = float(raster_tolerance)
    return {
        "trajectory": True,
        "comfort": True,
        "map_safety": map_safety,
        "navigation": navigation,
        "wrong_branch": (
            route_count >= minimum_route_samples
            and wrong_branch_count >= minimum_wrong_branch_samples
        ),
        "destination": (
            route_count >= minimum_route_samples
            and destination_count > 0
        ),
        "coverage": {
            name: int(metrics[name]["eligible_sample_count"])
            for name in METRIC_NAMES
        },
        "minimum_route_samples": minimum_route_samples,
        "minimum_wrong_branch_samples": (
            minimum_wrong_branch_samples
        ),
        "calibration": calibration,
    }


def _metric_pair(
    aggregates: Mapping[str, object],
    name: str,
) -> tuple[float, float]:
    metric = _aggregate_metrics(aggregates)[name]
    natural = metric["natural"]
    scene = metric["scene_balanced"]
    if natural is None or scene is None:
        raise ValueError(f"score metric {name} is unavailable")
    natural_value = _as_float(natural)
    scene_value = _as_float(scene)
    if not math.isfinite(natural_value) or not math.isfinite(scene_value):
        raise ValueError(f"score metric {name} is non-finite")
    return natural_value, scene_value


def _combined_metric(
    aggregates: Mapping[str, object],
    name: str,
) -> float:
    natural, scene = _metric_pair(aggregates, name)
    return 0.5 * natural + 0.5 * scene


def _bounded_inverse(value: float, scale: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("checkpoint utility input must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("checkpoint utility scale must be finite and positive")
    return 1.0 / (1.0 + value / scale)


def _weighted_component_score(
    components: Mapping[str, object],
    configured_weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    component_values = {
        str(name): _as_float(value)
        for name, value in components.items()
    }
    if not component_values:
        raise ValueError("checkpoint score has no active components")
    if any(
        name not in configured_weights
        for name in component_values
    ):
        raise ValueError("checkpoint score has an unknown component")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in component_values.values()
    ):
        raise ValueError("checkpoint component utilities must be in [0, 1]")
    active_weight = sum(
        float(configured_weights[name]) for name in component_values
    )
    if not math.isfinite(active_weight) or active_weight <= 0.0:
        raise ValueError("active checkpoint weight must be positive")
    effective_weights = {
        name: float(configured_weights[name]) / active_weight
        for name in component_values
    }
    score = sum(
        component_values[name] * effective_weights[name]
        for name in component_values
    )
    if not math.isfinite(score):
        raise ValueError("composite checkpoint score is non-finite")
    return float(score), effective_weights


def score_checkpoint(
    aggregates: Mapping[str, object],
    availability: Mapping[str, object],
) -> dict[str, object]:
    """Compute the versioned weighted score under frozen availability."""
    ade_natural, ade_scene = _metric_pair(
        aggregates,
        "ade_3s_m",
    )
    fde_natural, fde_scene = _metric_pair(
        aggregates,
        "fde_3s_m",
    )
    natural_trajectory = (
        0.6 * _bounded_inverse(
            ade_natural,
            UTILITY_SCALES["ade_3s_m"],
        )
        + 0.4 * _bounded_inverse(
            fde_natural,
            UTILITY_SCALES["fde_3s_m"],
        )
    )
    scene_trajectory = (
        0.6 * _bounded_inverse(
            ade_scene,
            UTILITY_SCALES["ade_3s_m"],
        )
        + 0.4 * _bounded_inverse(
            fde_scene,
            UTILITY_SCALES["fde_3s_m"],
        )
    )
    components = {
        "trajectory": 0.5 * (
            natural_trajectory + scene_trajectory
        ),
        "comfort": _bounded_inverse(
            _combined_metric(aggregates, "comfort_excess"),
            UTILITY_SCALES["comfort_excess"],
        ),
    }

    if bool(availability.get("map_safety", False)):
        components["map_safety"] = _bounded_inverse(
            _combined_metric(aggregates, "offroad_excess"),
            UTILITY_SCALES["offroad_excess"],
        )
    if bool(availability.get("navigation", False)):
        wrong_branch_available = bool(
            availability.get("wrong_branch", False)
        )
        navigation_parts = {
            "route": (
                _bounded_inverse(
                    _combined_metric(aggregates, "route_gap"),
                    UTILITY_SCALES["route_gap"],
                ),
                0.5 if wrong_branch_available else 0.7,
            ),
        }
        if wrong_branch_available:
            navigation_parts["wrong_branch"] = (
                _bounded_inverse(
                    _combined_metric(
                        aggregates,
                        "wrong_branch_excess",
                    ),
                    UTILITY_SCALES["wrong_branch_excess"],
                ),
                0.3,
            )
        if bool(availability.get("destination", False)):
            navigation_parts["destination"] = (
                _bounded_inverse(
                    _combined_metric(
                        aggregates,
                        "destination_error_m",
                    ),
                    UTILITY_SCALES["destination_error_m"],
                ),
                0.2 if wrong_branch_available else 0.3,
            )
        navigation_weight = sum(
            weight for _, weight in navigation_parts.values()
        )
        components["navigation"] = sum(
            utility * weight
            for utility, weight in navigation_parts.values()
        ) / navigation_weight

    score, effective_weights = _weighted_component_score(
        components,
        TOP_LEVEL_WEIGHTS,
    )
    return {
        "policy_version": SELECTOR_POLICY_VERSION,
        "score": float(score),
        "components": components,
        "effective_weights": effective_weights,
        "availability": dict(availability),
        "utility_scales": dict(UTILITY_SCALES),
        "min_delta": SELECTOR_MIN_DELTA,
    }


def build_selector_calibration_report(
    selections: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report utility saturation and +/-20% top-level weight sensitivity."""
    if not selections:
        raise ValueError("selector calibration needs checkpoint selections")
    components_by_checkpoint = []
    expected_components = None
    for index, selection in enumerate(selections):
        if selection.get("policy_version") != SELECTOR_POLICY_VERSION:
            raise ValueError(
                "selector calibration received another policy version"
            )
        components = selection.get("components")
        if not isinstance(components, Mapping):
            raise ValueError(
                f"checkpoint selection {index} has no components"
            )
        normalized = {
            str(name): _as_float(value)
            for name, value in components.items()
        }
        _weighted_component_score(normalized, TOP_LEVEL_WEIGHTS)
        component_names = frozenset(normalized)
        if expected_components is None:
            expected_components = component_names
        elif component_names != expected_components:
            raise ValueError(
                "selector component availability changed across checkpoints"
            )
        components_by_checkpoint.append(normalized)

    sorted_component_names = sorted(expected_components or ())
    saturation = {}
    for name in sorted_component_names:
        values = np.asarray(
            [components[name] for components in components_by_checkpoint],
            dtype=np.float64,
        )
        near_zero_fraction = float(np.mean(values <= 0.01))
        near_one_fraction = float(np.mean(values >= 0.99))
        saturation[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "near_zero_fraction": near_zero_fraction,
            "near_one_fraction": near_one_fraction,
            "almost_always_saturated": bool(
                max(near_zero_fraction, near_one_fraction) >= 0.90
            ),
        }

    def ranking(weights: Mapping[str, float]) -> tuple[list[int], list[float]]:
        scores = [
            _weighted_component_score(components, weights)[0]
            for components in components_by_checkpoint
        ]
        order = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )
        return order, scores

    baseline_ranking, baseline_scores = ranking(TOP_LEVEL_WEIGHTS)
    baseline_positions = {
        checkpoint_index: rank
        for rank, checkpoint_index in enumerate(baseline_ranking)
    }
    scenarios = []
    for name in sorted_component_names:
        for multiplier in (0.8, 1.2):
            weights = dict(TOP_LEVEL_WEIGHTS)
            weights[name] *= multiplier
            scenario_ranking, scenario_scores = ranking(weights)
            if len(selections) < 2:
                rank_correlation = None
            else:
                scenario_positions = {
                    checkpoint_index: rank
                    for rank, checkpoint_index in enumerate(
                        scenario_ranking
                    )
                }
                baseline = np.asarray([
                    baseline_positions[index]
                    for index in range(len(selections))
                ], dtype=np.float64)
                scenario = np.asarray([
                    scenario_positions[index]
                    for index in range(len(selections))
                ], dtype=np.float64)
                rank_correlation = float(np.corrcoef(
                    baseline,
                    scenario,
                )[0, 1])
            scenarios.append({
                "component": name,
                "multiplier": multiplier,
                "ranking": scenario_ranking,
                "scores": scenario_scores,
                "top_checkpoint_unchanged": (
                    scenario_ranking[0] == baseline_ranking[0]
                ),
                "spearman_rank_correlation": rank_correlation,
            })

    return {
        "schema_version": SELECTOR_CALIBRATION_VERSION,
        "selector_policy_version": SELECTOR_POLICY_VERSION,
        "checkpoint_count": len(selections),
        "rank_evidence_sufficient": len(selections) >= 2,
        "baseline_ranking": baseline_ranking,
        "baseline_scores": baseline_scores,
        "component_saturation": saturation,
        "almost_always_saturated_components": [
            name for name in sorted_component_names
            if saturation[name]["almost_always_saturated"]
        ],
        "weight_sensitivity": scenarios,
    }


def validate_frozen_availability(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    *,
    calibration_atol: float = 1e-9,
) -> None:
    """Reject component drift while tolerating insignificant float noise."""
    if calibration_atol < 0.0:
        raise ValueError("calibration_atol must be non-negative")
    expected_discrete = {
        key: value
        for key, value in expected.items()
        if key != "calibration"
    }
    observed_discrete = {
        key: value
        for key, value in observed.items()
        if key != "calibration"
    }
    if expected_discrete != observed_discrete:
        raise ValueError(
            "checkpoint selector discrete availability changed during run: "
            f"expected={expected_discrete} actual={observed_discrete}"
        )
    expected_calibration = expected.get("calibration", {})
    observed_calibration = observed.get("calibration", {})
    if (
        not isinstance(expected_calibration, Mapping)
        or not isinstance(observed_calibration, Mapping)
        or set(expected_calibration) != set(observed_calibration)
    ):
        raise ValueError(
            "checkpoint selector calibration coverage changed during run"
        )
    mismatches = {
        key: {
            "expected": _as_float(expected_calibration[key]),
            "actual": _as_float(observed_calibration[key]),
        }
        for key in expected_calibration
        if not math.isclose(
            _as_float(expected_calibration[key]),
            _as_float(observed_calibration[key]),
            rel_tol=0.0,
            abs_tol=calibration_atol,
        )
    }
    if mismatches:
        raise ValueError(
            "checkpoint selector calibration changed during run: "
            f"{mismatches}"
        )


def score_is_better(
    score: float,
    best_score: float,
    *,
    min_delta: float = SELECTOR_MIN_DELTA,
) -> bool:
    if not math.isfinite(score) or not math.isfinite(best_score):
        raise ValueError("checkpoint scores must be finite")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    return score > best_score + min_delta
