"""Scene-balanced validation aggregation and composite checkpoint scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


SELECTOR_POLICY_VERSION = "rollout_composite_selector_v1"
SELECTOR_MIN_DELTA = 0.0005
METRIC_NAMES = (
    "ade_3s_m",
    "fde_6_4s_m",
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
    "fde_6_4s_m",
    "comfort_excess",
)


def _finite_optional(record: Mapping[str, object], name: str) -> float | None:
    value = record.get(name)
    if value is None:
        return None
    numeric = float(value)
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
                float(record[name])
            )
        scene_means = [
            {
                "split_group_uid": group_uid,
                "value": float(np.mean(by_scene[group_uid])),
                "sample_count": len(by_scene[group_uid]),
            }
            for group_uid in sorted(by_scene)
        ]
        if eligible:
            natural = float(np.mean([
                float(record[name]) for record in eligible
            ]))
            scene_values = [
                float(scene["value"]) for scene in scene_means
            ]
            scene_balanced = float(np.mean(scene_values))
            scene_distribution = {
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
    metrics = aggregates["metrics"]
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
    metric = aggregates["metrics"][name]
    natural = metric["natural"]
    scene = metric["scene_balanced"]
    if natural is None or scene is None:
        raise ValueError(f"score metric {name} is unavailable")
    natural_value = float(natural)
    scene_value = float(scene)
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
    return 1.0 / (1.0 + value / scale)


def _clipped_utility(value: float, scale: float) -> float:
    return 1.0 - float(np.clip(value / scale, 0.0, 1.0))


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
        "fde_6_4s_m",
    )
    natural_trajectory = (
        0.6 * _bounded_inverse(ade_natural, 2.5)
        + 0.4 * _bounded_inverse(fde_natural, 6.0)
    )
    scene_trajectory = (
        0.6 * _bounded_inverse(ade_scene, 2.5)
        + 0.4 * _bounded_inverse(fde_scene, 6.0)
    )
    components = {
        "trajectory": 0.5 * (
            natural_trajectory + scene_trajectory
        ),
        "comfort": _clipped_utility(
            _combined_metric(aggregates, "comfort_excess"),
            0.15,
        ),
    }

    if bool(availability.get("map_safety", False)):
        components["map_safety"] = _clipped_utility(
            _combined_metric(aggregates, "offroad_excess"),
            0.10,
        )
    if bool(availability.get("navigation", False)):
        wrong_branch_available = bool(
            availability.get("wrong_branch", False)
        )
        navigation_parts = {
            "route": (
                _clipped_utility(
                    _combined_metric(aggregates, "route_gap"),
                    0.15,
                ),
                0.5 if wrong_branch_available else 0.7,
            ),
        }
        if wrong_branch_available:
            navigation_parts["wrong_branch"] = (
                _clipped_utility(
                    _combined_metric(
                        aggregates,
                        "wrong_branch_excess",
                    ),
                    1.0,
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
                    7.5,
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

    configured_weights = {
        "trajectory": 0.50,
        "comfort": 0.15,
        "map_safety": 0.15,
        "navigation": 0.20,
    }
    active_weight = sum(
        configured_weights[name] for name in components
    )
    effective_weights = {
        name: configured_weights[name] / active_weight
        for name in components
    }
    score = sum(
        components[name] * effective_weights[name]
        for name in components
    )
    if not math.isfinite(score):
        raise ValueError("composite checkpoint score is non-finite")
    return {
        "policy_version": SELECTOR_POLICY_VERSION,
        "score": float(score),
        "components": components,
        "effective_weights": effective_weights,
        "availability": dict(availability),
        "utility_scales": {
            "ade_3s_m": 2.5,
            "fde_6_4s_m": 6.0,
            "comfort_excess": 0.15,
            "offroad_excess": 0.10,
            "route_gap": 0.15,
            "destination_error_m": 7.5,
        },
        "min_delta": SELECTOR_MIN_DELTA,
    }


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
