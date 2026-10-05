"""Synthetic unit protocol for intervention cause *selection* (not #122).

This is **not** issue #122's external-truth validation and it is **not**
KIT LiDAR / object-GT recovery. We do not have those shards. What this
file is: a unit protocol for ``select_causes_by_intervention``.

Each scene is 2-D geometry (ego polyline + factor points). Truth,
``oracle_delta``, and ``blind_delta`` are *computed* from that geometry
— they are not typed onto the same line as ``in_conflict``. A reviewer
can move a factor and the labels change.

  * truth            — factor surface is inside the ~3.5 m path corridor
  * oracle_delta     — ``exp(-d / L)`` of distance-to-path (privileged
                       intervention proxy; same geometry, not a second
                       handwritten table)
  * blind_delta      — ``exp(-d / L)`` of distance to a distractor
                       attention point (model looks at the wrong place)
  * geometry teacher — independent occupancy of a *wider* forward envelope
                       (4 m), no intervention deltas, no taxonomy hash

Default provenance of the labeller under test remains
``counterfactual_weak`` (source weight 0.3). ``counterfactual_gt`` still
needs a real KIT LiDAR / HD-map pass; this protocol does not provide it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from data_processing.reasoning_label_generation.intervention_cause import (
    InterventionCauseCandidate,
    select_causes_by_intervention,
)


CONFLICT_HALF_WIDTH_M = 1.75
TEACHER_HALF_WIDTH_M = 4.0
DELTA_LENGTH_M = 2.0
EGO_PATH_LENGTH_M = 40.0

Point = tuple[float, float]
Path = tuple[Point, ...]


@dataclass(frozen=True)
class Factor:
    """A named object in the scene. No pre-typed conflict/delta fields."""

    name: str
    x: float
    y: float
    radius_m: float = 0.5


@dataclass(frozen=True)
class Scene:
    scene_id: str
    ego_path: Path
    factors: tuple[Factor, ...]
    # Where the blind labeller attends (not the ego path).
    attention_xy: Point


def _straight_path(length_m: float = EGO_PATH_LENGTH_M, n: int = 9) -> Path:
    return tuple((length_m * i / (n - 1), 0.0) for i in range(n))


def _point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float,
) -> tuple[float, float, float]:
    """Return ``(dist, along_on_segment, signed_cross)`` for one segment."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    seg_len = math.sqrt(ab2)
    if ab2 < 1e-18:
        return math.hypot(apx, apy), 0.0, 0.0
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx, qy = ax + t * abx, ay + t * aby
    dist = math.hypot(px - qx, py - qy)
    along = t * seg_len
    cross = (abx * apy - aby * apx) / seg_len
    return dist, along, cross


def project_to_path(x: float, y: float, path: Path) -> tuple[float, float, float]:
    """Nearest-point Frenet: ``(along_m, signed_cross_m, dist_m)``."""
    if len(path) < 2:
        raise ValueError("path must have at least two points")
    best_dist = float("inf")
    best_along = 0.0
    best_cross = 0.0
    arc = 0.0
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        dist, along, cross = _point_to_segment(x, y, ax, ay, bx, by)
        if dist < best_dist:
            best_dist = dist
            best_along = arc + along
            best_cross = cross
        arc += math.hypot(bx - ax, by - ay)
    return best_along, best_cross, best_dist


def surface_distance_to_path(factor: Factor, path: Path) -> float:
    _, _, dist = project_to_path(factor.x, factor.y, path)
    return max(0.0, dist - factor.radius_m)


def in_conflict(
    factor: Factor,
    path: Path,
    *,
    half_width_m: float = CONFLICT_HALF_WIDTH_M,
) -> bool:
    return surface_distance_to_path(factor, path) < half_width_m


def intervention_proxy(distance_m: float, *, length_m: float = DELTA_LENGTH_M) -> float:
    """Monotone stand-in for trajectory L2: closer → larger delta."""
    return math.exp(-distance_m / length_m)


def oracle_delta(factor: Factor, path: Path) -> float:
    return intervention_proxy(surface_distance_to_path(factor, path))


def blind_delta(factor: Factor, attention_xy: Point) -> float:
    dx = factor.x - attention_xy[0]
    dy = factor.y - attention_xy[1]
    dist = max(0.0, math.hypot(dx, dy) - factor.radius_m)
    return intervention_proxy(dist)


def geometry_teacher_causes(
    scene: Scene,
    *,
    half_width_m: float = TEACHER_HALF_WIDTH_M,
    max_along_m: float = EGO_PATH_LENGTH_M,
) -> set[str]:
    """Independent labeller: wide forward occupancy, no intervention deltas.

    Uses a 4 m envelope (two-lane) rather than the 1.75 m conflict corridor,
    and ignores the selector entirely. This is not ``MockTeacher`` (which
    hashes ``sample_id`` into the 27-label taxonomy).
    """
    names: set[str] = set()
    for factor in scene.factors:
        along, cross, _ = project_to_path(factor.x, factor.y, scene.ego_path)
        if 0.0 <= along <= max_along_m and abs(cross) < half_width_m:
            names.add(factor.name)
    return names


def scene_truth(scene: Scene) -> set[str]:
    return {f.name for f in scene.factors if in_conflict(f, scene.ego_path)}


def scene_candidates(scene: Scene, labeller: str) -> list[InterventionCauseCandidate]:
    out: list[InterventionCauseCandidate] = []
    for factor in scene.factors:
        if labeller == "oracle":
            delta = oracle_delta(factor, scene.ego_path)
        elif labeller == "blind":
            delta = blind_delta(factor, scene.attention_xy)
        else:
            raise ValueError(f"unknown labeller {labeller!r}")
        out.append(InterventionCauseCandidate(name=factor.name, trajectory_l2=delta))
    return out


def _scene(
    scene_id: str,
    factors: tuple[Factor, ...],
    attention: Point,
    path: Path | None = None,
) -> Scene:
    return Scene(scene_id, path or _straight_path(), factors, attention)


# Constructed 2-D layouts. Only positions are authored; conflict and deltas
# are derived in the functions above.
SCENES: tuple[Scene, ...] = (
    _scene("ped_in_crosswalk", (
        Factor("pedestrian_crossing", 12.0, 0.4, 0.4),
        Factor("lead_vehicle", 20.0, 3.2, 1.0),
        Factor("vru_conflict", 6.0, 10.0, 0.4),
    ), attention=(20.0, 3.2)),
    _scene("vru_entering_path", (
        Factor("vru_conflict", 10.0, 0.8, 0.4),
        Factor("route_turn", 35.0, 5.0, 0.3),
        Factor("occlusion", 4.0, 12.0, 0.5),
    ), attention=(35.0, 5.0)),
    _scene("red_light_ahead", (
        Factor("red_light", 30.0, 0.2, 0.3),
        Factor("poor_visibility", 15.0, 15.0, 0.5),
        Factor("lead_vehicle", 18.0, 3.0, 1.0),
    ), attention=(30.0, 0.2)),
    _scene("stopped_lead", (
        Factor("stopped_lead_vehicle", 16.0, 0.2, 1.0),
        Factor("pedestrian_about_to_cross", 10.0, 5.0, 0.4),
        Factor("route_lane_change", 25.0, 6.0, 0.3),
    ), attention=(16.0, 0.2)),
    _scene("cut_in", (
        Factor("cut_in_vehicle", 14.0, 1.0, 1.0),
        Factor("slow_lead_vehicle", 22.0, 3.4, 1.0),
        Factor("unknown_cause", 8.0, 12.0, 0.5),
    ), attention=(22.0, 3.4)),
    _scene("cross_traffic", (
        Factor("cross_traffic", 15.0, 0.5, 1.0),
        Factor("oncoming_vehicle", 20.0, 3.5, 1.0),
        Factor("yellow_light", 30.0, 8.0, 0.3),
    ), attention=(20.0, 3.5)),
    _scene("object_blocking", (
        Factor("object_blocking_path", 18.0, 0.3, 0.8),
        Factor("construction_blocking_path", 18.0, 4.5, 0.8),
        Factor("poor_visibility", 5.0, 14.0, 0.5),
    ), attention=(18.0, 0.3)),
    _scene("stop_sign", (
        Factor("stop_sign", 25.0, 0.4, 0.3),
        Factor("yield_sign", 25.0, 3.8, 0.3),
        Factor("route_merge", 10.0, 10.0, 0.3),
    ), attention=(25.0, 3.8)),
    _scene("lane_ending", (
        Factor("lane_ending", 28.0, 0.2, 0.3),
        Factor("blocked_lane", 28.0, 3.5, 0.5),
        Factor("lead_vehicle", 12.0, 3.2, 1.0),
    ), attention=(28.0, 0.2)),
    _scene("ped_about_to_cross", (
        Factor("pedestrian_about_to_cross", 11.0, 1.2, 0.4),
        Factor("human_direction", 11.0, 3.9, 0.4),
        Factor("lead_vehicle", 20.0, 3.2, 1.0),
    ), attention=(11.0, 3.9)),
    _scene("road_closed", (
        Factor("road_closed", 22.0, 0.0, 0.8),
        Factor("route_turn", 8.0, 3.5, 0.3),
        Factor("uncertainty_high", 3.0, 12.0, 0.5),
    ), attention=(22.0, 0.0)),
    # Weaker on-path factor sits inside the 1.75 m corridor but below
    # relative_to_max * peak, so the default selector must FN. That is
    # the point: this protocol can fail.
    _scene("slippery_plus_lead", (
        Factor("slippery_road", 8.0, 0.0, 0.5),
        Factor("slow_lead_vehicle", 22.0, 2.5, 1.0),
        Factor("occlusion", 10.0, 8.0, 0.5),
    ), attention=(10.0, 8.0)),
)


def _prf(pred: Iterable[str], truth: Iterable[str]) -> dict[str, float]:
    p, t = set(pred), set(truth)
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _micro(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_scenes": float(len(rows))}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def score_labeller(
    scenes: Sequence[Scene],
    labeller: str,
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
    top_k: int | None = 3,
) -> dict[str, Any]:
    rows = []
    per_scene = []
    for scene in scenes:
        pred = select_causes_by_intervention(
            scene_candidates(scene, labeller),
            min_delta=min_delta,
            relative_to_max=relative_to_max,
            top_k=top_k,
        )
        stats = _prf(pred, scene_truth(scene))
        rows.append(stats)
        per_scene.append({
            "scene_id": scene.scene_id,
            "truth": sorted(scene_truth(scene)),
            "pred": pred,
            **stats,
        })
    return {"micro": _micro(rows), "scenes": per_scene}


def geometry_teacher_jaccard(
    scenes: Sequence[Scene],
    labeller: str,
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
) -> dict[str, float]:
    scores = []
    for scene in scenes:
        pred = set(select_causes_by_intervention(
            scene_candidates(scene, labeller),
            min_delta=min_delta,
            relative_to_max=relative_to_max,
        ))
        scores.append(_jaccard(pred, geometry_teacher_causes(scene)))
    return {
        "mean_jaccard": float(sum(scores) / max(len(scores), 1)),
        "n": float(len(scores)),
    }


def run_cause_label_unit_protocol(
    *,
    min_delta: float = 1e-3,
    relative_to_max: float | None = 0.5,
) -> dict[str, Any]:
    oracle = score_labeller(
        SCENES, "oracle", min_delta=min_delta, relative_to_max=relative_to_max,
    )
    blind = score_labeller(
        SCENES, "blind", min_delta=min_delta, relative_to_max=relative_to_max,
    )
    return {
        "claim": "synthetic_unit_protocol",
        "source": "synthetic_geometry_unit_protocol",
        "not_claimed": [
            "kit_lidar_object_gt",
            "issue_122_external_truth_validation",
        ],
        "n_scenes": len(SCENES),
        "min_delta": min_delta,
        "relative_to_max": relative_to_max,
        "conflict_half_width_m": CONFLICT_HALF_WIDTH_M,
        "teacher_half_width_m": TEACHER_HALF_WIDTH_M,
        "provenance_emitted": "counterfactual_weak",
        "oracle_vs_geometry": oracle["micro"],
        "blind_vs_geometry": blind["micro"],
        "oracle_vs_geometry_teacher_jaccard": geometry_teacher_jaccard(
            SCENES, "oracle", min_delta=min_delta, relative_to_max=relative_to_max,
        ),
        "blind_vs_geometry_teacher_jaccard": geometry_teacher_jaccard(
            SCENES, "blind", min_delta=min_delta, relative_to_max=relative_to_max,
        ),
        "oracle_scenes": oracle["scenes"],
        "blind_scenes": blind["scenes"],
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_cause_label_unit_protocol(min_delta=args.min_delta)
    text = json.dumps(
        {k: v for k, v in report.items() if k != "blind_scenes"},
        indent=2,
    )
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
