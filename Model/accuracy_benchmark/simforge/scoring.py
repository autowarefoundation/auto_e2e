"""ADE/FDE definitions, aggregation and plots for the SIMFORGE-D1.5 KITScenes evaluation.

Definitions (identical to the posted issue #210 results):

* ``first``: trajectory index 0 of the six candidates;
* ``mean``: the point-wise average of the six candidates, then ADE/FDE;
* ``min``: the best of the six candidates per anchor. This is an oracle
  diagnostic (it uses the ground truth to pick a candidate) and is labelled as such.

ADE averages the Euclidean XY error over the first ``h`` steps of a horizon of
``h`` seconds; FDE is the error at step ``h``. Horizon ``h`` s is waypoint
index ``10 h - 1`` (1 s -> 9, 5 s -> 49).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from accuracy_benchmark.simforge.contracts import SCORING_HORIZONS_S, TIME_STEP_S

METRIC_NAMES: tuple[str, ...] = ("ade_first", "fde_first", "ade_mean", "fde_mean", "min_ade", "min_fde")
ORACLE_METRICS: frozenset[str] = frozenset({"min_ade", "min_fde"})


def horizon_label(horizon_s: float) -> str:
    return f"{float(horizon_s):g}s"


def horizon_steps(horizon_s: float) -> int:
    return int(round(float(horizon_s) / TIME_STEP_S))


def sample_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, valid: np.ndarray, horizons_s: Sequence[float] = SCORING_HORIZONS_S) -> dict[str, dict[str, float] | None]:
    """Six metrics per horizon for one anchor. ``pred_xy``: (K, 64, 2); ``gt_xy``: (64, 2)."""
    predictions = np.asarray(pred_xy, dtype=np.float64)
    truth = np.asarray(gt_xy, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if predictions.ndim != 3 or predictions.shape[2] != 2 or truth.shape != predictions.shape[1:]:
        raise ValueError("pred_xy must be (K, T, 2) and gt_xy (T, 2)")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must be finite")
    errors = np.linalg.norm(predictions - truth[None], axis=2)  # (K, T)
    mean_errors = np.linalg.norm(predictions.mean(axis=0) - truth, axis=1)  # (T,)
    results: dict[str, dict[str, float] | None] = {}
    for horizon in horizons_s:
        steps = horizon_steps(horizon)
        label = horizon_label(horizon)
        if steps > predictions.shape[1] or not mask[:steps].all():
            results[label] = None  # genuine ground truth does not cover this horizon
            continue
        per_candidate_ade = errors[:, :steps].mean(axis=1)
        per_candidate_fde = errors[:, steps - 1]
        results[label] = {
            "ade_first": float(per_candidate_ade[0]),
            "fde_first": float(per_candidate_fde[0]),
            "ade_mean": float(mean_errors[:steps].mean()),
            "fde_mean": float(mean_errors[steps - 1]),
            "min_ade": float(per_candidate_ade.min()),
            "min_fde": float(per_candidate_fde.min()),
        }
    return results


def aggregate(per_anchor: Sequence[Mapping[str, Any]], horizons_s: Sequence[float] = SCORING_HORIZONS_S) -> dict[str, dict[str, Any]]:
    """Per horizon: number of anchors with genuine GT and the mean of each metric over them."""
    summary: dict[str, dict[str, Any]] = {}
    for horizon in horizons_s:
        label = horizon_label(horizon)
        rows = [item["metrics"][label] for item in per_anchor if item["metrics"].get(label) is not None]
        entry: dict[str, Any] = {"n": len(rows)}
        for name in METRIC_NAMES:
            entry[name] = float(np.mean([row[name] for row in rows])) if rows else None
        summary[label] = entry
    return summary


def write_report(output_dir: Path, per_anchor: Sequence[Mapping[str, Any]], *, header: Mapping[str, Any]) -> dict[str, Path]:
    """``per_anchor.csv``, ``summary.json`` and ``summary.md`` with the oracle labels attached."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_anchor.csv"
    fields = ["sample_uid", "scene_id", "frame_index"] + [
        f"{name}_{horizon_label(h)}" for h in SCORING_HORIZONS_S for name in METRIC_NAMES
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in per_anchor:
            row: dict[str, Any] = {"sample_uid": item["sample_uid"], "scene_id": item["scene_id"], "frame_index": item["frame_index"]}
            for h in SCORING_HORIZONS_S:
                metrics = item["metrics"].get(horizon_label(h))
                for name in METRIC_NAMES:
                    row[f"{name}_{horizon_label(h)}"] = None if metrics is None else metrics[name]
            writer.writerow(row)
    summary = {
        **dict(header),
        "anchor_count": len(per_anchor),
        "metric_definitions": {
            "ade_first": "ADE of candidate 0",
            "fde_first": "FDE of candidate 0",
            "ade_mean": "ADE of the point-wise mean of the six candidates",
            "fde_mean": "FDE of the point-wise mean of the six candidates",
            "min_ade": "ORACLE DIAGNOSTIC: minimum per-candidate ADE over the six candidates",
            "min_fde": "ORACLE DIAGNOSTIC: minimum per-candidate FDE over the six candidates",
        },
        "aggregate": aggregate(per_anchor),
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path = output_dir / "summary.md"
    md_path.write_text(markdown_table(summary), encoding="utf-8")
    return {"per_anchor_csv": csv_path, "summary_json": json_path, "summary_md": md_path}


def markdown_table(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary.get('label', '')}",
        "",
        "ADE/FDE in metres (XY). `minADE`/`minFDE` are ORACLE DIAGNOSTICS (best of six, chosen with the ground truth).",
        "",
        "| Horizon | n | ADE First | FDE First | ADE Mean | FDE Mean | minADE (oracle) | minFDE (oracle) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, entry in summary["aggregate"].items():
        if entry["n"] == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {label} | {entry['n']} | {entry['ade_first']:.3f} | {entry['fde_first']:.3f} | "
            f"{entry['ade_mean']:.3f} | {entry['fde_mean']:.3f} | {entry['min_ade']:.3f} | {entry['min_fde']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot_anchor(
    path: Path,
    *,
    title: str,
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    gt_valid: np.ndarray,
) -> Path:
    """Ground truth against the six candidates in the t0 frame (x forward up, y left drawn left)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 7.5))
    for candidate in pred_xy:
        axis.plot(-candidate[:, 1], candidate[:, 0], color="tab:blue", alpha=0.35, linewidth=1)
    gt = gt_xy[np.asarray(gt_valid, dtype=bool)]
    if len(gt):
        axis.plot(-gt[:, 1], gt[:, 0], color="black", linewidth=2, label="ground truth")
    axis.plot(0, 0, marker="*", color="tab:red", markersize=12, label="t0 (rear axle)")
    axis.set_xlabel("lateral y [m] (left drawn left)")
    axis.set_ylabel("forward x [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.set_title(title, fontsize=8)
    axis.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
