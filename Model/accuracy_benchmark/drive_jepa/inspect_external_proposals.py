"""Verify saved benchmark identity and score retained native proposals offline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from accuracy_benchmark.drive_jepa.source import sdk_split_scene_ids
from accuracy_benchmark.drive_jepa.external_kitscenes import (
    EXPECTED_VAL_SAMPLE_UID_SHA256,
    published_digest_ids,
    resample_prediction,
    score_predictions,
    write_json,
)


def candidate_totals(errors):
    """Per-sample candidate first/mean and independent ADE/FDE oracle minima."""
    if errors.ndim != 3 or not torch.isfinite(errors).all():
        raise ValueError("Expected finite [sample,candidate,time] displacement errors")
    results = {
        policy: {} for policy in ("candidate_first", "candidate_mean", "oracle_min")
    }
    for seconds in (1, 2, 3, 4):
        count = seconds * 10
        if errors.shape[-1] < count:
            continue
        ade = errors[..., :count].mean(-1)
        fde = errors[..., count - 1]
        reductions = {
            "candidate_first": (ade[:, 0], fde[:, 0]),
            "candidate_mean": (ade.mean(-1), fde.mean(-1)),
            "oracle_min": (ade.min(-1).values, fde.min(-1).values),
        }
        for policy, (a, f) in reductions.items():
            results[policy][seconds] = np.array(
                [a.double().sum().item(), len(a), f.double().sum().item(), len(f)]
            )
    return results


def inspect(results: Path, baseline: Path):
    scenes = set(sdk_split_scene_ids("val"))
    if len(scenes) != 117 or {p.stem for p in results.glob("*.npz")} != scenes:
        raise ValueError("Incomplete or different scene population")
    totals = {
        p: {s: np.zeros(4) for s in (1, 2, 3, 4)}
        for p in ("candidate_first", "candidate_mean", "oracle_min")
    }
    selected_totals = {f"{s}s": np.zeros(4) for s in (1, 2, 3, 4, 5)}
    all_uids = []
    candidate_count = None
    for scene in sorted(scenes):
        with (
            np.load(results / f"{scene}.npz", allow_pickle=False) as result,
            np.load(baseline / f"{scene}.npz", allow_pickle=False) as base,
        ):
            for key in ("sample_uid", "timestamp_ns", "target_xy"):
                if not np.array_equal(result[key], base[key]):
                    raise ValueError(
                        f"Baseline identity or target mismatch: {scene}/{key}"
                    )
            proposals = result["native_proposals"]
            scores = result["proposal_scores"]
            times = result["proposal_times_s"]
            count = proposals.shape[1]
            if (
                proposals.shape[2:] != (8, 3)
                or scores.shape != proposals.shape[:2]
                or not np.isfinite(proposals).all()
                or not np.isfinite(scores).all()
            ):
                raise ValueError("Invalid saved native proposal data")
            if candidate_count is not None and count != candidate_count:
                raise ValueError("Candidate count changed across scenes")
            candidate_count = count
            resampled = np.stack(
                [
                    resample_prediction(p[:, :2], times)
                    for p in proposals.reshape(-1, 8, 3)
                ]
            )
            resampled = resampled.reshape(len(proposals), count, -1, 2)
            chosen = resampled[np.arange(len(proposals)), scores.argmax(-1)]
            if not np.array_equal(chosen, result["prediction_xy"]):
                raise ValueError(
                    "Saved selection differs from native learned-score argmax"
                )
            errors = torch.linalg.vector_norm(
                torch.from_numpy(resampled)
                - torch.from_numpy(result["target_xy"][:, None, : resampled.shape[2]]),
                dim=-1,
            )
            for policy, horizons in candidate_totals(errors).items():
                for seconds, values in horizons.items():
                    totals[policy][seconds] += values
            for horizon, values in score_predictions(
                chosen, result["target_xy"]
            ).items():
                selected_totals[horizon] += values
            all_uids.extend(result["sample_uid"].tolist())
    digest = published_digest_ids(all_uids)
    if (
        len(all_uids) != 11035
        or len(set(all_uids)) != 11035
        or digest != EXPECTED_VAL_SAMPLE_UID_SHA256
    ):
        raise ValueError("Published validation sample population differs")
    metrics: dict[str, dict[str, float | None]] = {}
    for policy, horizons in totals.items():
        metrics[policy] = {
            f"{metric}_{seconds}s_m": float(values[index] / values[index + 1])
            for seconds, values in horizons.items()
            for metric, index in (("ade", 0), ("fde", 2))
        }
        metrics[policy].update(ade_5s_m=None, fde_5s_m=None)
    metrics["selected"] = {
        f"{metric}_{horizon}_m": float(values[index] / values[index + 1])
        if values[index + 1]
        else None
        for horizon, values in selected_totals.items()
        for metric, index in (("ade", 0), ("fde", 2))
    }
    report = json.loads((results / "report.json").read_text())
    for key, value in metrics["selected"].items():
        if value != report["metrics"]["trajectory"][key]:
            raise ValueError(
                "Recomputed selected metric differs from evaluation report"
            )
    audit = {
        "scene_count": 117,
        "sample_count": 11035,
        "candidate_count": candidate_count,
        "published_sample_uid_sha256": digest,
        "baseline_ids_timestamps_targets_identical": True,
        "selected_matches_learned_score_argmax": True,
        "selected_metrics_recomputed": True,
        "metrics": metrics,
        "units": "metres",
        "candidate_mean_policy": "mean candidate error per sample, then equal-weight mean over samples",
        "oracle_min_policy": "minimum candidate ADE/FDE independently per sample and horizon; uses targets, not deployed selection",
        "interpolation_policy": "same 10Hz origin interpolation as selected trajectory; no extrapolation",
        "5s_unavailable_reason": "official checkpoint predicts 4 seconds",
        "command_provenance": report.get("command_provenance"),
        "adapter_input_policy": report.get("adapter_input_policy"),
    }
    write_json(results / "proposal_metrics.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path(".benchmark/results/drive_jepa_val")
    )
    args = parser.parse_args()
    audit = inspect(args.results, args.baseline)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in ("scene_count", "sample_count", "candidate_count", "metrics")
            },
            indent=2,
        )
    )
