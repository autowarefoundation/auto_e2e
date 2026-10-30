# SPDX-License-Identifier: Apache-2.0
"""Verify coverage, pose-derived inputs and independently rescore raw predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .common import (
    DENSE_UID_SHA256,
    EXPECTED,
    coverage,
    membership,
    read_json,
    sha256,
    uid_hash,
    write_json,
)
from .eval_kitscenes import verified_scene
from .metrics import independent_metrics, row_trajectories
from .protocol import UNIAD_TO_KIT, phase_inputs, validate_poses
from .shared_metrics import (
    SOURCE_COMMIT,
    SOURCE_SHA256,
    evaluate,
    load_metric_code,
    self_test,
)


def audit_inputs(scene, poses, track, full=False):
    """Re-derive CAN and rotation independently of the preparation helper."""
    streams = scene["tracks"][track]
    actual_ids = [f["sample_uid"] for s in streams for f in s["frames"] if f["scored"]]
    membership(actual_ids, actual_ids)
    if track == "keyframes" and full:
        expected = [
            f"kitscenes-v1-{scene['scene_id']}-f{i:06d}"
            for i in range(40, len(poses) - 30, 90)
        ]
        membership(actual_ids, expected)
    if track == "dense" and full:
        scored = [f["frame"] for s in streams for f in s["frames"] if f["scored"]]
        expected_phases = {p: indices for p, indices, _ in phase_inputs(scored)}
        if len(streams) != len(expected_phases):
            raise ValueError("Missing/duplicate dense phase stream")
    maximum = 0.0
    for stream in streams:
        if stream["physical_scene_id"] != scene["scene_id"]:
            raise ValueError("Mixed physical scenes")
        indices = [f["frame"] for f in stream["frames"]]
        if (
            not indices
            or min(indices) < 0
            or max(indices) >= len(poses)
            or np.any(np.diff(indices) != 5)
        ):
            raise ValueError("Invalid native 2 Hz stream")
        if track == "keyframes":
            scored_indices = [f["frame"] for f in stream["frames"] if f["scored"]]
            if (
                len(indices) != 9
                or scored_indices != [indices[-1]]
                or (indices[-1] - 40) % 90
            ):
                raise ValueError(
                    "Keyframe requires one anchor and nine causal frames spanning 4 seconds"
                )
        elif full and indices != expected_phases[indices[0] % 5]:
            raise ValueError("Dense phase history differs from evaluated protocol")
        for frame in stream["frames"]:
            i = frame["frame"]
            r = Rotation.from_quat(poses[i, 4:8])
            matrix = r.as_matrix()
            velocity = acceleration = angular_velocity = np.zeros(3)
            if i >= 2:
                dt, dt0 = (
                    poses[i, 0] - poses[i - 1, 0],
                    poses[i - 1, 0] - poses[i - 2, 0],
                )
                velocity = (poses[i, 1:4] - poses[i - 1, 1:4]) / dt
                previous = (poses[i - 1, 1:4] - poses[i - 2, 1:4]) / dt0
                acceleration = matrix.T @ ((velocity - previous) / ((dt + dt0) / 2))
                angular_velocity = (
                    Rotation.from_quat(poses[i - 1, 4:8]).inv() * r
                ).as_rotvec() / dt
            position = poses[i, 1:4] - poses[indices[0], 1:4]
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0]) % (2 * np.pi)
            expected = np.r_[
                position,
                r.as_quat()[[3, 0, 1, 2]],
                acceleration,
                angular_velocity,
                matrix.T @ velocity,
                yaw,
                np.degrees(yaw),
            ]
            for actual, wanted in (
                (frame["can_bus"], expected),
                (frame["l2g_t"], position),
                (frame["l2g_r_mat"], (matrix @ UNIAD_TO_KIT).T),
                (frame["timestamp"], poses[i, 0]),
            ):
                actual = np.asarray(actual)
                if actual.shape != np.asarray(wanted).shape:
                    raise ValueError("Input audit shape mismatch")
                error = float(np.max(np.abs(actual - wanted)))
                if not np.isfinite(error) or error > 1e-6:
                    raise ValueError("Pose-derived model input audit failed")
                maximum = max(maximum, error)
    return maximum


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_collection(data, targets, shared_code):
    result = {}
    maximum = 0.0
    for target_name, variants in data.items():
        result[target_name] = {}
        target = np.asarray(targets[target_name])
        for variant, rows in variants.items():
            prediction = np.asarray(rows)
            independent = independent_metrics(prediction, target)
            shared, counts = evaluate(shared_code, prediction, target)
            for horizon in ("1", "2", "3"):
                if counts[horizon] != dict(ade=len(target), fde=len(target)):
                    raise ValueError("Incomplete shared metric counts")
                for metric in ("ade_m", "fde_m"):
                    difference = abs(
                        independent[horizon][metric] - shared[horizon][metric]
                    )
                    maximum = max(maximum, difference)
                    if difference > 1e-5:
                        raise ValueError(
                            "Independent raw scoring and shared evaluator disagree"
                        )
            result[target_name][variant] = dict(
                interpolated_10hz=shared,
                native_2hz=independent_metrics(
                    prediction[:, 4::5], target[:, 4::5], hz=2
                ),
                independent_10hz=independent,
                shared_counts=counts,
            )
    return result, maximum


def audit(prepared, run, metric_source, output):
    if output.exists():
        raise FileExistsError("Choose a new audit output path")
    manifest, execution = (
        read_json(prepared / "manifest.json"),
        read_json(run / "status.json"),
    )
    if (
        manifest["status"] != "ready"
        or sha256(prepared / "manifest.json") != execution["prepared_manifest_sha256"]
    ):
        raise ValueError("Prepared input identity changed")
    shared = load_metric_code(metric_source)
    self_test(shared)
    report = dict(
        status="auditing",
        full_evaluation=manifest["full_evaluation"],
        tracks={},
        shared_metric_commit=SOURCE_COMMIT,
        shared_metric_sha256=SOURCE_SHA256,
        horizons_4_5="N/A",
    )
    try:
        for track in execution["tracks"]:
            data, targets, seen, expected_ids = {}, {}, [], []
            totals = dict.fromkeys(EXPECTED[track], 0)
            hashes, hardware, input_error = {}, [], 0.0
            for record in manifest["scenes"]:
                scene, directory = verified_scene(prepared, record)
                poses = np.loadtxt(directory / "source_metadata/poses.txt")
                validate_poses(
                    poses,
                    np.loadtxt(directory / "source_metadata/timestamp.reference.txt"),
                )
                input_error = max(
                    input_error,
                    audit_inputs(scene, poses, track, manifest["full_evaluation"]),
                )
                streams = scene["tracks"][track]
                expected_coverage = coverage(streams)
                destination = run / track / scene["scene_id"]
                receipt = read_json(destination / "run.json")
                if (
                    receipt["status"] != "predictions_complete"
                    or receipt["coverage"] != expected_coverage
                    or receipt["input_sha256"] != record["sha256"]
                    or not receipt["checkpoint_loaded_strict"]
                    or receipt.get("adapter_source_sha256")
                    != execution.get("adapter_source_sha256")
                ):
                    raise ValueError("Incomplete/mismatched scene run")
                raw_path = destination / "samples.jsonl"
                if sha256(raw_path) != receipt["samples_sha256"]:
                    raise ValueError("Raw prediction hash changed")
                hashes[scene["scene_id"]] = receipt["samples_sha256"]
                hardware.append(
                    {
                        k: receipt[k]
                        for k in (
                            "gpu",
                            "torch",
                            "cuda",
                            "versions",
                            "wall_seconds",
                            "model_frames",
                        )
                        if k in receipt
                    }
                )
                frames = {
                    f["sample_uid"]: (f, s["scene_id"])
                    for s in streams
                    for f in s["frames"]
                    if f["scored"]
                }
                rows = read_rows(raw_path)
                membership([r["sample_uid"] for r in rows], frames)
                expected_ids.extend(frames)
                for row in rows:
                    uid = row["sample_uid"]
                    frame, stream_id = frames[uid]
                    if (
                        row["prediction_frame"] != "body_FLU"
                        or row["frame"] != frame["frame"]
                        or row["timestamp"] != frame["timestamp"]
                        or row["scene_id"] != stream_id
                        or row["physical_scene_id"] != scene["scene_id"]
                        or row["fixed_command"] != 2
                    ):
                        raise ValueError("Prediction identity/coordinate mismatch")
                    predicted, truth = row_trajectories(row, frame, poses, track)
                    for name, values in truth.items():
                        targets.setdefault(name, []).append(values)
                        for variant, path in predicted[name].items():
                            data.setdefault(name, {}).setdefault(variant, []).append(
                                path
                            )
                    seen.append(uid)
                for key, value in expected_coverage.items():
                    totals[key] += value
            membership(seen, expected_ids)
            if (
                totals != manifest["coverage"][track]
                or uid_hash(seen) != manifest["sample_uid_sha256"][track]
            ):
                raise ValueError("Prepared and actual coverage differ")
            if manifest["full_evaluation"]:
                if totals != EXPECTED[track] or (
                    track == "dense" and uid_hash(seen) != DENSE_UID_SHA256
                ):
                    raise ValueError("Full validation coverage failed")
            scored, difference = score_collection(data, targets, shared)
            report["tracks"][track] = dict(
                coverage=totals,
                sample_uid_sha256=uid_hash(seen),
                metrics=scored,
                maximum_shared_difference_m=difference,
                maximum_input_audit_error=input_error,
                prediction_sha256=hashes,
                hardware_and_runtime=hardware,
            )
        report["status"] = "passed_independent_raw_rescore"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output, report)
    return report


def compare_samples(actual, reference):
    """Unchanged 0.05 m bound, equal IDs; accepts original archived sample files."""
    new, old = read_rows(actual), read_rows(reference)
    membership([r["sample_uid"] for r in new], [r["sample_uid"] for r in old])
    if not new:
        raise ValueError("Empty comparison")
    old = {r["sample_uid"]: r for r in old}
    maximum = 0.0
    for row in new:
        prior = old[row["sample_uid"]]
        if (
            row["frame"] != prior["frame"]
            or row["prediction_frame"] != prior["prediction_frame"]
        ):
            raise ValueError("Reference coordinate/frame mismatch")
        p = np.asarray(row["predictions_body_xy_2hz"]["fixed_forward"])
        q = np.asarray(prior["predictions_body_xy_2hz"]["fixed_forward"])
        if (
            p.shape != (6, 2)
            or q.shape != (6, 2)
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
        ):
            raise ValueError("Invalid comparison trajectories")
        maximum = max(maximum, float(np.max(np.abs(p - q))))
    return dict(
        status="passed" if maximum <= 0.05 else "failed",
        samples=len(new),
        maximum_prediction_difference_m=maximum,
        tolerance_m=0.05,
        actual_sha256=sha256(actual),
        reference_sha256=sha256(reference),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("rescore")
    score.add_argument("--prepared", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--metric-source", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file; previous audits are preserved")
    if args.command == "rescore":
        audit(args.prepared, args.run, args.metric_source, args.out)
    else:
        report = compare_samples(args.actual, args.reference)
        write_json(args.out, report)
        if report["status"] != "passed":
            raise SystemExit("Prediction comparison failed; outputs preserved")


if __name__ == "__main__":
    main()
