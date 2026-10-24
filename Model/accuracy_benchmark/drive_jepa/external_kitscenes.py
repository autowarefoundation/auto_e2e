"""Local, model-independent raw KITScenes val evaluation and prediction export."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from accuracy_benchmark.drive_jepa.source import (
    KITSCENES_DATA_REVISION,
    KITSCENES_SDK_REVISION,
    PinnedKITScenesDownloader,
    sdk_split_scene_ids,
)
from data_processing.contract_versions import UID_SCHEMA_VERSION
from accuracy_benchmark.drive_jepa.trajectory_scoring import (
    DISPLACEMENT_HORIZON_STEPS,
    displacement_horizon_totals,
)

POPULATION_CAMERAS = (
    "camera_base_front_center",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear_left",
    "camera_ring_rear",
    "camera_ring_rear_right",
)
EXPECTED_VAL_SAMPLES = 11_035
# Published v3.5 epoch-5 overlay replay report at the immutable mirror revision.
EXPECTED_VAL_SAMPLE_UID_SHA256 = (
    "00f824202c97b47ec74561d247e8e30686e5505eef2ad2bc0acba97e41ce9925"
)
REFERENCE_REPORT_URL = (
    "https://huggingface.co/AutowareFoundation/auto_e2e/resolve/"
    "7262c151599c1b5e4a99f0da9c3196ff9ff615fe/"
    "evaluations/overlay-replay-v1/kitscenes-val/kitscenes-epoch-5/report.json"
)


def digest_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def published_digest_ids(ids):
    """Overlay replay reports hash sorted UID lines with a terminal LF.

    Reactive evaluation uses no terminal LF, so preserve both explicitly.
    """
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def check_sdk(root):
    import kitscenes

    if git_revision(root) != KITSCENES_SDK_REVISION:
        raise ValueError(f"SDK checkout must be {KITSCENES_SDK_REVISION}")
    if Path(kitscenes.__file__).resolve().parent != (root / "kitscenes").resolve():
        raise ValueError("Imported kitscenes differs from --sdk-root; set PYTHONPATH")


def audit_episode_ids(path, sdk_root):
    supplied = Path(path).read_text(encoding="utf-8-sig").split()
    if len(supplied) != len(set(supplied)):
        raise ValueError("Episode list contains duplicate IDs")
    val = set(sdk_split_scene_ids("val"))
    overlap_path = (
        sdk_root
        / "kitscenes/split/generated_splits/default_geo_split_v1_0/overlap_train_val.txt"
    )
    overlap = set(overlap_path.read_text().split())
    unknown = set(supplied) - val - overlap
    if unknown or val - set(supplied):
        raise ValueError(
            f"Episode list differs: unknown={sorted(unknown)}, missing_val={sorted(val - set(supplied))}"
        )
    return {
        "supplied_episode_count": len(supplied),
        "official_val_count": len(val),
        "overlap_count": len(set(supplied) & overlap),
        "exact_val_plus_overlap": set(supplied) == val | overlap,
        "scene_uid_sha256": digest_ids([f"kitscenes-{scene}" for scene in val]),
    }


def contiguous_length(indices):
    for expected, actual in enumerate(indices):
        if expected != actual:
            return expected
    return len(indices)


def scene_samples(dataset, scene_id):
    """Match AutoE2E v3.5 benchmark enumeration without resizing images/maps."""
    loader = dataset.get_sensor_loader(scene_id)
    poses = dataset.get_scene(scene_id).ego_poses
    reference = loader.get_reference_timestamps()
    usable = min(
        len(poses),
        len(reference),
        *[
            contiguous_length(loader.get_frame_indices(camera))
            for camera in POPULATION_CAMERAS
        ],
    )
    if usable < 91:
        raise ValueError(
            f"Scene {scene_id} cannot satisfy 40-history/50-future selection"
        )
    poses = poses[:usable]
    stamps = np.array([p.timestamp_ns for p in poses], dtype=np.int64)
    if (
        np.any(np.diff(stamps) <= 0)
        or np.max(np.abs(stamps - reference[:usable])) > 50_000_000
    ):
        raise ValueError(f"Pose/reference timestamps are misaligned in {scene_id}")
    # Horizons and image history rely on the original 10 Hz timeline.
    nominal = stamps[0] + np.arange(usable, dtype=np.int64) * 100_000_000
    if np.max(np.abs(stamps - nominal)) > 50_000_000:
        raise ValueError(f"Scene {scene_id} does not have an aligned 10 Hz timeline")
    positions = np.array([p.translation[:2] for p in poses], dtype=np.float64)
    yaw = Rotation.from_quat([p.rotation for p in poses]).as_euler("ZYX")[:, 0]
    for frame in range(40, usable - 50):
        # Causal status: use only current/past poses, never future GT derivatives.
        dt_now = (stamps[frame] - stamps[frame - 1]) / 1e9
        dt_past = (stamps[frame - 1] - stamps[frame - 2]) / 1e9
        velocity_world = (positions[frame] - positions[frame - 1]) / dt_now
        previous_velocity = (positions[frame - 1] - positions[frame - 2]) / dt_past
        acceleration_world = (velocity_world - previous_velocity) / (
            (dt_now + dt_past) / 2
        )
        cosine, sine = np.cos(yaw[frame]), np.sin(yaw[frame])
        to_ego = np.array([[cosine, sine], [-sine, cosine]])
        target = (positions[frame + 1 : frame + 51] - positions[frame]) @ to_ego.T
        yield {
            "scene_id": scene_id,
            "frame_idx": frame,
            "timestamp_ns": int(stamps[frame]),
            "sample_uid": f"kitscenes-{UID_SCHEMA_VERSION}-{scene_id}-f{frame:06d}",
            "target_xy": target.astype(np.float32),
            "loader": loader,
            "velocity_xy": (to_ego @ velocity_world).astype(np.float32),
            "acceleration_xy": (to_ego @ acceleration_world).astype(np.float32),
        }


def resample_prediction(prediction, times):
    """Interpolate at 10 Hz from origin; never extrapolate past native support."""
    prediction = np.asarray(prediction, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if (
        prediction.shape != (len(times), 2)
        or not np.isfinite(prediction).all()
        or not np.isfinite(times).all()
        or not len(times)
        or times[0] <= 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Adapter must return finite ego FLU metres at positive increasing times"
        )
    steps = min(50, int(np.floor(times[-1] * 10 + 1e-7)))
    if steps < 10:
        raise ValueError("Adapter must predict at least 1 second")
    query = np.arange(1, steps + 1) / 10
    source_t = np.concatenate([[0.0], times])
    source_xy = np.concatenate([np.zeros((1, 2)), prediction], axis=0)
    return np.column_stack(
        [np.interp(query, source_t, source_xy[:, axis]) for axis in range(2)]
    ).astype(np.float32)


def score_predictions(predictions, targets):
    steps = predictions.shape[1]
    errors = torch.linalg.vector_norm(
        torch.from_numpy(predictions) - torch.from_numpy(targets[:, :steps]),
        dim=-1,
    )
    horizons = {
        name: count
        for name, count in DISPLACEMENT_HORIZON_STEPS.items()
        if count <= min(steps, 50)
    }
    return displacement_horizon_totals(
        errors, torch.ones_like(errors, dtype=torch.bool), horizons
    )


@contextmanager
def raw_scene(args, scene_id):
    from kitscenes import KITScenesDataset

    if args.raw_root is not None:
        yield KITScenesDataset(root=args.raw_root, split="val")
        return
    args.staging_root.mkdir(parents=True, exist_ok=True)
    # Only a directory created by this run is removed; user datasets are preserved.
    with tempfile.TemporaryDirectory(
        prefix="kitscenes_scene_", dir=args.staging_root
    ) as temporary:
        downloader = PinnedKITScenesDownloader(
            temporary, revision=args.dataset_revision
        )
        archive = downloader.archives[scene_id]
        required = 2 * archive.size_bytes + 5 * 1024**3
        if shutil.disk_usage(temporary).free < required:
            raise RuntimeError(
                f"Need at least {required / 1024**3:.1f} GiB staging space for {scene_id}"
            )
        downloader.download([scene_id], expected_split="val")
        yield KITScenesDataset(root=temporary, split="val")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate(args):
    check_sdk(args.sdk_root)
    official = sdk_split_scene_ids("val")
    selected = args.scene or official
    if len(selected) != len(set(selected)) or set(selected) - set(official):
        raise ValueError("--scene must contain unique official val UUIDs")
    if not args.stream and args.raw_root is None:
        raise ValueError("Provide --raw-root or --stream")
    args.commands = {}
    if args.command_file:
        args.commands = json.loads(args.command_file.read_text())
        for command in args.commands.values():
            if len(command) != 4 or not np.isfinite(command).all():
                raise ValueError(
                    "Command file values must be finite four-dimensional vectors"
                )
    if not np.isfinite(args.command).all():
        raise ValueError("Fallback command must be finite")
    command_provenance = None
    if args.command_file:
        metadata_path = args.command_file.with_suffix(".provenance.json")
        if metadata_path.exists():
            command_provenance = json.loads(metadata_path.read_text(encoding="utf-8"))
            if command_provenance.get("commands_sha256") != file_sha256(
                args.command_file
            ):
                raise ValueError("Command provenance does not match the command file")
    provenance = {
        "schema_version": "external_kitscenes_val_v1",
        "sdk_revision": KITSCENES_SDK_REVISION,
        "dataset_revision": args.dataset_revision,
        "raw_revision_verification": "archive_sha256"
        if args.stream
        else "user_supplied_unverified",
        "reference_report_url": REFERENCE_REPORT_URL,
        "expected_sample_uid_sha256": EXPECTED_VAL_SAMPLE_UID_SHA256,
        "auto_e2e_revision": git_revision(Path(__file__).resolve().parents[3]),
        "drive_jepa_revision": git_revision(args.drive_jepa_root),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "adapter": args.adapter,
        "camera": args.camera,
        "precision": args.precision,
        "evaluator_sha256": file_sha256(__file__),
        "scoring_sha256": file_sha256(
            Path(__file__).with_name("trajectory_scoring.py")
        ),
        "adapter_sha256": file_sha256(
            importlib.import_module(args.adapter.split(":", 1)[0]).__file__
        ),
        "fallback_command": args.command,
        "command_file_sha256": file_sha256(args.command_file)
        if args.command_file
        else None,
        "input_track": "camera_ego_command"
        if args.command_file
        else "camera_ego_missing_command",
        "history_steps": 40,
        "future_steps": 50,
        "frequency_hz": 10,
        "image_history_interval_s": 0.5,
        "status_policy": "causal_backward_pose_differences",
        "trajectory_policy": "ego_flu_xy_linear_interpolation_origin_no_extrapolation",
        "selected_scene_ids": sorted(selected),
    }
    if command_provenance is not None:
        provenance["command_provenance"] = command_provenance
    input_policy = getattr(
        importlib.import_module(args.adapter.split(":", 1)[0]), "INPUT_POLICY", None
    )
    if input_policy is not None:
        provenance["adapter_input_policy"] = input_policy
    if args.episode_ids:
        provenance["episode_audit"] = audit_episode_ids(args.episode_ids, args.sdk_root)
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != provenance:
        raise ValueError("Output belongs to a different run; choose a new --output")
    write_json(run_path, provenance)
    module, name = args.adapter.split(":", 1)
    adapter = getattr(importlib.import_module(module), name)(args)
    totals = {
        name: np.zeros(4, dtype=np.float64) for name in ("1s", "2s", "3s", "4s", "5s")
    }
    all_uids, completed_scenes = [], []
    supported_steps = None
    for scene_id in sorted(selected):
        output_path = args.output / f"{scene_id}.npz"
        if not output_path.exists():
            records, predictions, targets = [], [], []
            native_proposals, proposal_scores = [], []
            with raw_scene(args, scene_id) as dataset:
                for sample in scene_samples(dataset, scene_id):
                    if args.command_file and sample["sample_uid"] not in args.commands:
                        raise ValueError(
                            f"Command file is missing {sample['sample_uid']}"
                        )
                    prediction = resample_prediction(
                        adapter.predict(sample), adapter.prediction_times_s
                    )
                    records.append((sample["sample_uid"], sample["timestamp_ns"]))
                    predictions.append(prediction)
                    targets.append(sample["target_xy"])
                    if getattr(adapter, "last_proposals", None) is not None:
                        native_proposals.append(adapter.last_proposals.copy())
                        proposal_scores.append(adapter.last_scores.copy())
                    if len(records) % 25 == 0:
                        print(f"{scene_id}: {len(records)} predictions", flush=True)
            if not records:
                raise ValueError(f"No samples in {scene_id}")
            temporary = output_path.with_suffix(".tmp.npz")
            proposal_artifacts = {}
            if native_proposals:
                if len(native_proposals) != len(records):
                    raise ValueError("Missing proposals for some evaluated samples")
                proposal_artifacts = dict(
                    native_proposals=np.stack(native_proposals),
                    proposal_scores=np.stack(proposal_scores),
                    proposal_times_s=np.asarray(adapter.prediction_times_s),
                )
            np.savez_compressed(
                temporary,
                prediction_xy=np.stack(predictions),
                target_xy=np.stack(targets),
                sample_uid=np.array([r[0] for r in records]),
                timestamp_ns=np.array([r[1] for r in records], dtype=np.int64),
                **proposal_artifacts,
            )
            temporary.replace(output_path)
        with np.load(output_path, allow_pickle=False) as data:
            predictions, targets = data["prediction_xy"], data["target_xy"]
            uids = data["sample_uid"].tolist()
        if supported_steps is None:
            supported_steps = predictions.shape[1]
        if predictions.shape[1] != supported_steps:
            raise ValueError("Prediction support differs between scenes")
        for name, values in score_predictions(predictions, targets).items():
            totals[name] += values
        all_uids.extend(uids)
        completed_scenes.append(scene_id)
        trajectory = {}
        for name, (ade_sum, ade_count, fde_sum, fde_count) in totals.items():
            trajectory.update(
                {
                    f"ade_{name}_m": float(ade_sum / ade_count) if ade_count else None,
                    f"fde_{name}_m": float(fde_sum / fde_count) if fde_count else None,
                    f"ade_{name}_sample_count": int(ade_count),
                    f"fde_{name}_sample_count": int(fde_count),
                }
            )
        report = {
            **provenance,
            "source_split": "val",
            "sample_count": len(all_uids),
            "sample_uid_sha256": digest_ids(all_uids),
            "published_sample_uid_sha256": published_digest_ids(all_uids),
            "prediction_support_seconds": supported_steps / 10,
            "scene_uid_sha256": digest_ids(
                [f"kitscenes-{uid}" for uid in completed_scenes]
            ),
            "completed_scene_count": len(completed_scenes),
            "expected_scene_count": 117,
            "expected_sample_count": EXPECTED_VAL_SAMPLES,
            "population_matches_count": len(all_uids) == EXPECTED_VAL_SAMPLES
            and set(selected) == set(official),
            "exact_sample_identity_verified": False,
            "evaluation_role": "external_val_subset"
            if args.scene
            else "external_val_in_progress",
            "metrics": {"trajectory": trajectory},
        }
        if report["completed_scene_count"] == len(selected):
            report["evaluation_role"] = (
                "external_val_subset" if args.scene else "external_val_complete"
            )
            if not args.scene and len(all_uids) != EXPECTED_VAL_SAMPLES:
                write_json(args.output / "report.json", report)
                raise ValueError(
                    f"Full val expected 11035 samples, got {len(all_uids)}"
                )
            if not args.scene:
                report["exact_sample_identity_verified"] = (
                    published_digest_ids(all_uids) == EXPECTED_VAL_SAMPLE_UID_SHA256
                )
                if not report["exact_sample_identity_verified"]:
                    write_json(args.output / "report.json", report)
                    raise ValueError(
                        "Full-val sample UID digest differs from published AutoE2E v3.5"
                    )
            if args.reference_report and not args.scene:
                reference = json.loads(args.reference_report.read_text())
                identity = reference.get(
                    "target_identity", reference.get("metrics", {})
                )
                actual_digest = (
                    published_digest_ids(all_uids)
                    if "target_identity" in reference
                    else digest_ids(all_uids)
                )
                if (
                    reference.get("source_split") != "val"
                    or identity.get("sample_uid_sha256") != actual_digest
                ):
                    raise ValueError(
                        "Sample UID digest differs from AutoE2E reference report"
                    )
                report["exact_sample_identity_verified"] = True
        write_json(args.output / "report.json", report)
        print(
            f"Completed {scene_id}: {len(uids)} samples; accumulated {len(all_uids)}",
            flush=True,
        )
    if len(set(all_uids)) != len(all_uids):
        raise ValueError("Duplicate sample UIDs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-root", type=Path, default=Path(".benchmark/vendor/kitscenes")
    )
    parser.add_argument("--episode-ids", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--drive-jepa-root", type=Path, default=Path("../Drive-JEPA"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".benchmark/weights/drive_jepa_perception_free_agent_vitl.ckpt"),
    )
    parser.add_argument(
        "--adapter",
        default="accuracy_benchmark.drive_jepa.drive_jepa_adapter:create_adapter",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--raw-root", type=Path)
    sources.add_argument(
        "--stream",
        action="store_true",
        help="Download and discard one temporary raw scene at a time",
    )
    parser.add_argument("--staging-root", type=Path, default=Path(".benchmark/staging"))
    parser.add_argument("--dataset-revision", default=KITSCENES_DATA_REVISION)
    parser.add_argument(
        "--output", type=Path, default=Path(".benchmark/results/drive_jepa_val")
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="Official val UUID; repeat for an explicitly labeled subset",
    )
    parser.add_argument(
        "--camera", choices=POPULATION_CAMERAS, default="camera_base_front_center"
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--command",
        type=float,
        nargs=4,
        default=[0.0, 0.0, 0.0, 0.0],
        help="Declared fallback command; zero vector denotes missing command",
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        help="JSON sample_uid -> four-dimensional NAVSIM command, supplied from route inputs",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="AutoE2E val report for exact sample UID digest verification",
    )
    args = parser.parse_args()
    if args.audit_only:
        if not args.episode_ids:
            parser.error("--audit-only requires --episode-ids")
        print(json.dumps(audit_episode_ids(args.episode_ids, args.sdk_root), indent=2))
    else:
        print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
