"""SIMFORGE-D1.5 on the KITScenes val population (issue #210).

Commands (run from the repository root with ``PYTHONPATH=Model``):

    setup       fetch the pinned Alpamayo reference code, the archive manifest and the model snapshot
    population  derive and verify the 202 keyframes and their horizon counts (no network)
    index       build the cached sparse archive indices of the selected scenes (parallel)
    run         predict, persist, then build ground truth for every anchor (resumable)
    score       ADE/FDE per anchor, aggregate table and optional GT-vs-candidates plots

Per anchor the order is: past-only input -> prediction saved -> ground truth built.
The ground truth never reaches the model.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = REPOSITORY_ROOT / ".benchmark" / "simforge"
logger = logging.getLogger("simforge_kitscenes")

from accuracy_benchmark.simforge.contracts import (  # noqa: E402
    ALPAMAYO_REPO_URL,
    ALPAMAYO_REVISION,
    CAMERA_CONFIGS,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_HORIZON_COUNTS,
    KITSCENES_DATA_REVISION,
    KITSCENES_REPO_ID,
    POPULATION_ID,
    SCORING_HORIZONS_S,
    SIMFORGE_MODEL_REPO_ID,
    SIMFORGE_MODEL_REVISION,
    CameraConfig,
    camera_config,
    model_contract,
)
from accuracy_benchmark.simforge.samples import (  # noqa: E402
    Anchor,
    PopulationMismatchError,
    build_ground_truth,
    build_input_sample,
    derive_anchors,
    load_population,
    select_anchors,
    validate_evaluation_anchor,
    verify_population,
)
from accuracy_benchmark.simforge.scoring import plot_anchor, sample_metrics, write_report  # noqa: E402

RUN_MANIFEST = "run_manifest.json"
IMMUTABLE_KEYS = ("population_id", "population_uid_sha256", "camera_config", "model", "dataset", "alpamayo_revision", "run_id")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    """Write ``arrays`` atomically and return the SHA-256 of the bytes as persisted on disk."""
    import io

    buffer = io.BytesIO()
    savez: Any = np.savez  # keyword arrays; numpy's stub types the keyword space too narrowly
    savez(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, path)
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {"repository_head": head}


def _alpamayo_source(workspace: Path, explicit: str | None) -> Path:
    return Path(explicit) if explicit else workspace / "alpamayo1.5" / "src"


def _anchors_from_args(args: argparse.Namespace) -> list[Anchor]:
    anchors = _population_anchors()
    selected = select_anchors(anchors, scene_ids=args.scene or (), limit=args.limit)
    if args.sample_uid:
        wanted = set(args.sample_uid)
        unknown = wanted - {a.uid for a in anchors}
        if unknown:
            raise SystemExit(f"not population anchors: {sorted(unknown)}")
        selected = [a for a in selected if a.uid in wanted]
    if not selected:
        raise SystemExit("no anchors selected")
    return selected


def _lock(output: Path) -> TextIO:
    output.mkdir(parents=True, exist_ok=True)
    handle = open(output / ".lock", "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise SystemExit(f"{output} is in use by another run") from error
    return handle


def _output_dir(args: argparse.Namespace, config: CameraConfig) -> Path:
    return Path(args.output) if args.output else DEFAULT_WORKSPACE / "results" / config.name


def _population_anchors() -> list[Anchor]:
    bundled = load_population()
    anchors = derive_anchors(bundled["scene_usable_frames"])
    verify_population(anchors, bundled)
    return anchors


def _run_identity(config: CameraConfig) -> dict[str, Any]:
    """Fields that define a run. ``run_id`` is their digest and is stamped on every stored record."""
    population_uids = sorted(anchor.uid for anchor in _population_anchors())
    identity = {
        "population_id": POPULATION_ID,
        "population_uid_sha256": hashlib.sha256("\n".join(population_uids).encode()).hexdigest(),
        "camera_config": config.to_metadata(),
        "model": model_contract(),
        "dataset": {"repo_id": KITSCENES_REPO_ID, "revision": KITSCENES_DATA_REVISION},
        "alpamayo_revision": ALPAMAYO_REVISION,
    }
    identity["run_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def _run_manifest(config: CameraConfig) -> dict[str, Any]:
    return {
        **_run_identity(config),
        "configuration_label": config.run_label,
        "label_note": "the report is labelled as the full benchmark only when all 202 population anchors are scored",
        "future_gt_access": "post_prediction_only",
        "code": _code_revision(),
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    from data_parsing.kit_scenes.source import fetch_archive_manifest

    from accuracy_benchmark.simforge.kitscenes_data import resolve_hf_token
    from accuracy_benchmark.simforge.model import download_snapshot, verify_snapshot

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    checkout = workspace / "alpamayo1.5"
    if not checkout.exists():
        # Fetch exactly the pinned commit (not a moving branch) into an empty repository.
        checkout.mkdir(parents=True)
        subprocess.run(["git", "-C", str(checkout), "init", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", ALPAMAYO_REPO_URL], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", ALPAMAYO_REVISION], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != ALPAMAYO_REVISION:
        raise SystemExit(f"Alpamayo checkout is at {head}, expected {ALPAMAYO_REVISION}")
    logger.info("Alpamayo reference code at %s", ALPAMAYO_REVISION)
    token = resolve_hf_token()
    manifest_dir = workspace / "kitscenes"
    fetch_archive_manifest(manifest_dir, repo_id=KITSCENES_REPO_ID, revision=KITSCENES_DATA_REVISION, token=token)
    logger.info("KITScenes archive manifest at revision %s", KITSCENES_DATA_REVISION)
    snapshot = Path(args.model_snapshot_dir) if args.model_snapshot_dir else download_snapshot(args.model_cache_dir, token)
    identity = verify_snapshot(snapshot)
    logger.info("model %s@%s verified: %s", SIMFORGE_MODEL_REPO_ID, SIMFORGE_MODEL_REVISION, identity["snapshot_dir"])
    print(f"setup complete: {checkout}, {manifest_dir}, {snapshot}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build and cache the sparse TAR index of every selected scene, several scenes at a time."""
    from concurrent.futures import ThreadPoolExecutor

    from data_parsing.kit_scenes.source import parse_archive_manifest

    from accuracy_benchmark.simforge.kitscenes_data import KitSceneStore, resolve_hf_token

    workspace = Path(args.workspace)
    manifest_path = Path(args.manifest) if args.manifest else workspace / "kitscenes" / "data" / "sequence_archives.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"archive manifest {manifest_path} is missing; run the setup command")
    scenes = sorted({a.scene_id for a in _anchors_from_args(args)})
    store = KitSceneStore(
        archives=parse_archive_manifest(manifest_path),
        cache_root=workspace / "cache",
        repo_id=KITSCENES_REPO_ID,
        revision=KITSCENES_DATA_REVISION,
        token=resolve_hf_token(),
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for count, scene_id in enumerate(pool.map(lambda sid: store.open(sid).scene_id, scenes), start=1):
            logger.info("[%d/%d] index ready for %s", count, len(scenes), scene_id)
    print(f"indexed {len(scenes)} scenes into {workspace / 'cache'}")
    return 0


def cmd_population(args: argparse.Namespace) -> int:
    bundled = load_population()
    anchors = derive_anchors(bundled["scene_usable_frames"])
    report = verify_population(anchors, bundled)
    payload = {
        "population_id": POPULATION_ID,
        "rule": "frames 40 + 90k per val scene with >= 30 genuine future frames (issue #210)",
        "scene_count": len(bundled["scene_usable_frames"]),
        **report,
        "expected_anchor_count": EXPECTED_ANCHOR_COUNT,
        "expected_horizon_counts": EXPECTED_HORIZON_COUNTS,
        "anchors": [
            {"sample_uid": a.uid, "scene_id": a.scene_id, "frame_index": a.frame_index, "available_future_steps": a.available_future_steps}
            for a in anchors
        ],
    }
    if args.output:
        _write_json(Path(args.output), payload)
    print(json.dumps({k: payload[k] for k in ("anchor_count", "horizon_counts", "scene_count", "matches_bundled_list")}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:

    from accuracy_benchmark.simforge.kitscenes_data import KitSceneStore, resolve_hf_token
    from accuracy_benchmark.simforge.model import SimForgeRunner
    from data_parsing.kit_scenes.source import parse_archive_manifest

    config = camera_config(args.camera_config)
    workspace = Path(args.workspace)
    output = _output_dir(args, config)
    lock = _lock(output)
    anchors = _anchors_from_args(args)
    manifest_path = Path(args.manifest) if args.manifest else workspace / "kitscenes" / "data" / "sequence_archives.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"archive manifest {manifest_path} is missing; run the setup command")
    expected = _run_manifest(config)
    run_manifest_path = output / RUN_MANIFEST
    if run_manifest_path.is_file():
        _check_manifest(json.loads(run_manifest_path.read_text()), expected, output)
    else:
        _write_json(run_manifest_path, expected)
    run_id = expected["run_id"]

    def is_done(anchor: Anchor) -> bool:
        meta = output / "predictions" / f"{anchor.uid}.json"
        if not meta.is_file():
            return False
        stored = json.loads(meta.read_text())
        return (
            stored.get("run_id") == run_id
            and (output / "predictions" / f"{anchor.uid}.npz").is_file()
            and _sha256_file(output / "predictions" / f"{anchor.uid}.npz") == stored["prediction_npz_sha256"]
        )

    pending = [a for a in anchors if not is_done(a)]
    logger.info("%d anchors selected, %d already predicted, %d pending", len(anchors), len(anchors) - len(pending), len(pending))
    if pending:
        token = resolve_hf_token()
        archives = parse_archive_manifest(manifest_path)
        store = KitSceneStore(archives=archives, cache_root=workspace / "cache", repo_id=KITSCENES_REPO_ID, revision=KITSCENES_DATA_REVISION, token=token)
        snapshot = Path(args.model_snapshot_dir) if args.model_snapshot_dir else _resolve_snapshot(args, token)
        runner = SimForgeRunner(snapshot=snapshot, alpamayo_src=_alpamayo_source(workspace, args.alpamayo_src), device=args.device)
        started = time.perf_counter()
        runner.load()
        logger.info("model loaded in %.1fs", time.perf_counter() - started)
        _write_json(output / "model_provenance.json", runner.provenance)
        scenes: dict[str, Any] = {}
        for position, anchor in enumerate(pending, start=1):
            t_anchor = time.perf_counter()
            if anchor.scene_id not in scenes:
                scenes[anchor.scene_id] = store.open(anchor.scene_id)
            scene = scenes[anchor.scene_id]
            # Only the scene/frame identity reaches the input builder; the population's
            # future metadata stays with ``anchor`` for the evaluation step below.
            sample = build_input_sample(scene, anchor.for_inference(), config)
            prediction = runner.predict(sample)
            prediction_sha = _save_npz(
                output / "predictions" / f"{anchor.uid}.npz",
                pred_xyz=prediction["pred_xyz"],
                pred_rot=prediction["pred_rot"],
                ego_history_xyz=sample.ego_xyz,
                ego_history_rot=sample.ego_rot,
            )
            _write_json(
                output / "predictions" / f"{anchor.uid}.json",
                {
                    "sample_uid": anchor.uid,
                    "scene_id": anchor.scene_id,
                    "frame_index": anchor.frame_index,
                    "t0_ns": sample.t0_ns,
                    "run_id": run_id,
                    "camera_config": config.name,
                    "input": sample.provenance,
                    "image_records": [frame.record() for slot in sample.images for frame in slot],
                    "prompt": {
                        "image_count": prediction["image_count"],
                        "input_token_count": prediction["input_token_count"],
                        "image_grid_thw": prediction["image_grid_thw"],
                        "processor_image_hw_per_image": prediction["processor_image_hw_per_image"],
                        "raw_image_hw_per_slot": prediction["raw_image_hw_per_slot"],
                    },
                    "generated_text": prediction["generated_text"],
                    "timing": {"generation_s": prediction["elapsed_s"], "anchor_total_s": round(time.perf_counter() - t_anchor, 3)},
                    "gpu_peak_allocated_mib": prediction["gpu_peak_allocated_mib"],
                    "prediction_npz_sha256": prediction_sha,
                },
            )
            # Evaluation starts only now, after the prediction is persisted: the live scene
            # is checked against the fixed population, then the canonical ground truth is built.
            try:
                population_check = validate_evaluation_anchor(scene, anchor, config)
            except PopulationMismatchError as error:
                raise SystemExit(f"population/data mismatch after prediction: {error}") from error
            truth = build_ground_truth(scene, anchor)
            truth_sha = _save_npz(  # NPZ first; its digest goes into the JSON written next
                output / "ground_truth" / f"{anchor.uid}.npz",
                xyz=truth.xyz,
                rot=truth.rot,
                valid=truth.valid,
                timestamps_ns=truth.timestamps_ns,
            )
            _write_json(
                output / "ground_truth" / f"{anchor.uid}.json",
                {
                    "sample_uid": anchor.uid,
                    "run_id": run_id,
                    "npz_sha256": truth_sha,
                    "t0_ns": truth.t0_ns,
                    "population_check": population_check,
                    "available_future_steps": truth.available_steps,
                    "origin": "rear_axle_centre_of_kit_poses (kit-poses, canonical)",
                    "source": "poses.txt at dataset revision " + KITSCENES_DATA_REVISION,
                },
            )
            logger.info(
                "[%d/%d] %s: %d tokens, generation %.1fs, peak %s MiB",
                position,
                len(pending),
                anchor.uid,
                prediction["input_token_count"],
                prediction["elapsed_s"],
                prediction["gpu_peak_allocated_mib"],
            )
    lock.close()
    scope = (
        "all 202 population anchors"
        if len(anchors) == EXPECTED_ANCHOR_COUNT
        else f"{len(anchors)} of {EXPECTED_ANCHOR_COUNT} anchors: PARTIAL / DIAGNOSTIC RUN, NOT THE FULL ISSUE #210 BENCHMARK"
    )
    print(f"predictions in {output / 'predictions'} for {scope} [{config.name}]")
    return 0


def _resolve_snapshot(args: argparse.Namespace, token: str | None) -> Path:
    from accuracy_benchmark.simforge.model import download_snapshot

    return download_snapshot(args.model_cache_dir, token)


def _check_manifest(stored: dict[str, Any], expected: dict[str, Any], output: Path) -> None:
    if stored.get("camera_config", {}).get("name") != expected["camera_config"]["name"]:
        raise SystemExit(
            f"{output} holds a {stored.get('camera_config', {}).get('name')!r} run; refusing to treat it as "
            f"{expected['camera_config']['name']!r}"
        )
    mismatched = [key for key in IMMUTABLE_KEYS if stored.get(key) != expected.get(key)]
    if mismatched:
        raise SystemExit(f"{output} was created for a different run; mismatched {mismatched}")


def _validated_records(output: Path, run_id: str) -> tuple[list[str], list[str]]:
    """UIDs whose prediction (and ground truth) records belong to this run; fail on anything else."""
    population = {anchor.uid for anchor in _population_anchors()}
    problems: list[str] = []
    prediction_json = {p.stem for p in (output / "predictions").glob("*.json")}
    prediction_npz = {p.stem for p in (output / "predictions").glob("*.npz")}
    truth_json = {p.stem for p in (output / "ground_truth").glob("*.json")}
    truth_npz = {p.stem for p in (output / "ground_truth").glob("*.npz")}
    for name, files in (("prediction .npz without .json", prediction_npz - prediction_json),
                        ("prediction .json without .npz", prediction_json - prediction_npz),
                        ("ground truth .npz without .json", truth_npz - truth_json),
                        ("ground truth .json without .npz", truth_json - truth_npz),
                        ("ground truth without a prediction", truth_json - prediction_json),
                        ("not a population anchor", (prediction_json | truth_json) - population)):
        if files:
            problems.append(f"{name}: {sorted(files)[:3]}")
    for uid in sorted(prediction_json & prediction_npz):
        meta = json.loads((output / "predictions" / f"{uid}.json").read_text())
        if meta.get("run_id") != run_id or meta.get("sample_uid") != uid:
            problems.append(f"{uid}: prediction belongs to another run")
        elif _sha256_file(output / "predictions" / f"{uid}.npz") != meta.get("prediction_npz_sha256"):
            problems.append(f"{uid}: prediction file does not match its recorded digest")
    for uid in sorted(truth_json & truth_npz):
        truth_meta = json.loads((output / "ground_truth" / f"{uid}.json").read_text())
        if truth_meta.get("sample_uid") != uid:
            problems.append(f"{uid}: ground truth record names sample_uid {truth_meta.get('sample_uid')!r}")
        elif truth_meta.get("run_id") != run_id:
            problems.append(f"{uid}: ground truth belongs to another run")
        elif _sha256_file(output / "ground_truth" / f"{uid}.npz") != truth_meta.get("npz_sha256"):
            problems.append(f"{uid}: ground truth file does not match its recorded digest")
    if problems:
        raise SystemExit(f"{output}: stored records are inconsistent with the run manifest: " + "; ".join(problems))
    scored = sorted(prediction_json & truth_json)
    unscored = sorted(prediction_json - truth_json)
    return scored, unscored


def cmd_score(args: argparse.Namespace) -> int:
    config = camera_config(args.camera_config)
    output = _output_dir(args, config)
    manifest_path = output / RUN_MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"{output} has no {RUN_MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    _check_manifest(manifest, _run_identity(config), output)
    # From here on the stored manifest is authoritative for what was run.
    stored_config = camera_config(manifest["camera_config"]["name"])
    scored, unscored = _validated_records(output, manifest["run_id"])
    population = sorted(anchor.uid for anchor in _population_anchors())
    complete = scored == population
    if complete:
        label = stored_config.run_label
    else:
        label = (
            f"PARTIAL / DIAGNOSTIC RUN ({len(scored)} of {len(population)} anchors scored): "
            f"NOT THE FULL ISSUE #210 BENCHMARK [{stored_config.name}]"
        )
    rows: list[dict[str, Any]] = []
    for uid in scored:
        meta = json.loads((output / "predictions" / f"{uid}.json").read_text())
        truth_meta = json.loads((output / "ground_truth" / f"{uid}.json").read_text())
        if (meta["sample_uid"], meta["run_id"]) != (truth_meta["sample_uid"], truth_meta["run_id"]):
            raise SystemExit(f"{uid}: prediction and ground truth records are not a pair")
        prediction = np.load(output / "predictions" / f"{uid}.npz")
        truth = np.load(output / "ground_truth" / f"{uid}.npz")
        valid = truth["valid"]
        metrics = sample_metrics(prediction["pred_xyz"][:, :, :2], truth["xyz"][:, :2], valid, SCORING_HORIZONS_S)
        rows.append(
            {
                "sample_uid": uid,
                "scene_id": meta["scene_id"],
                "frame_index": meta["frame_index"],
                "metrics": metrics,
                "_pred": prediction["pred_xyz"],
                "_gt": truth["xyz"],
                "_valid": valid,
            }
        )
    header = {
        "label": label,
        "complete_population": complete,
        "full_benchmark": complete and stored_config.benchmark,
        "scored_anchor_count": len(scored),
        "population_anchor_count": len(population),
        "predicted_without_ground_truth": unscored,
        "run_id": manifest["run_id"],
        "camera_config": manifest["camera_config"],
        "model": manifest["model"],
        "dataset": manifest["dataset"],
        "population_id": manifest["population_id"],
        "ground_truth_track": "canonical (kit-poses)",
    }
    paths = write_report(output / "scores", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], header=header)
    if args.plots:
        for row in rows:
            plot_anchor(
                output / "plots" / f"{row['sample_uid']}.png",
                title=f"{row['sample_uid']}\n{label}",
                pred_xy=row["_pred"][:, :, :2],
                gt_xy=row["_gt"][:, :2],
                gt_valid=row["_valid"],
            )
    print(Path(paths["summary_md"]).read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="clone the pinned Alpamayo code, fetch the manifest and the model snapshot")
    setup.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    setup.add_argument("--model-cache-dir", default=None)
    setup.add_argument("--model-snapshot-dir", default=None, help="use an already downloaded pinned snapshot")
    setup.set_defaults(func=cmd_setup)

    population = subparsers.add_parser("population", help="derive and verify the 202 keyframes (no network)")
    population.add_argument("--output", default=None, help="optional JSON path")
    population.set_defaults(func=cmd_population)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--camera-config", required=True, choices=sorted(CAMERA_CONFIGS))
        sub.add_argument("--output", default=None, help="results directory (default: <workspace>/results/<config>)")
        sub.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
        sub.add_argument("--scene", action="append", default=None, help="restrict to a scene id (repeatable)")
        sub.add_argument("--limit", type=int, default=None, help="first N selected anchors")
        sub.add_argument("--sample-uid", action="append", default=None, help="exact population anchor (repeatable)")

    index = subparsers.add_parser("index", help="build and cache the sparse archive index of the selected scenes in parallel")
    add_common(index)
    index.add_argument("--manifest", default=None, help="archive manifest CSV (default: <workspace>/kitscenes/data/sequence_archives.csv)")
    index.add_argument("--workers", type=int, default=8, help="scenes indexed concurrently")
    index.set_defaults(func=cmd_index)

    run = subparsers.add_parser("run", help="predict, persist and build ground truth (resumable)")
    add_common(run)
    run.add_argument("--manifest", default=None, help="archive manifest CSV (default: <workspace>/kitscenes/data/sequence_archives.csv)")
    run.add_argument("--alpamayo-src", default=None, help="Alpamayo src directory at the pinned revision")
    run.add_argument("--model-snapshot-dir", default=None)
    run.add_argument("--model-cache-dir", default=None)
    run.add_argument("--device", default="cuda")
    run.set_defaults(func=cmd_run)

    score = subparsers.add_parser("score", help="ADE/FDE per anchor, aggregate table and optional plots")
    add_common(score)
    score.add_argument("--plots", action="store_true")
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
