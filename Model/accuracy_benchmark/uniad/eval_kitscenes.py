# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged official UniAD model on prepared original-image inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from .common import (
    ASSETS,
    CONFIG_SHA256,
    UNIAD_COMMIT,
    adapter_hashes,
    coverage,
    read_json,
    safe_child,
    sha256,
    write_json,
)
from .inputs import input_kwargs, reset_scene
from .protocol import to_kit_xy


def now():
    return datetime.now(timezone.utc).isoformat()


def verified_scene(prepared, record):
    path = safe_child(prepared, record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError("Prepared scene manifest changed")
    scene = read_json(path)
    if scene["scene_id"] != record["scene_id"]:
        raise ValueError("Scene identity mismatch")
    for name, digest in scene["metadata_sha256"].items():
        if sha256(safe_child(path.parent / "source_metadata", name)) != digest:
            raise ValueError("Prepared source metadata changed")
    return scene, path.parent


def verify_model_assets(repo, assets):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UNIAD_COMMIT:
        raise ValueError("UniAD revision differs from evaluated model")
    # Ignore filesystem mode bits on Windows; any tracked source edit is rejected.
    changed = subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.filemode=false", "diff", "HEAD", "--"],
        text=True,
    )
    if changed:
        raise ValueError("UniAD tracked sources are modified")
    config = repo / "projects/configs/stage2_e2e/base_e2e.py"
    if sha256(config) != CONFIG_SHA256:
        raise ValueError("Official config checksum mismatch")
    for name, expected in ASSETS.items():
        if sha256(assets / name) != expected:
            raise ValueError("Official model asset checksum mismatch: " + name)
    return config


def run_worker(args):
    """Fresh process/model/RNG for each physical scene and cohort, as evaluated."""
    import torch

    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="started",
        started_utc=now(),
        scene=args.scene,
        track=args.track,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        primary_command="fixed_forward_2",
        adapter_source_sha256=adapter_hashes(),
        oracle_diagnostic=args.oracle_diagnostic,
        horizons_seconds=[1, 2, 3],
        horizons_4_5="N/A",
        collision_metric="N/A",
    )
    hook = None
    planning_class = original_optimizer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "UniAD inference requires a CUDA GPU and its separate validated environment"
            )
        manifest = read_json(args.prepared / "manifest.json")
        if manifest["status"] != "ready":
            raise ValueError("Preparation is not complete")
        (record,) = [s for s in manifest["scenes"] if s["scene_id"] == args.scene]
        scene, _ = verified_scene(args.prepared, record)
        streams = scene["tracks"][args.track]
        config_path = verify_model_assets(args.uniad_dir, args.assets_dir)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        report.update(
            gpu=torch.cuda.get_device_name(0),
            input_sha256=record["sha256"],
            source_revision=UNIAD_COMMIT,
            config_sha256=CONFIG_SHA256,
            asset_sha256=ASSETS,
        )
        os.chdir(args.uniad_dir)
        sys.path.insert(0, str(args.uniad_dir))
        import mmcv
        import mmdet
        import mmdet3d
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        importlib.import_module("projects.mmdet3d_plugin")
        from projects.mmdet3d_plugin.uniad.dense_heads.planning_head import (
            PlanningHeadSingleMode,
        )

        report["versions"] = dict(
            mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmdet3d=mmdet3d.__version__
        )
        cfg = Config.fromfile(str(config_path))
        cfg.model.motion_head.anchor_info_path = str(
            args.assets_dir / "data/motion_anchor_infos_mode6.pkl"
        )
        if (
            not cfg.model.planning_head.use_col_optim
            or cfg.model.planning_head.planning_steps != 6
        ):
            raise ValueError("Official six-step collision-optimized planner required")
        cfg.dump(str(args.out / "resolved_config.py"))
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
        checkpoint = load_checkpoint(
            model,
            str(args.assets_dir / "ckpts/uniad_base_e2e.pth"),
            map_location="cpu",
            strict=True,
        )
        del checkpoint
        report["checkpoint_loaded_strict"] = True
        model.cuda().eval()
        captured = []

        def capture_planner(module, positional):
            captured[:] = positional

        hook = model.planning_head.register_forward_pre_hook(capture_planner)
        planning_class = PlanningHeadSingleMode
        original_optimizer = planning_class.collision_optimization
        context = {}
        count, model_frames = 0, 0
        with (
            (args.out / "samples.jsonl").open("x", encoding="utf-8") as output,
            (args.out / "planner_observations.jsonl").open(
                "x", encoding="utf-8"
            ) as observations,
        ):

            def observed_optimizer(head, trajectory, occupancy):
                context["call_in_frame"] += 1
                before = trajectory.detach().cpu().numpy().copy()
                result = original_optimizer(head, trajectory, occupancy)
                if context["scored"]:
                    after = result.detach().cpu().numpy().copy()
                    mask = occupancy.detach().cpu().numpy()
                    row = dict(
                        context,
                        before_body_xy=to_kit_xy(before[0]).tolist(),
                        after_body_xy=to_kit_xy(after[0]).tolist(),
                        occupancy_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                        occupied_cells=int(np.count_nonzero(mask)),
                        optimizer_max_change_m=float(np.max(np.abs(after - before))),
                    )
                    observations.write(json.dumps(row, allow_nan=False) + "\n")
                    observations.flush()
                return result

            planning_class.collision_optimization = observed_optimizer
            for stream in streams:
                for position, frame in enumerate(stream["frames"]):
                    context.update(
                        scene_id=stream["scene_id"],
                        frame=frame["frame"],
                        sample_uid=frame.get("sample_uid"),
                        scored=frame["scored"],
                        call_in_frame=0,
                    )
                    kwargs = input_kwargs(
                        frame, stream["scene_id"], LiDARInstance3DBoxes
                    )
                    if position == 0:
                        reset_scene(model, kwargs["img_metas"][0][0])
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    with torch.no_grad():
                        result = model(return_loss=False, rescale=True, **kwargs)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - started
                    model_frames += 1
                    native = to_kit_xy(
                        result[0]["planning"]["result_planning"]["sdc_traj"][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if frame["scored"]:
                        predictions = dict(fixed_forward=native.tolist())
                        if args.oracle_diagnostic and args.track == "dense":
                            oracle = native.copy()
                            if frame["oracle_command"] != 2:
                                if len(captured) != 6:
                                    raise RuntimeError(
                                        "Missing planner diagnostic inputs"
                                    )
                                planner_inputs = tuple(captured[:5]) + (
                                    torch.tensor(
                                        [frame["oracle_command"]], device="cuda"
                                    ),
                                )
                                with torch.no_grad():
                                    alternative = model.planning_head(*planner_inputs)
                                oracle = to_kit_xy(
                                    alternative["sdc_traj"][0].detach().cpu().numpy()
                                )
                                del alternative, planner_inputs
                            predictions["oracle_turn"] = oracle.tolist()
                        row = dict(
                            scene_id=stream["scene_id"],
                            physical_scene_id=args.scene,
                            frame=frame["frame"],
                            timestamp=frame["timestamp"],
                            sample_uid=frame["sample_uid"],
                            fixed_command=2,
                            prediction_frame="body_FLU",
                            inference_seconds=elapsed,
                            predictions_body_xy_2hz=predictions,
                        )
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        output.flush()
                        count += 1
                    captured.clear()
                    del result, kwargs
            expected = coverage(streams)
            if count != expected["samples"] or model_frames != expected["model_frames"]:
                raise ValueError("Incomplete scene inference")
            report.update(
                status="predictions_complete",
                coverage=expected,
                model_frames=model_frames,
                peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
                samples_sha256=sha256(args.out / "samples.jsonl"),
            )
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if hook is not None:
            hook.remove()
        if planning_class is not None and original_optimizer is not None:
            planning_class.collision_optimization = original_optimizer
        report.update(ended_utc=now(), wall_seconds=time.monotonic() - start)
        write_json(args.out / "run.json", report)


def run(args):
    manifest = read_json(args.prepared / "manifest.json")
    if manifest["status"] != "ready":
        raise ValueError("Requires completed preparation")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = dict(
        status="running",
        adapter_source_sha256=adapter_hashes(),
        started_utc=now(),
        prepared_manifest_sha256=sha256(args.prepared / "manifest.json"),
        tracks=[args.track] if args.track != "both" else ["dense", "keyframes"],
        full_evaluation=manifest["full_evaluation"],
        completed=[],
        oracle_diagnostic=args.oracle_diagnostic,
    )
    write_json(args.out / "status.json", status)
    try:
        # Launching fresh workers preserves the evaluated per-scene seed/loading boundary.
        for record in manifest["scenes"]:
            verified_scene(args.prepared, record)
            for track in status["tracks"]:
                destination = args.out / track / record["scene_id"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "Model.accuracy_benchmark.uniad.eval_kitscenes",
                    "--worker",
                    "--prepared",
                    str(args.prepared),
                    "--out",
                    str(destination),
                    "--uniad-dir",
                    str(args.uniad_dir),
                    "--assets-dir",
                    str(args.assets_dir),
                    "--track",
                    track,
                    "--scene",
                    record["scene_id"],
                ]
                if args.oracle_diagnostic:
                    command.append("--oracle-diagnostic")
                with (
                    destination.with_suffix(".out").open("x") as out,
                    destination.with_suffix(".err").open("x") as err,
                ):
                    subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[3],
                        stdout=out,
                        stderr=err,
                        check=True,
                    )
                status["completed"].append(dict(scene=record["scene_id"], track=track))
                write_json(args.out / "status.json", status)
                print(json.dumps(status["completed"][-1]), flush=True)
        from .audit_results import audit

        audit(
            args.prepared,
            args.out,
            args.assets_dir / "reactive_stage_runner.py",
            args.out / "summary.json",
        )
        status["status"] = "complete_scientifically_provisional"
    except Exception:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status.update(ended_utc=now(), wall_seconds=time.monotonic() - start)
        write_json(args.out / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--uniad-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory; failures are retained",
    )
    parser.add_argument(
        "--track", choices=("both", "dense", "keyframes"), default="both"
    )
    parser.add_argument(
        "--oracle-diagnostic",
        action="store_true",
        help="Also rescore dense planner with future-derived turn commands; never primary",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scene", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("prepared", "uniad_dir", "assets_dir", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        if args.track == "both" or not args.scene:
            parser.error("Workers require one scene and track")
        run_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
