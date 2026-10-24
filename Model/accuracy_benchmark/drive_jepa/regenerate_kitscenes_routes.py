"""Stream pinned raw val scenes, verify reconstructed routes, derive commands."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation

from accuracy_benchmark.drive_jepa.external_kitscenes import (
    check_sdk,
    raw_scene,
    contiguous_length,
    POPULATION_CAMERAS,
    published_digest_ids,
    EXPECTED_VAL_SAMPLE_UID_SHA256,
    write_json,
    file_sha256,
)
from accuracy_benchmark.drive_jepa.kitscenes_route_20m import command_from_route
from accuracy_benchmark.drive_jepa.source import (
    KITSCENES_DATA_REVISION,
    sdk_split_scene_ids,
)
from navigation.artifacts import decode_scene_navigation


def prepare(args):
    repo = Path(__file__).resolve().parents[3]
    args.sdk_root = args.sdk_root.resolve()
    check_sdk(args.sdk_root)
    args.staging_root = args.staging_root.resolve()
    args.dataset_revision = KITSCENES_DATA_REVISION
    official = set(sdk_split_scene_ids("val"))
    baseline = args.population.resolve()
    scenes = sorted(p.stem for p in baseline.glob("*.npz"))
    if set(scenes) != official or len(scenes) != 117:
        raise ValueError("Baseline scenes differ from pinned official val")
    if args.scene:
        if args.scene not in official:
            raise ValueError("Requested scene is outside official val")
        scenes = [args.scene]
    provenance = json.loads(
        (args.route_inputs / "sample_route_provenance.json").read_text()
    )
    args.output.mkdir(parents=True, exist_ok=True)

    def linux_path(path):
        path = path.resolve()
        if path.is_relative_to(repo):
            return "/workspace/" + path.relative_to(repo).as_posix()
        if args.raw_root and path.is_relative_to(args.raw_root.resolve()):
            return "/raw/" + path.relative_to(args.raw_root.resolve()).as_posix()
        raise ValueError(
            "Generated route artifacts must be inside the repository workspace"
        )

    all_commands, all_details, hashes = {}, {}, {}
    for scene in scenes:
        result = args.output / scene / "commands.json"
        nav_path = args.output / scene / "scene_navigation.json"
        with np.load(baseline / f"{scene}.npz", allow_pickle=False) as base:
            uids = base["sample_uid"].tolist()
            timestamps = base["timestamp_ns"].tolist()
        expected_hashes = {provenance[u]["scene_navigation_sha256"] for u in uids}
        if len(expected_hashes) != 1:
            raise ValueError("Published samples disagree on scene route hash")
        expected_hash = expected_hashes.pop()
        if not result.exists():
            metadata = {}
            mirror = args.route_inputs / "mirror/datasets/kitscenes-val/v3.5/shards"
            for shard in sorted(mirror.glob("*.tar")):
                with tarfile.open(shard, "r:") as archive:
                    for member in archive:
                        if f"-{scene}-f" not in member.name or not member.name.endswith(
                            ".navigation_meta.json"
                        ):
                            continue
                        uid = member.name[: -len(".navigation_meta.json")]
                        payload = archive.extractfile(member).read()
                        if (
                            uid not in provenance
                            or hashlib.sha256(payload).hexdigest()
                            != provenance[uid]["metadata_sha256"]
                        ):
                            raise ValueError("Route metadata provenance differs")
                        metadata[uid] = json.loads(payload)
            if set(metadata) != set(uids):
                raise ValueError("Published scene metadata population differs")
            for uid, timestamp in zip(uids, timestamps):
                if metadata[uid]["sample_timestamp_ns"] != timestamp:
                    raise ValueError("Published sample timestamps differ")
            nav_path.parent.mkdir(parents=True, exist_ok=True)
            published_metadata = nav_path.parent / "published_navigation_meta.json"
            write_json(published_metadata, metadata[uids[0]])
            if nav_path.is_file():
                if file_sha256(nav_path) != expected_hash:
                    raise ValueError("Retained scene route differs from published hash")
            else:
                with raw_scene(args, scene) as dataset:
                    loader = dataset.get_sensor_loader(scene)
                    poses = dataset.get_scene(scene).ego_poses
                    usable = min(
                        len(poses),
                        len(loader.get_reference_timestamps()),
                        *[
                            contiguous_length(loader.get_frame_indices(c))
                            for c in POPULATION_CAMERAS
                        ],
                    )
                    frames = list(range(40, usable - 50))
                    if [f"kitscenes-v1-{scene}-f{f:06d}" for f in frames] != uids:
                        raise ValueError("Raw scene enumeration differs from baseline")
                    poses = poses[:usable]
                    trace_path = nav_path.parent / "trace.npz"
                    positions = np.array(
                        [p.translation[:2] for p in poses], dtype=np.float64
                    )
                    yaws = Rotation.from_quat([p.rotation for p in poses]).as_euler(
                        "ZYX"
                    )[:, 0]
                    trace_timestamps = np.array(
                        [p.timestamp_ns for p in poses], dtype=np.int64
                    )
                    np.savez_compressed(
                        nav_path.parent / "trace.raw.npz",
                        positions=positions,
                        yaws=yaws,
                        timestamps=trace_timestamps,
                    )
                    changes = {}
                    for uid, frame in zip(uids, frames):
                        published_pose = metadata[uid]["sample_pose_enu"]
                        published_xy = np.array(
                            [published_pose["x_enu_m"], published_pose["y_enu_m"]]
                        )
                        published_yaw = published_pose["yaw_rad"]
                        if (
                            not np.allclose(
                                positions[frame], published_xy, rtol=0, atol=1e-12
                            )
                            or not np.isclose(
                                yaws[frame], published_yaw, rtol=0, atol=1e-12
                            )
                            or trace_timestamps[frame] != published_pose["timestamp_ns"]
                        ):
                            raise ValueError(
                                "Published current pose differs materially from raw SDK pose"
                            )
                        if (
                            not np.array_equal(positions[frame], published_xy)
                            or yaws[frame] != published_yaw
                        ):
                            changes[uid] = {
                                "raw_xy": positions[frame].tolist(),
                                "published_xy": published_xy.tolist(),
                                "raw_yaw": float(yaws[frame]),
                                "published_yaw": published_yaw,
                            }
                        positions[frame] = published_xy
                        yaws[frame] = published_yaw
                    write_json(
                        nav_path.parent / "trace_serialization_reconciliation.json",
                        changes,
                    )
                    np.savez_compressed(
                        trace_path,
                        positions=positions,
                        yaws=yaws,
                        timestamps=trace_timestamps,
                    )
                    worker_args = [
                        "docker",
                        "run",
                        "--rm",
                        "--mount",
                        f"type=bind,source={repo},target=/workspace",
                        *(
                            [
                                "--mount",
                                f"type=bind,source={args.raw_root.resolve()},target=/raw,readonly",
                            ]
                            if args.raw_root
                            else []
                        ),
                        args.docker_image,
                        "python",
                        "Model/accuracy_benchmark/drive_jepa/regenerate_navigation.py",
                        "--scene-id",
                        scene,
                        "--scene-path",
                        linux_path(loader.scene_path),
                        "--trace-path",
                        linux_path(trace_path),
                        "--output",
                        linux_path(nav_path),
                        "--source-revision",
                        KITSCENES_DATA_REVISION,
                        "--expected-sha256",
                        expected_hash,
                        "--published-metadata",
                        linux_path(published_metadata),
                    ]
                    try:
                        subprocess.run(worker_args, check=True)
                    except subprocess.CalledProcessError:
                        debug_source = nav_path.parent / "sdk_debug_source"
                        debug_source.mkdir(exist_ok=True)
                        shutil.copytree(
                            loader.scene_path / "maps",
                            debug_source / "maps",
                            dirs_exist_ok=True,
                        )
                        shutil.copy2(
                            loader.scene_path / "poses.txt", debug_source / "poses.txt"
                        )
                        raise
            _, route = decode_scene_navigation(nav_path.read_bytes())
            commands, details = {}, {}
            for uid in uids:
                commands[uid], details[uid] = command_from_route(route, metadata[uid])
            write_json(nav_path.parent / "command_details.json", details)
            write_json(result, commands)
        if file_sha256(nav_path) != expected_hash:
            raise ValueError(
                "Saved reconstructed route hash differs from published artifact"
            )
        commands = json.loads(result.read_text())
        if set(commands) != set(uids):
            raise ValueError("Saved scene command identity differs")
        all_commands.update(commands)
        all_details.update(
            json.loads((nav_path.parent / "command_details.json").read_text())
        )
        hashes[scene] = expected_hash
        print(
            f"Verified route commands {len(hashes)}/{len(scenes)} scenes, {len(all_commands)} samples",
            flush=True,
        )
        write_json(
            args.output / "progress.json",
            {
                "scene_count": len(hashes),
                "sample_count": len(all_commands),
                "scene_hashes": hashes,
            },
        )
    if not args.scene and (
        len(all_commands) != 11035
        or published_digest_ids(all_commands) != EXPECTED_VAL_SAMPLE_UID_SHA256
    ):
        raise ValueError("Final 20 m command population differs from published val")
    write_json(args.output / "commands.json", all_commands)
    write_json(
        args.output / "commands.provenance.json",
        {
            "command_policy": "route_centerline_20m_lateral_2m_v1",
            "command_order": ["left", "straight", "right", "unknown"],
            "commands_sha256": file_sha256(args.output / "commands.json"),
            "scene_count": len(hashes),
            "sample_count": len(all_commands),
            "published_sample_uid_sha256": published_digest_ids(all_commands),
            "scene_navigation_sha256": hashes,
            "all_scene_navigation_hashes_match_v35": True,
            "command_counts": dict(
                Counter(
                    ["left", "straight", "right", "unknown"][v.index(1.0)]
                    for v in all_commands.values()
                )
            ),
            "reason_counts": dict(Counter(v["reason"] for v in all_details.values())),
            "ego_reference": "KITScenes recorded ego-pose origin; no vehicle-specific rear-axle offset applied",
            "route_provenance": "v3.5 offline whole-scene map-matched trace reconstruction, verified by exact published hashes",
            "uses_future_ground_truth_for_command_extraction": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument(
        "--output", type=Path, default=Path(".benchmark/route_inputs_20m")
    )
    parser.add_argument(
        "--sdk-root", type=Path, default=Path(".benchmark/vendor/kitscenes")
    )
    parser.add_argument(
        "--population", type=Path, default=Path(".benchmark/population")
    )
    parser.add_argument(
        "--route-inputs", type=Path, default=Path(".benchmark/route_inputs")
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--staging-root", type=Path, default=Path(".benchmark/staging"))
    parser.add_argument("--docker-image", default="autoe2e-kitscenes-navigation:local")
    prepare(parser.parse_args())
