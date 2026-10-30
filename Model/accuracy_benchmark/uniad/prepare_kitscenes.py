# SPDX-License-Identifier: Apache-2.0
"""Build original-image evaluation inputs directly from pinned public releases.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .common import (
    DENSE_UID_SHA256,
    EXPECTED,
    MIRROR_PREFIX,
    MIRROR_REPO,
    MIRROR_REVISION,
    RAW_REPO,
    RAW_REVISION,
    coverage,
    read_json,
    safe_child,
    sha256,
    uid_hash,
    verified_download,
    write_json,
)
from .geometry import (
    camera_calibration,
    decode_gps,
    decode_pose,
    match_cameras,
    raw_projection,
    wgs84_xy,
    world_delta_to_published,
)
from .protocol import (
    CAMERAS,
    UNIAD_TO_KIT,
    oracle_turn,
    past_can_bus,
    phase_inputs,
    validate_poses,
)

METADATA = ("poses.txt", "timestamp.reference.txt", "calibration/calib.json")


def select_shards(release):
    if (release.get("dataset"), release.get("version"), release.get("status")) != (
        "kitscenes-val",
        "v3.5",
        "ready",
    ):
        raise ValueError("Unexpected mirror release")
    partitions = {}
    for item in release["partitions"]:
        pid = item["partition_id"]
        if not re.fullmatch(r"part-[0-9a-f]{16}", pid) or pid in partitions:
            raise ValueError("Invalid/duplicate partition")
        split = re.findall(r"/kitscenes-v35-(val|overlap)-[^/]+/", item["source_uri"])
        if len(split) != 1:
            raise ValueError("Unknown publisher split")
        partitions[pid] = dict(item, split=split[0])
    selected = {pid: p for pid, p in partitions.items() if p["split"] == "val"}
    if (
        len(selected) != 117
        or sum(p["sample_count"] for p in selected.values()) != 11035
    ):
        raise ValueError(
            "Expected 117 Val partitions / 11035 samples, excluding overlap"
        )
    counts = dict.fromkeys(selected, 0)
    names = []
    for shard in release["shard_entries"]:
        match = re.fullmatch(r"(part-[0-9a-f]{16})-train-\d{6}\.tar", shard["name"])
        if not match:
            raise ValueError("Unexpected shard name")
        if match[1] in selected:
            if shard["key"] != "kitscenes-val/v3.5/shards/" + shard["name"]:
                raise ValueError("Shard key/name mismatch")
            names.append("datasets/" + shard["key"])
            counts[match[1]] += 1
    if len(names) != len(set(names)) or any(
        counts[pid] != p["shard_count"] for pid, p in selected.items()
    ):
        raise ValueError("Duplicate or missing selected shard")
    return sorted(names)


def read_shard(path):
    """Read bounded metadata only; packed/resized JPEGs are never model inputs."""
    blobs = {}
    suffixes = (
        ".meta.json",
        ".calib.json",
        ".bev_history_index.json",
        ".trajectory_xy.npz",
        ".pose.npy",
        ".gps.npy",
    )
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            safe_child(path.parent, member.name)
            if not member.name.endswith(suffixes):
                continue
            if (
                member.name in blobs
                or not member.isfile()
                or not 0 < member.size <= 2 * 2**20
            ):
                raise ValueError("Invalid selected mirror member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Unreadable mirror member")
            blobs[member.name] = handle.read()
    anchors = {}
    scene = None
    for name, blob in blobs.items():
        if not name.endswith(".meta.json"):
            continue
        meta = json.loads(blob)
        sid = meta["split_group_uid"].removeprefix("kitscenes-")
        if not re.fullmatch(r"[0-9a-f-]{36}", sid) or scene not in (None, sid):
            raise ValueError("Invalid/mixed physical scene in mirror shard")
        scene = sid
        index, uid = meta["frame_idx"], meta["sample_uid"]
        if (
            uid != f"kitscenes-v1-{sid}-f{index:06d}"
            or index in anchors
            or name != uid + ".meta.json"
        ):
            raise ValueError("Unexpected or duplicate mirror sample")
        calib = json.loads(blobs[uid + ".calib.json"])
        if (
            calib.get("temporal_frame_offsets") != list(range(-7, 1))
            or calib.get("temporal_frame_interval_us") != 500000
        ):
            raise ValueError("Mirror temporal contract changed")
        history = json.loads(blobs[uid + ".bev_history_index.json"])
        expected = [
            [f"kitscenes-v1-{sid}-r{index + d:06d}-c{c}" for c in range(6)]
            for d in range(-35, 0, 5)
        ]
        if history != expected:
            raise ValueError("Mirror history is not causal 2 Hz")
        with np.load(
            io.BytesIO(blobs[uid + ".trajectory_xy.npz"]), allow_pickle=False
        ) as arrays:
            target, valid = arrays["trajectory_xy_m"], arrays["trajectory_valid"]
            if (
                target.shape != (64, 2)
                or valid.shape != (64,)
                or not valid[:30].all()
                or not np.isfinite(target[:30]).all()
            ):
                raise ValueError(
                    "Missing/invalid 3-second target; no exclusions allowed"
                )
            target = target[:30].copy()
        pose = decode_pose(blobs[uid + ".pose.npy"])
        gps = decode_gps(blobs[uid + ".gps.npy"])
        if not np.allclose(
            gps[0], [pose["latitude"], pose["longitude"]], atol=1e-9, rtol=0
        ):
            raise ValueError("Mirror GPS origin mismatch")
        if not np.allclose(target, wgs84_xy(gps[1:31], pose), atol=1e-5, rtol=0):
            raise ValueError("Mirror target differs from the pinned WGS84 convention")
        anchors[index] = dict(
            sample_uid=uid,
            target_xy_10hz=target.tolist(),
            evaluation_pose=pose,
            projection_512=calib["projection"]["matrix"],
        )
    if scene is None or not anchors:
        raise ValueError("Empty mirror shard")
    return scene, anchors


def extract_selected(archive_path, root, sid, indices, metadata_only=False):
    wanted = set(METADATA)
    if not metadata_only:
        wanted |= {
            f"{camera}/{i:010d}.{ext}"
            for camera in CAMERAS
            for i in indices
            for ext in ("jpg", "jpeg", "png")
        }
    found = set()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            safe_child(root, member.name)
            parts = PurePosixPath(member.name).parts
            relative = PurePosixPath(
                *(parts[1:] if parts[0] == sid else parts)
            ).as_posix()
            if relative not in wanted:
                continue
            if (
                relative in found
                or not member.isfile()
                or not 0 < member.size <= 64 * 2**20
            ):
                raise ValueError("Invalid or duplicate original archive member")
            found.add(relative)
            target = safe_child(root, relative)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Unreadable original archive member")
            blob = handle.read()
            if len(blob) != member.size:
                raise ValueError("Truncated original archive")
            if target.exists():
                if sha256(target) != hashlib.sha256(blob).hexdigest():
                    raise ValueError("Existing source data differs; refusing overwrite")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(blob)
    if not set(METADATA) <= found:
        raise ValueError("Missing original metadata")


def image_records(raw, index, calibrations):
    records = []
    for camera, calibration in zip(CAMERAS, calibrations):
        paths = [
            safe_child(raw, f"{camera}/{index:010d}.{ext}")
            for ext in ("jpg", "jpeg", "png")
        ]
        paths = [p for p in paths if p.is_file()]
        if len(paths) != 1:
            raise ValueError(f"Missing/ambiguous image: {camera} frame {index}")
        path = paths[0]
        with Image.open(path) as image:
            if image.size != calibration.image_size:
                raise ValueError("Original image/calibration extent mismatch")
        records.append(dict(path=str(path.resolve()), sha256=sha256(path)))
    return records


def scene_streams(sid, anchors, poses, raw, payload, smoke=False):
    """Construct the evaluated five dense streams and independent 4s windows."""
    if payload["reference"] != "base_frame":
        raise ValueError("Calibration reference is not base_frame")
    calibrations = [camera_calibration(payload, name) for name in CAMERAS]
    projections = [raw_projection(c, c.image_size, c.image_size) for c in calibrations]
    resized = [raw_projection(c, c.image_size, (1600, 900)) for c in calibrations]
    shapes = [[c.image_size[1], c.image_size[0]] for c in calibrations]
    for index, anchor in anchors.items():
        if index < 35 or index + 30 >= len(poses):
            raise ValueError("Mirror anchor lacks complete history/future")
        rotation = Rotation.from_quat(poses[index, 4:8]).as_matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pose = anchor["evaluation_pose"]
        if abs(pose["timestamp_ns"] / 1e9 - poses[index, 0]) > 1e-6:
            raise ValueError("Mirror/raw timestamp mismatch")
        error = (pose["heading_deg"] - (90 - np.degrees(yaw)) + 180) % 360 - 180
        if abs(error) > 1e-5:
            raise ValueError("Mirror/raw heading mismatch")
        check = world_delta_to_published(
            poses[index + 1 : index + 31, 1:3] - poses[index, 1:3], pose
        )
        if np.max(np.linalg.norm(check - anchor["target_xy_10hz"], axis=1)) > 0.005:
            raise ValueError("Mirror/raw targets disagree by over 0.005 m")
        packed = np.asarray(anchor["projection_512"], dtype=float)
        if packed.shape != (6, 3, 4) or not np.isfinite(packed).all():
            raise ValueError("Invalid packed projection")
        transform = np.eye(4)
        transform[:3, :3] = UNIAD_TO_KIT
        padded = np.tile(np.eye(4), (6, 1, 1))
        padded[:, :3] = packed[[0, 2, 1, 4, 3, 5]] @ transform
        padded[:, 0] *= 1600 / 512
        padded[:, 1] *= 900 / 512
        match_cameras(padded, resized)
    phases = phase_inputs(list(anchors))
    keyframes = list(range(40, len(poses) - 30, 90))
    if smoke:
        phase, indices, scored = phases[0]
        phases = [(phase, indices[:9], scored.intersection(indices[:9]))]
        keyframes = keyframes[:1]
    needed = {i for _, indices, _ in phases for i in indices}
    needed |= {i for anchor in keyframes for i in range(anchor - 40, anchor + 1, 5)}
    images = {i: image_records(raw, i, calibrations) for i in sorted(needed)}

    def timeline(name, indices, scored, track):
        stream = dict(scene_id=f"{sid}:{name}", physical_scene_id=sid, frames=[])
        for i in indices:
            rotation = Rotation.from_quat(poses[i, 4:8]).as_matrix()
            can = past_can_bus(poses, i)
            can[:3] = poses[i, 1:4] - poses[indices[0], 1:4]
            frame = dict(
                frame=i,
                timestamp=float(poses[i, 0]),
                scored=i in scored,
                raw_kitscenes_images=images[i],
                image_shapes=shapes,
                lidar2img=np.asarray(projections).tolist(),
                can_bus=can.tolist(),
                l2g_r_mat=(rotation @ UNIAD_TO_KIT).T.tolist(),
                l2g_t=can[:3].tolist(),
                body_to_utm_xy=rotation[:2, :2].tolist(),
            )
            if i in scored:
                velocity = (poses[i, 1:4] - poses[i - 1, 1:4]) / (
                    poses[i, 0] - poses[i - 1, 0]
                )
                frame.update(
                    sample_uid=f"kitscenes-v1-{sid}-f{i:06d}",
                    past_velocity_xy=(rotation.T @ velocity)[:2].tolist(),
                    past_velocity_world_xy=velocity[:2].tolist(),
                    oracle_command=2,
                )
                if track == "dense":
                    frame.update(
                        {
                            key: anchors[i][key]
                            for key in ("target_xy_10hz", "evaluation_pose")
                        }
                    )
                    frame["oracle_command"] = oracle_turn(
                        np.asarray(frame["target_xy_10hz"])
                    )
            stream["frames"].append(frame)
        return stream

    return {
        "dense": [
            timeline(f"phase{p}", indices, scored, "dense")
            for p, indices, scored in phases
        ],
        "keyframes": [
            timeline(f"keyframe{i}", list(range(i - 40, i + 1, 5)), {i}, "keyframes")
            for i in keyframes
        ],
    }


def prepare(args):
    args.out = args.out.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.mirror_dir = args.mirror_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="preparing",
        full_evaluation=not args.scenes and not args.smoke,
        raw_revision=RAW_REVISION,
        mirror_revision=MIRROR_REVISION,
        scenes=[],
        scope="smoke" if args.smoke else "subset" if args.scenes else "full",
    )
    write_json(args.out / "manifest.json", report)
    try:
        release_name = MIRROR_PREFIX + "shards/manifest.json"
        if args.download:
            verified_download(
                MIRROR_REPO, MIRROR_REVISION, release_name, args.mirror_dir
            )
        release_path = safe_child(args.mirror_dir, release_name)
        names = select_shards(read_json(release_path))
        report["mirror_manifest_sha256"] = sha256(release_path)
        scenes = {}
        for name in names:
            path = (
                verified_download(MIRROR_REPO, MIRROR_REVISION, name, args.mirror_dir)
                if args.download
                else safe_child(args.mirror_dir, name)
            )
            sid, anchors = read_shard(path)
            if sid in scenes:
                raise ValueError("Repeated physical scene")
            scenes[sid] = (anchors, sha256(path))
        all_ids = [
            a["sample_uid"] for anchors, _ in scenes.values() for a in anchors.values()
        ]
        if (
            len(scenes) != 117
            or len(all_ids) != 11035
            or uid_hash(all_ids) != DENSE_UID_SHA256
        ):
            raise ValueError(
                "Mirror membership differs from evaluated 11035 unique IDs"
            )
        selected = args.scenes or list(scenes)
        if len(selected) != len(set(selected)) or not set(selected) <= set(scenes):
            raise ValueError("Unknown/duplicate scene selection")
        if args.smoke:
            selected = selected[:1]
        totals = {track: dict.fromkeys(EXPECTED[track], 0) for track in EXPECTED}
        membership_ids = {track: [] for track in EXPECTED}
        for sid in selected:
            anchors, shard_sha = scenes[sid]
            raw = safe_child(args.raw_dir, sid)
            raw.mkdir(parents=True, exist_ok=True)
            archive = None
            archive_sha = None
            if args.download:
                archive = verified_download(
                    RAW_REPO,
                    RAW_REVISION,
                    f"data/val/{sid}.tar",
                    args.raw_dir / "archives",
                    "dataset",
                )
                archive_sha = sha256(archive)
                extract_selected(archive, raw, sid, [], metadata_only=True)
            poses = np.loadtxt(safe_child(raw, "poses.txt"))
            validate_poses(
                poses, np.loadtxt(safe_child(raw, "timestamp.reference.txt"))
            )
            indices = {
                i for _, timeline, _ in phase_inputs(list(anchors)) for i in timeline
            }
            indices |= {
                i
                for anchor in range(40, len(poses) - 30, 90)
                for i in range(anchor - 40, anchor + 1, 5)
            }
            if archive is not None:
                extract_selected(archive, raw, sid, indices)
            tracks = scene_streams(
                sid,
                anchors,
                poses,
                raw,
                read_json(raw / "calibration/calib.json"),
                args.smoke,
            )
            destination = args.out / "scenes" / sid
            destination.mkdir(parents=True)
            for name in METADATA:
                target = safe_child(destination / "source_metadata", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_child(raw, name), target)
            scene = dict(
                scene_id=sid,
                tracks=tracks,
                mirror_shard_sha256=shard_sha,
                original_archive_sha256=archive_sha,
                metadata_sha256={
                    name: sha256(destination / "source_metadata" / name)
                    for name in METADATA
                },
            )
            write_json(destination / "inputs.json", scene)
            report["scenes"].append(
                dict(
                    scene_id=sid,
                    path=f"scenes/{sid}/inputs.json",
                    sha256=sha256(destination / "inputs.json"),
                )
            )
            for track, streams in tracks.items():
                for key, value in coverage(streams).items():
                    totals[track][key] += value
                membership_ids[track].extend(
                    f["sample_uid"] for s in streams for f in s["frames"] if f["scored"]
                )
            print(
                json.dumps(
                    dict(scene=sid, prepared=len(report["scenes"]), coverage=totals)
                ),
                flush=True,
            )
            write_json(args.out / "manifest.json", report)
        for track, ids in membership_ids.items():
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate IDs within an evaluation cohort")
        if report["full_evaluation"] and totals != EXPECTED:
            raise ValueError(f"Incomplete full coverage: {totals}")
        report.update(
            status="ready",
            coverage=totals,
            sample_uid_sha256={t: uid_hash(v) for t, v in membership_ids.items()},
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out / "manifest.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Original scene directories; downloads also go under archives/",
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        required=True,
        help="HF model snapshot root containing datasets/kitscenes-val/v3.5/",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New prepared-input directory"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch and verify pinned releases (large); otherwise use supplied local files",
    )
    parser.add_argument(
        "--scenes", nargs="+", help="Explicit subset, never reported as full validation"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="First selected scene: two dense anchors/nine frames and one nine-frame keyframe window",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
