#!/usr/bin/env python3
"""Build a Qwen-Drive-1.0 planning scene file (JSONL) from KITScenes Multimodal.

The script only READS the dataset. It writes a single JSONL file (plus an optional
summary JSON) somewhere outside the dataset root; no image, pose or metadata file of
KITScenes is created, moved or modified. Images are referenced by path relative to
``--kitscenes-root/data``, which is what you later pass to Qwen-Drive as
``--image-root``.

One output line = one planning query, in the record format of
``Qwen-Drive-1.0/docs/data.md``:

* 3 cameras x 4 timestamps (t-1.5 s, t-1.0 s, t-0.5 s, t) -> ``messages[0].content``
* 1.5 s ego history and 5 s future at 10 Hz, current ego frame (x fwd, y left)
* ego status, driving command, navigation command, instruction text
* ``meta_info`` with ``token`` = ``<scene_id>-<frame_idx>`` and ``scene_token``

Keyframes follow the KITScenes end-to-end protocol by default: non-overlapping 9 s
windows (4 s past + 5 s future) at 10 Hz, i.e. frame 40, 130, 220, ... of every scene.
For ``test_e2e`` (future withheld) the last frame of each scene is the keyframe.

Usage
-----
    export KITSCENES_ROOT=/data/kitscenes          # contains data/<split>/<scene_id>/
    python kitscenes_to_qwendrive.py --split test \
        --output scenes/kitscenes_test.jsonl

    # then, in the Qwen-Drive checkout:
    python scripts/run_planning.py --model Qwen-Drive-1.0-4B \
        --planner Qwen-Drive-1.0-4B/planner-rl \
        --scenes scenes/kitscenes_test.jsonl \
        --image-root $KITSCENES_ROOT/data \
        --output outputs/kitscenes_test/predictions.jsonl \
        --mode reasoning_planning --num-samples 6

Requires: the ``kitscenes`` devkit (github.com/KIT-MRT/kitscenes), numpy, scipy, Pillow.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

log = logging.getLogger("kitscenes_to_qwendrive")

# ---------------------------------------------------------------------------
# Qwen-Drive conventions (copied from qwen_drive/scene.py and the released scene files)
# ---------------------------------------------------------------------------

VIEW_TAGS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")
NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")      # index = nav_command
INTENTS = ("GO_STRAIGHT", "GO_LEFT", "GO_RIGHT")
# driving_command one-hot order: left, straight, right, unknown
DRIVING_COMMAND = {0: [0, 1, 0, 0], 1: [1, 0, 0, 0], 2: [0, 0, 1, 0], None: [0, 0, 0, 1]}
FRAME_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains 4 temporal frames "
    "captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn：\n"
)

HZ = 10.0
DT = 1.0 / HZ
NUM_HIST = 16          # 1.5 s at 10 Hz incl. current
NUM_FUTURE = 50        # 5.0 s at 10 Hz
CAM_OFFSETS = (-15, -10, -5, 0)   # frame offsets of the 4 camera timestamps (10 Hz timeline)
DERIV_PAD = 5          # extra 10 Hz samples on each side for smooth derivatives
ROUND = 5              # decimals stored, as in the released files

# KITScenes e2e protocol: 4 s past + 5 s future, non-overlapping 9 s windows
PROTOCOL_PAST = 40
PROTOCOL_WINDOW = 90

DEFAULT_CAMERAS = ("camera_ring_front", "camera_ring_front_left", "camera_ring_front_right")


# ---------------------------------------------------------------------------
# Pose handling
# ---------------------------------------------------------------------------

@dataclass
class PoseTrack:
    """Ego poses of one scene (world <- reference/body frame), time-interpolable."""

    t: np.ndarray            # (N,) seconds, strictly increasing
    xyz: np.ndarray          # (N, 3) UTM metres
    rot: Rotation            # N rotations, body -> world

    @classmethod
    def from_ego_poses(cls, poses) -> "PoseTrack":
        t = np.array([p.timestamp_ns for p in poses], dtype=np.float64) / 1e9
        xyz = np.array([p.translation for p in poses], dtype=np.float64)
        quat = np.array([p.rotation for p in poses], dtype=np.float64)  # qx qy qz qw
        order = np.argsort(t, kind="stable")
        t, xyz, quat = t[order], xyz[order], quat[order]
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        return cls(t=t[keep], xyz=xyz[keep], rot=Rotation.from_quat(quat[keep]))

    def covers(self, times: np.ndarray, tol: float = 0.02) -> np.ndarray:
        return (times >= self.t[0] - tol) & (times <= self.t[-1] + tol)

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, Rotation]:
        tc = np.clip(times, self.t[0], self.t[-1])
        xyz = np.stack([np.interp(tc, self.t, self.xyz[:, i]) for i in range(3)], axis=1)
        return xyz, Slerp(self.t, self.rot)(tc)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _planar_heading(rot_rel: Rotation) -> np.ndarray:
    """Heading of each body x-axis projected onto the current ego x/y plane."""
    fwd = rot_rel.apply(np.array([1.0, 0.0, 0.0]))
    return np.arctan2(fwd[:, 1], fwd[:, 0])


def ego_kinematics(track: PoseTrack, t0: float, smooth: bool = True) -> dict:
    """10 Hz history/future of pose, velocity, acceleration in the ego frame at ``t0``.

    Positions and headings are expressed in the body frame at t0. Velocities and
    accelerations are expressed in the body frame of *their own* timestamp, as in the
    released Qwen-Drive scene files.
    """
    k = np.arange(-(NUM_HIST - 1) - DERIV_PAD, NUM_FUTURE + DERIV_PAD + 1)
    times = t0 + k * DT
    covered = track.covers(times)
    xyz, rot = track.sample(times)

    # World-frame derivatives, computed only over samples the recording covers so that
    # the clamped (constant) positions beyond its ends never bias the velocity — this
    # matters for test_e2e, where the recording stops at the keyframe. mode="interp"
    # fits the edge windows one-sidedly instead of padding.
    idx = np.flatnonzero(covered)
    seg = xyz[idx[0]: idx[-1] + 1]
    vel_w = np.zeros_like(xyz)
    acc_w = np.zeros_like(xyz)
    if smooth and len(seg) >= 9:
        v = savgol_filter(seg, 7, 3, deriv=1, delta=DT, axis=0, mode="interp")
        a = savgol_filter(seg, 9, 3, deriv=2, delta=DT, axis=0, mode="interp")
    else:
        v = np.gradient(seg, DT, axis=0, edge_order=2 if len(seg) > 2 else 1)
        a = np.gradient(v, DT, axis=0)
    vel_w[idx[0]: idx[-1] + 1], acc_w[idx[0]: idx[-1] + 1] = v, a
    vel_w[: idx[0]], acc_w[: idx[0]] = v[0], a[0]
    vel_w[idx[-1] + 1:], acc_w[idx[-1] + 1:] = v[-1], a[-1]
    vel_b = rot.inv().apply(vel_w)[:, :2]
    acc_b = rot.inv().apply(acc_w)[:, :2]

    # poses relative to the current body frame
    i0 = int(np.where(k == 0)[0][0])
    rot0 = rot[i0]
    rel_xyz = rot0.inv().apply(xyz - xyz[i0])
    heading = _wrap(_planar_heading(rot0.inv() * rot))
    heading[i0] = 0.0
    rel_xyz[i0] = 0.0
    pose = np.column_stack([rel_xyz[:, :2], heading])
    yawrate = np.gradient(np.unwrap(heading), DT)

    hist = slice(DERIV_PAD, DERIV_PAD + NUM_HIST)                     # t0-1.5 .. t0
    fut = slice(DERIV_PAD + NUM_HIST, DERIV_PAD + NUM_HIST + NUM_FUTURE)  # t0+0.1 .. t0+5.0

    # poses beyond the recording are clamped to its last pose; mark them invalid
    fut_valid = covered[fut].astype(np.int64)
    hist_valid = covered[hist].astype(np.int64)
    return {
        "hist_pose": pose[hist], "hist_vel": vel_b[hist], "hist_acc": acc_b[hist],
        "hist_yawrate": yawrate[hist], "hist_valid": hist_valid,
        "fut_pose": pose[fut], "fut_vel": vel_b[fut], "fut_acc": acc_b[fut],
        "fut_yawrate": yawrate[fut], "fut_valid": fut_valid,
        "cur_vel": vel_b[i0], "cur_acc": acc_b[i0], "cur_yawrate": float(yawrate[i0]),
    }


def nav_command_from_future(fut_pose: np.ndarray, fut_valid: np.ndarray,
                            lateral_threshold: float) -> int | None:
    """0 straight / 1 left / 2 right from the GT future (KITScenes protocol), None if unknown."""
    idx = np.flatnonzero(fut_valid)
    if idx.size == 0:
        return None
    y_end = float(fut_pose[idx[-1], 1])
    if y_end > lateral_threshold:
        return 1
    if y_end < -lateral_threshold:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def target_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Aspect-preserving (width, height) whose shorter side is ``short_side``."""
    s = short_side / min(width, height)
    return int(round(width * s / 2) * 2), int(round(height * s / 2) * 2)


def r(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), ROUND).tolist()


def instruction_text(hist_pose: np.ndarray, nav_command: int) -> str:
    lines = "".join(
        " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *hist_pose[i])
        for label, i in zip(FRAME_LABELS, (0, 5, 10, 15))
    )
    return f"{INSTRUCTION_HEADER}{lines}2. Active navigation command: [{NAV_COMMANDS[nav_command]}]"


def build_record(*, scene_id, split, frame_idx, t0, kin, frames, cameras, nav, nav_source,
                 image_sizes, hist_short, cur_short) -> dict:
    content: list[dict[str, Any]] = []
    for tag, cam in zip(VIEW_TAGS, cameras):
        w, h = image_sizes[cam]
        content.append({"text": tag})
        for j, (offset, rel_path) in enumerate(frames[cam]):
            rw, rh = target_size(w, h, cur_short if offset == 0 else hist_short)
            content.append({"text": f"frame: {j}"})
            content.append({
                "image": rel_path, "width": w, "height": h,
                "timestamp_idx": j, "timestamp_offset_sec": offset * DT,
                "resized_width": rw, "resized_height": rh,
            })

    hist_pose = np.round(kin["hist_pose"], ROUND)
    nav_cmd = 0 if nav is None else nav          # model needs one; flagged in meta
    content.append({"text": instruction_text(hist_pose, nav_cmd)})

    speed = float(np.hypot(*kin["cur_vel"]))
    driving_command = DRIVING_COMMAND[nav]
    hist_times = r(np.arange(-(NUM_HIST - 1), 1) * DT)
    trajectory = {
        "hist_traj_10hz": r(hist_pose),
        "hist_vel_10hz": r(kin["hist_vel"]),
        "hist_acc_10hz": r(kin["hist_acc"]),
        "hist_yawrate_10hz": r(kin["hist_yawrate"]),
        "hist_times_10hz": hist_times,
        "hist_valid_mask_10hz": kin["hist_valid"].tolist(),
        "future_traj_10hz": r(kin["fut_pose"]),
        "future_vel_10hz": r(kin["fut_vel"]),
        "future_acc_10hz": r(kin["fut_acc"]),
        "future_yawrate_10hz": r(kin["fut_yawrate"]),
        "future_times_10hz": r(np.arange(1, NUM_FUTURE + 1) * DT),
        "future_valid_mask_10hz": kin["fut_valid"].tolist(),
        "hist_hz_10hz": HZ, "future_hz_10hz": HZ, "traj_10hz_hz": HZ,
        "traj_10hz_source": "kitscenes_poses_txt_slerp_10hz_savgol",
        "ego_status": {
            "in_global_frame": False,
            "ego_pose": [0.0, 0.0, 0.0],
            "ego_velocity": r(kin["cur_vel"]),
            "ego_acceleration": r(kin["cur_acc"]),
            "driving_command": driving_command,
        },
        "current_state": {
            "ego_velocity": r(kin["cur_vel"]), "ego_acceleration": r(kin["cur_acc"]),
            "ego_yawrate": round(kin["cur_yawrate"], ROUND), "speed": round(speed, ROUND),
        },
        "nav_command": nav_cmd,
        "driving_command": driving_command,
    }
    meta = {
        "dataset": "KITScenes_Multimodal",
        "task_name": "trajectory_prediction",
        "split": split,
        "token": f"{scene_id}-{frame_idx}",
        "scene_token": scene_id,
        "frame_idx": frame_idx,
        "timestamp_s": round(t0, 6),
        "cam_order": list(cameras),
        "num_history_frames": 4,
        "history_time_offsets": [o * DT for o in CAM_OFFSETS],
        "image_hz": 2.0,
        "traj_hz": HZ,
        "intent": INTENTS[nav] if nav is not None else "UNKNOWN",
        "nav_command_source": nav_source,
        "num_valid_future": int(kin["fut_valid"].sum()),
    }
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
        "meta_info": meta,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------

def select_keyframes(mode: str, num_frames: int, stride: int, min_future: int) -> list[int]:
    first = -CAM_OFFSETS[0]                         # need 1.5 s of camera history
    if mode == "protocol":
        ks = list(range(PROTOCOL_PAST, num_frames, PROTOCOL_WINDOW))
    elif mode == "last":
        ks = [num_frames - 1]
    elif mode == "stride":
        ks = list(range(first, num_frames, stride))
    else:
        raise ValueError(mode)
    return [k for k in ks if k >= first and (mode == "last" or k + min_future < num_frames)]


def load_keyframe_file(path: Path) -> dict[str, list[int]]:
    """CSV with columns scene_id,frame_idx (header optional)."""
    out: dict[str, list[int]] = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in ("scene_id", "scene", "#"):
                continue
            out.setdefault(row[0].strip(), []).append(int(row[1]))
    return out


def load_command_file(path: Path) -> dict[str, int]:
    """CSV scene_id,frame_idx,command with command in {straight,left,right} or {0,1,2}."""
    names = {"straight": 0, "go_straight": 0, "left": 1, "turn_left": 1, "go_left": 1,
             "right": 2, "turn_right": 2, "go_right": 2}
    out: dict[str, int] = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in ("scene_id", "scene", "#"):
                continue
            c = row[2].strip().lower()
            out[f"{row[0].strip()}-{int(row[1])}"] = int(c) if c.isdigit() else names[c]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kitscenes-root", type=Path, default=os.environ.get("KITSCENES_ROOT"),
                   help="dataset root containing data/<split>/<scene_id>/ (or $KITSCENES_ROOT)")
    p.add_argument("--split", required=True,
                   choices=["train", "val", "test", "test_e2e", "overlap_train_val"])
    p.add_argument("--output", type=Path, required=True, help="output JSONL path")
    p.add_argument("--keyframes", choices=["protocol", "stride", "last"], default=None,
                   help="protocol: KITScenes e2e 9 s windows (frame 40, 130, ...); "
                        "stride: every --stride frames; last: last frame of each scene. "
                        "Default: 'last' for test_e2e, else 'protocol'.")
    p.add_argument("--stride", type=int, default=10, help="frame stride for --keyframes stride")
    p.add_argument("--keyframe-file", type=Path, default=None,
                   help="CSV scene_id,frame_idx overriding keyframe selection (e.g. official list)")
    p.add_argument("--min-future", type=int, default=30,
                   help="minimum number of future 10 Hz poses a keyframe needs "
                        "(30 = 3 s, the primary KITScenes horizon; ignored for 'last')")
    p.add_argument("--cameras", nargs=3, default=list(DEFAULT_CAMERAS),
                   metavar=("FRONT", "FRONT_LEFT", "FRONT_RIGHT"),
                   help="KITScenes cameras mapped to Qwen-Drive's 3 views, in that order")
    p.add_argument("--history-short-side", type=int, default=320,
                   help="short side (px) history frames are resized to (~320p in Qwen-Drive)")
    p.add_argument("--current-short-side", type=int, default=720,
                   help="short side (px) the current frame is resized to (~720p in Qwen-Drive)")
    p.add_argument("--lateral-threshold", type=float, default=2.0,
                   help="|y| (m) at the end of the GT future that makes a left/right command")
    p.add_argument("--command-file", type=Path, default=None,
                   help="CSV scene_id,frame_idx,command overriding the derived command")
    p.add_argument("--no-smoothing", action="store_true",
                   help="plain finite differences instead of Savitzky-Golay derivatives")
    p.add_argument("--scenes", nargs="*", default=None, help="restrict to these scene ids")
    p.add_argument("--skip-missing-images", action="store_true",
                   help="drop a keyframe whose camera frame is missing instead of failing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.kitscenes_root is None:
        p.error("--kitscenes-root not given and $KITSCENES_ROOT not set")
    if args.keyframes is None:
        args.keyframes = "last" if args.split == "test_e2e" else "protocol"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    from kitscenes import KITScenesDataset
    from kitscenes.sensors import SensorDataLoader

    root = Path(args.kitscenes_root)
    if _is_inside(args.output, root):
        log.error("refusing to write inside the dataset root (%s); choose another --output", root)
        return 2

    ds = KITScenesDataset(root=root, split=args.split)
    image_root = root / "data" if (root / "data").is_dir() else root
    scene_ids = ds.scene_ids
    if args.scenes:
        wanted = set(args.scenes)
        scene_ids = [s for s in scene_ids if s in wanted]
    split_file_ids = _split_file_count(args.split)
    log.info("split %s: %d scenes on disk (%s listed in split file), image root %s",
             args.split, len(scene_ids), split_file_ids, image_root)
    if split_file_ids and len(ds.scene_ids) < split_file_ids:
        log.warning("only %d of %d %s scenes are on disk; the file will be incomplete",
                    len(ds.scene_ids), split_file_ids, args.split)

    keyframe_override = load_keyframe_file(args.keyframe_file) if args.keyframe_file else None
    commands = load_command_file(args.command_file) if args.command_file else {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".partial")
    stats: Counter[str] = Counter()
    speed_sum = 0.0
    with open(tmp_path, "w") as out:
        for n, scene_id in enumerate(scene_ids, 1):
            scene = ds.get_scene(scene_id)
            loader: SensorDataLoader = ds.get_sensor_loader(scene_id)
            num_frames = int(scene.metadata.num_frames)
            poses = scene.ego_poses
            if len(poses) < 2:
                log.warning("%s: no usable poses.txt, skipped", scene_id)
                stats["scenes_without_poses"] += 1
                continue
            track = PoseTrack.from_ego_poses(poses)
            ref_ts = scene.timestamps_ns.astype(np.float64) / 1e9

            missing_cams = [c for c in args.cameras if c not in loader.get_camera_names()]
            if missing_cams:
                log.warning("%s: cameras %s missing, skipped", scene_id, missing_cams)
                stats["scenes_missing_cameras"] += 1
                continue

            if keyframe_override is not None:
                keyframes = sorted(keyframe_override.get(scene_id, []))
            else:
                # frames with a pose: the recording may end (test_e2e) before the timeline
                last_posed = int(np.searchsorted(ref_ts, track.t[-1] + 0.02, side="right")) - 1
                usable = min(num_frames, last_posed + 1)
                keyframes = select_keyframes(args.keyframes, usable, args.stride, args.min_future)
            if not keyframes:
                stats["scenes_without_keyframes"] += 1
                log.debug("%s: %d frames, no keyframe fits", scene_id, num_frames)
                continue

            image_sizes = {}
            for cam in args.cameras:
                image_sizes[cam] = tuple(loader.get_camera_image_size(cam, keyframes[0]))

            for kf in keyframes:
                if kf + CAM_OFFSETS[0] < 0 or kf >= num_frames:
                    stats["keyframes_out_of_range"] += 1
                    continue
                frames: dict[str, list[tuple[int, str]]] = {}
                ok = True
                for cam in args.cameras:
                    frames[cam] = []
                    for off in CAM_OFFSETS:
                        try:
                            path = loader.get_camera_image_path(cam, kf + off)
                        except FileNotFoundError:
                            ok = False
                            break
                        frames[cam].append((off, os.path.relpath(path, image_root)))
                    if not ok:
                        break
                if not ok:
                    if args.skip_missing_images:
                        stats["keyframes_missing_images"] += 1
                        continue
                    raise FileNotFoundError(f"{scene_id} frame {kf}: camera image missing")

                t0 = float(np.interp(kf, np.arange(num_frames), ref_ts))
                # anchor on the pose closest to the frame's reference time
                t0 = float(track.t[np.argmin(np.abs(track.t - t0))])
                kin = ego_kinematics(track, t0, smooth=not args.no_smoothing)

                token = f"{scene_id}-{kf}"
                nav: int | None
                if token in commands:
                    nav, nav_source = commands[token], "command_file"
                else:
                    nav = nav_command_from_future(kin["fut_pose"], kin["fut_valid"],
                                                  args.lateral_threshold)
                    nav_source = "gt_future_lateral" if nav is not None else "unknown"
                if nav is None:
                    stats["records_unknown_command"] += 1

                speed = float(np.hypot(*kin["cur_vel"]))
                if np.abs(kin["hist_yawrate"]).max() > 1.2 or \
                        np.linalg.norm(kin["hist_acc"], axis=1).max() > 9.81:
                    stats["records_kinematic_outlier"] += 1

                rec = build_record(
                    scene_id=scene_id, split=args.split, frame_idx=kf, t0=t0, kin=kin,
                    frames=frames, cameras=args.cameras, nav=nav, nav_source=nav_source,
                    image_sizes=image_sizes, hist_short=args.history_short_side,
                    cur_short=args.current_short_side,
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["records"] += 1
                stats[f"command_{rec['meta_info']['intent']}"] += 1
                speed_sum += speed
                stats["records_full_5s_future"] += int(kin["fut_valid"].all())
            stats["scenes_used"] += 1
            if n % 20 == 0:
                log.info("%d/%d scenes, %d records", n, len(scene_ids), stats["records"])

    os.replace(tmp_path, args.output)
    summary: dict[str, Any] = dict(stats)
    if stats["records"]:
        summary["mean_speed_mps"] = round(speed_sum / stats["records"], 3)
    summary.update(split=args.split, keyframes=args.keyframes, cameras=args.cameras,
                   output=str(args.output))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %d records to %s", stats["records"], args.output)
    log.info("summary: %s", json.dumps(summary))
    return 0


def _split_file_count(split: str) -> int | None:
    try:
        import kitscenes.dataset as kd
        f = kd._SPLIT_DIR / kd._SPLIT_FILES[split]
        return sum(1 for line in f.read_text().splitlines() if line.strip())
    except Exception:  # noqa: BLE001 - informational only
        return None


if __name__ == "__main__":
    sys.exit(main())
