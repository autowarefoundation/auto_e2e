#!/usr/bin/env python3
"""Prepare KITScenes val for end-to-end planning evaluation of Qwen-Drive-1.0 and Alpamayo-1.5.

One scene file serves both models, so they see the same samples, the same t0 and the
same ground-truth future:

* Qwen-Drive-1.0 reads it directly (scripts/run_planning.py --scenes ... --image-root ...):
  ChatML record with 3 views (front, front-left, front-right) x 4 frames at 2 Hz,
  hist_traj/vel/acc_10hz (16), ego_status, nav_command, future_traj_10hz (50).
* eval_alpamayo15_kitscenes.py reads the extra "alpamayo" block: 4 cameras
  (cross-left, front-wide, cross-right, front-tele) x 4 frames at 10 Hz and the 3D ego
  history (16 poses, xyz + rotation).

Population: 117 scenes / 11,035 samples
    The official KITScenes val split (kitscenes/split/.../validation.txt, 117 scenes) with
    the t0 frames used by the AutoE2E v3.5 mirror, read from its sample_pose.parquet
    (one row per sample, ~0.5 MB) and cross-checked against its shards/manifest.json.
    Both are fetched from huggingface.co/AutowareFoundation/auto_e2e on first use.
    --population rule uses t0 in [40, N-51] per scene instead (4 s history, full 5 s
    future), which reproduces the mirror; the script reports any difference.

Coordinates
    Ego poses come from each scene's poses.txt (TUM, UTM frame), interpolated at the
    reference timestamps. Trajectories are in the ego frame at t0: x forward, y left,
    heading positive for a left turn, metres / radians. The forward axis of the KITScenes
    reference frame is detected from the driving direction and reported; it is +x unless
    the manifest says otherwise.

Navigation command (Qwen-Drive input)
    Derived from the ground-truth future: lateral offset at 5 s > +threshold -> TURN LEFT,
    < -threshold -> TURN RIGHT, else GO STRAIGHT (default 2.0 m). This uses the future,
    like the 'oracle command' rows of the published table. Alpamayo gets no command.

Run in a Python environment with the kitscenes package (pip install -e . in the kitscenes
repo) and huggingface_hub:

    export KITSCENES_ROOT=/path/to/KITScenes-Multimodal        # contains data/val/<scene>/
    python e2e_eval/prepare_kitscenes_e2e.py --out-dir data_e2e/kitscenes_val

Outputs in --out-dir:
    kitscenes_val_e2e.jsonl          one record per sample (both models)
    kitscenes_val_eval_index.csv     the 11,035 sample uids (used by both scorers)
    prepare_manifest.json            counts, checks, settings, image root, hashes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EXPECTED_SAMPLES, EXPECTED_SCENES, FUTURE_STEPS, HISTORY_STEPS, HZ, make_uid, sha256_file,
)

# ----------------------------------------------------------------------- constants
MIRROR_REPO = "AutowareFoundation/auto_e2e"
MIRROR_PATH = "datasets/kitscenes-val/v3.5"
MIRROR_FILES = ("shards/manifest.json", "geo/sample_pose.parquet")
VAL_JOB = "kitscenes-v35-val-e0d55e79"
JOB_RE = re.compile(r"/(kitscenes-v35-[a-z]+-[0-9a-f]{8})-.*?-n(\d+)-\d+/")
RULE_FIRST_T0 = 40                     # 4 s history
DATASET_NAME = "KITScenes-val"

# Qwen-Drive: views in record order, KITScenes camera per view, frame offsets (2 Hz).
QWEN_VIEWS = (("<FRONT VIEW>", "camera_ring_front"),
              ("<FRONT LEFT VIEW>", "camera_ring_front_left"),
              ("<FRONT RIGHT VIEW>", "camera_ring_front_right"))
QWEN_FRAME_OFFSETS = (-15, -10, -5, 0)
QWEN_NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")       # nav_command 0, 1, 2
QWEN_INTENTS = ("GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT")
# driving_command one-hot order: left, straight, right, unknown
QWEN_DRIVING_COMMAND = {0: [0, 1, 0, 0], 1: [1, 0, 0, 0], 2: [0, 0, 1, 0]}
QWEN_HISTORY_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
# Same text as qwen_drive.scene.DrivingScene.instruction() for 4 frames at 0.5 s.
QWEN_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains 4 temporal "
    "frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn：\n"
)

# Alpamayo-1.5: camera slot -> KITScenes camera, frame offsets (10 Hz).
#   0 cross_left_120fov, 1 front_wide_120fov, 2 cross_right_120fov, 6 front_tele_30fov
ALPAMAYO_CAMERAS = {0: "camera_ring_front_left", 1: "camera_ring_front",
                    2: "camera_ring_front_right", 6: "camera_base_front_center"}
ALPAMAYO_TELE_SLOT = 6
ALPAMAYO_TELE_HFOV_DEG = 30.0
ALPAMAYO_FRAME_OFFSETS = (-3, -2, -1, 0)


# --------------------------------------------------------------------- utilities
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def r5(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 5).tolist()


# --------------------------------------------------------------- mirror population
def fetch_mirror_meta(meta_dir: Path, revision: Optional[str]) -> Path:
    base = meta_dir / MIRROR_PATH
    missing = [f for f in MIRROR_FILES if not (base / f).exists()]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("pip install huggingface_hub (or use --population rule, or put the mirror "
                     f"files under {base})")
        for f in missing:
            log(f"fetching {MIRROR_REPO}/{MIRROR_PATH}/{f}")
            hf_hub_download(MIRROR_REPO, f"{MIRROR_PATH}/{f}", repo_type="model",
                            revision=revision, local_dir=meta_dir)
    return base


def mirror_population(base: Path, scene_ids: List[str]) -> Tuple[Dict[str, List[int]], List[str]]:
    """{scene_id: sorted t0 frames} from sample_pose.parquet, plus check lines."""
    import pyarrow.parquet as pq

    table = pq.read_table(base / "geo/sample_pose.parquet").to_pydict()
    for col in ("episode_id", "frame_index"):
        if col not in table:
            sys.exit(f"sample_pose.parquet has no column {col!r}; columns: {list(table)}")
    wanted = set(scene_ids)
    frames: Dict[str, List[int]] = defaultdict(list)
    for sid, f in zip(table["episode_id"], table["frame_index"]):
        if sid in wanted:
            frames[sid].append(int(f))
    pop = {s: sorted(set(v)) for s, v in frames.items()}

    manifest = json.loads((base / "shards/manifest.json").read_text())
    val_counts = sorted(int(p["sample_count"]) for p in manifest["partitions"]
                        if (m := JOB_RE.search(p.get("source_uri", ""))) and m.group(1) == VAL_JOB)
    match = sorted(len(v) for v in pop.values()) == val_counts
    checks = [f"mirror sample_pose rows: {len(table['episode_id']):,}; for local val scenes: "
              f"{sum(map(len, pop.values())):,} samples in {len(pop)} scenes",
              f"per-scene sample counts match the {len(val_counts)} '{VAL_JOB}' partitions of the "
              f"mirror manifest: {match}"]
    return pop, checks


def rule_population(num_frames: Dict[str, int]) -> Dict[str, List[int]]:
    return {s: list(range(RULE_FIRST_T0, n - FUTURE_STEPS)) for s, n in num_frames.items()
            if n - FUTURE_STEPS > RULE_FIRST_T0}


# --------------------------------------------------------------------- poses
class ScenePoses:
    """World poses (UTM) interpolated at the reference timestamps of one scene."""

    def __init__(self, scene) -> None:
        ts = np.asarray(scene.timestamps_ns, dtype=np.int64)
        poses = scene.ego_poses
        if len(poses) < 2:
            raise ValueError("fewer than 2 ego poses")
        pt = np.array([p.timestamp_ns for p in poses], dtype=np.int64)
        order = np.argsort(pt)
        pt = pt[order]
        trans = np.array([poses[i].translation for i in order], dtype=np.float64)
        quats = np.array([poses[i].rotation for i in order], dtype=np.float64)   # qx qy qz qw
        keep = np.concatenate([[True], np.diff(pt) > 0])
        pt, trans, quats = pt[keep], trans[keep], quats[keep]
        t_s, pt_s = ts / 1e9, pt / 1e9
        self.valid = (t_s >= pt_s[0] - 1e-3) & (t_s <= pt_s[-1] + 1e-3)
        tq = np.clip(t_s, pt_s[0], pt_s[-1])
        self.timestamps_ns = ts
        self.p = np.stack([np.interp(tq, pt_s, trans[:, i]) for i in range(3)], axis=-1)   # (T, 3)
        self.R = Slerp(pt_s, Rotation.from_quat(quats))(tq).as_matrix()                  # (T, 3, 3)

    def forward_angle(self, min_speed: float = 2.0) -> Tuple[Optional[float], int, float]:
        """Median direction of travel in the body frame (rad), number of moving frames,
        and median z-axis alignment with world up."""
        v_world = (self.p[2:] - self.p[:-2]) / (2.0 / HZ)
        v_body = np.einsum("tji,tj->ti", self.R[1:-1], v_world)      # R^T v
        speed = np.linalg.norm(v_body[:, :2], axis=1)
        moving = (speed > min_speed) & self.valid[1:-1]
        up = float(np.median(self.R[:, 2, 2]))
        if moving.sum() < 10:
            return None, int(moving.sum()), up
        ang = np.arctan2(v_body[moving, 1], v_body[moving, 0])
        return float(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))), int(moving.sum()), up


def relative(poses: ScenePoses, t0: int, frames: np.ndarray, r_fix: np.ndarray):
    """Positions (n, 3), rotations (n, 3, 3) and yaw (n,) of `frames` in the ego frame at t0.
    r_fix rotates the KITScenes body frame so that x is forward."""
    R0 = poses.R[t0] @ r_fix.T
    Rk = poses.R[frames] @ r_fix.T
    xyz = (poses.p[frames] - poses.p[t0]) @ R0                       # R0^T (p_k - p_0)
    rot = np.einsum("ji,njk->nik", R0, Rk)                           # R0^T R_k
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    return xyz, rot, yaw


# ------------------------------------------------------------------- record build
def qwen_instruction(hist: np.ndarray, nav: int) -> str:
    lines = "".join(" -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *hist[i * 5])
                    for i, label in enumerate(QWEN_HISTORY_LABELS))
    return f"{QWEN_INSTRUCTION_HEADER}{lines}2. Active navigation command: [{QWEN_NAV_COMMANDS[nav]}]"


def resized(size: Tuple[int, int], height: int) -> Tuple[int, int]:
    w, h = size
    return int(round(height * w / h)), int(height)


def build_record(scene_id: str, t0: int, poses: ScenePoses, r_fix: np.ndarray, scene_rel: str,
                 cam_sizes: Dict[str, Tuple[int, int]], tele_crop: dict, args) -> dict:
    T = len(poses.timestamps_ns)
    hist_frames = np.arange(t0 - HISTORY_STEPS + 1, t0 + 1)
    fut_frames = np.arange(t0 + 1, t0 + FUTURE_STEPS + 1)
    hist_valid = (hist_frames >= 0) & poses.valid[np.clip(hist_frames, 0, T - 1)]
    fut_valid = (fut_frames < T) & poses.valid[np.clip(fut_frames, 0, T - 1)]

    h_xyz, h_rot, h_yaw = relative(poses, t0, np.clip(hist_frames, 0, T - 1), r_fix)
    f_xyz, _, f_yaw = relative(poses, t0, np.clip(fut_frames, 0, T - 1), r_fix)
    hist = np.column_stack([h_xyz[:, :2], h_yaw])
    fut = np.column_stack([f_xyz[:, :2], f_yaw])
    vel = np.gradient(hist[:, :2], 1.0 / HZ, axis=0)
    acc = np.gradient(vel, 1.0 / HZ, axis=0)

    last = int(np.flatnonzero(fut_valid)[-1]) if fut_valid.any() else 0
    lateral = float(fut[last, 1])
    nav = 1 if lateral > args.command_threshold else 2 if lateral < -args.command_threshold else 0

    content: List[dict] = []
    for tag, cam in QWEN_VIEWS:
        content.append({"text": tag})
        for i, off in enumerate(QWEN_FRAME_OFFSETS):
            w, h = resized(cam_sizes[cam], args.current_height if off == 0 else args.history_height)
            content.append({"text": f"frame: {i}"})
            content.append({"image": f"{scene_rel}/{cam}/{max(t0 + off, 0):010d}.jpg",
                            "resized_width": w, "resized_height": h})
    # Format the history exactly as Qwen-Drive would from the stored values (rounded, float32),
    # so the prompt equals DrivingScene.instruction() for this record.
    content.append({"text": qwen_instruction(np.asarray(r5(hist), dtype=np.float32), nav)})

    alp_cams: Dict[str, Dict[str, Any]] = {}
    for slot, cam in ALPAMAYO_CAMERAS.items():
        entry: Dict[str, Any] = {"camera": cam,
                 "images": [f"{scene_rel}/{cam}/{max(t0 + off, 0):010d}.jpg" for off in ALPAMAYO_FRAME_OFFSETS]}
        if slot == ALPAMAYO_TELE_SLOT:
            entry["crop"] = tele_crop
        alp_cams[str(slot)] = entry

    uid = make_uid(scene_id, t0)
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
        "meta_info": {
            "dataset": DATASET_NAME, "task_name": "trajectory_prediction",
            "token": uid, "scene_token": scene_id, "frame_idx": int(t0),
            "t0_timestamp_ns": int(poses.timestamps_ns[t0]),
            "intent": QWEN_INTENTS[nav], "nav_command": nav,
            "driving_command": QWEN_DRIVING_COMMAND[nav],
            "cam_order": [tag for tag, _ in QWEN_VIEWS],
        },
        "trajectory": {
            "hist_traj_10hz": r5(hist), "hist_vel_10hz": r5(vel), "hist_acc_10hz": r5(acc),
            "hist_valid_mask_10hz": hist_valid.tolist(),
            "future_traj_10hz": r5(fut), "future_valid_mask_10hz": fut_valid.tolist(),
            "ego_status": {"in_global_frame": False, "ego_pose": [0.0, 0.0, 0.0],
                           "ego_velocity": r5(vel[-1]), "ego_acceleration": r5(acc[-1]),
                           "driving_command": QWEN_DRIVING_COMMAND[nav]},
            "nav_command": nav,
        },
        "alpamayo": {
            "cameras": alp_cams, "frame_offsets": list(ALPAMAYO_FRAME_OFFSETS),
            "ego_history_xyz": r5(h_xyz), "ego_history_rot": r5(h_rot),
        },
    }


def tele_crop_params(loader, cam: str) -> dict:
    """Centre crop of `cam` (pinhole) down to a 30 deg horizontal FOV, centred on the
    principal point. width_frac = tan(15 deg) / tan(hfov / 2)."""
    calib = loader.get_camera_calibration(cam)
    f, cu, cv = calib.intrinsic[0, 0], calib.intrinsic[0, 2], calib.intrinsic[1, 2]
    w, h = calib.image_size if calib.image_size else loader.get_camera_image_size(cam, 0)
    hfov = 2.0 * math.degrees(math.atan(w / (2.0 * f)))
    frac = 1.0 if hfov <= ALPAMAYO_TELE_HFOV_DEG else (
        math.tan(math.radians(ALPAMAYO_TELE_HFOV_DEG / 2)) / math.tan(math.radians(hfov / 2)))
    return {"source_hfov_deg": round(hfov, 2), "target_hfov_deg": ALPAMAYO_TELE_HFOV_DEG,
            "width_frac": round(frac, 5), "center_u": round(cu / w, 5), "center_v": round(cv / h, 5)}


def camera_size(loader, cam: str) -> Tuple[int, int]:
    calib = loader.get_camera_calibration(cam)
    if calib.image_size:
        return calib.image_size
    return loader.get_camera_image_size(cam, 0)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitscenes-root", type=Path, default=os.environ.get("KITSCENES_ROOT"),
                    help="KITScenes-Multimodal root containing data/val (default: $KITSCENES_ROOT)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", type=Path, default=Path("data_e2e/kitscenes_val"))
    ap.add_argument("--population", choices=("mirror", "rule"), default="mirror",
                    help="t0 frames from the AutoE2E mirror (default) or the rule t0 in [40, N-51]")
    ap.add_argument("--mirror-meta-dir", type=Path, default=None,
                    help="cache for the mirror metadata (default: <out-dir>/auto_e2e_meta)")
    ap.add_argument("--mirror-revision", default=None, help="pin the auto_e2e commit")
    ap.add_argument("--command-threshold", type=float, default=2.0,
                    help="lateral offset (m) of the 5 s ground truth for TURN LEFT / RIGHT")
    ap.add_argument("--history-height", type=int, default=320, help="Qwen-Drive history frame height")
    ap.add_argument("--current-height", type=int, default=720, help="Qwen-Drive current frame height")
    ap.add_argument("--forward-axis-deg", type=float, default=None,
                    help="override the detected forward direction of the KITScenes body frame "
                         "(0 = +x, 90 = +y, ...)")
    ap.add_argument("--limit-scenes", type=int, default=0, help="debug: only the first N scenes")
    ap.add_argument("--no-strict", action="store_true", help="do not fail on count mismatches")
    args = ap.parse_args()

    if args.kitscenes_root is None:
        sys.exit("set --kitscenes-root or $KITSCENES_ROOT")
    try:
        from kitscenes import KITScenesDataset
    except ImportError:
        sys.exit("the kitscenes package is required: pip install -e <path to kitscenes repo>")

    ds = KITScenesDataset(root=args.kitscenes_root, split=args.split)
    scene_ids = ds.scene_ids[: args.limit_scenes] if args.limit_scenes else ds.scene_ids
    log(f"{ds}: using {len(scene_ids)} scenes")
    checks: List[str] = [f"local {args.split} scenes: {len(scene_ids)} (expected {EXPECTED_SCENES})"]

    # ---- pass 1: poses, frame counts, forward axis
    poses: Dict[str, ScenePoses] = {}
    failed: Dict[str, str] = {}
    angles, ups = [], []
    for sid in scene_ids:
        try:
            p = ScenePoses(ds.get_scene(sid))
        except Exception as e:                                              # noqa: BLE001
            failed[sid] = f"{type(e).__name__}: {e}"
            continue
        poses[sid] = p
        ang, n_moving, up = p.forward_angle()
        ups.append(up)
        if ang is not None:
            angles.append((sid, ang))
    for sid, err in failed.items():
        log(f"WARNING: scene {sid} skipped: {err}")

    if args.forward_axis_deg is not None:
        fwd = math.radians(args.forward_axis_deg)
        checks.append(f"forward axis: {args.forward_axis_deg:g} deg (--forward-axis-deg)")
    else:
        if not angles:
            sys.exit("cannot detect the forward axis (no moving scenes); pass --forward-axis-deg")
        snapped = [round(math.degrees(a) / 90.0) % 4 * 90 for _, a in angles]
        fwd_deg, votes = Counter(snapped).most_common(1)[0]
        fwd = math.radians(fwd_deg)
        spread = [abs((math.degrees(a) - fwd_deg + 180) % 360 - 180) for _, a in angles]
        checks.append(f"forward axis of the KITScenes body frame: {fwd_deg} deg from +x "
                      f"({votes}/{len(angles)} moving scenes agree, median deviation "
                      f"{float(np.median(spread)):.1f} deg)")
        if fwd_deg != 0:
            log(f"NOTE: KITScenes body frame drives along {fwd_deg} deg; rotating to x-forward")
    r_fix = rot_z(-fwd)
    up = float(np.median(ups)) if ups else float("nan")
    checks.append(f"body z axis vs world up (median cos): {up:.3f}")
    if not up > 0.9:
        log("WARNING: the body z axis does not point up; check the pose convention")

    # ---- population
    num_frames = {sid: len(p.timestamps_ns) for sid, p in poses.items()}
    rule = rule_population(num_frames)
    if args.population == "mirror":
        meta_dir = args.mirror_meta_dir or (args.out_dir / "auto_e2e_meta")
        population, mchecks = mirror_population(fetch_mirror_meta(meta_dir, args.mirror_revision),
                                                list(poses))
        checks += mchecks
        diff = [s for s in poses if population.get(s, []) != rule.get(s, [])]
        checks.append(f"rule t0 in [{RULE_FIRST_T0}, N-51] reproduces the mirror for "
                      f"{len(poses) - len(diff)}/{len(poses)} scenes"
                      + (f" (differs: {diff[:5]})" if diff else ""))
    else:
        population = rule
    n_pop = sum(len(v) for v in population.values())
    checks.append(f"population: {n_pop:,} samples in {len(population)} scenes "
                  f"(expected {EXPECTED_SAMPLES:,} in {EXPECTED_SCENES})")
    log(checks[-1])

    # ---- pass 2: records
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = args.out_dir / f"kitscenes_{args.split}_e2e.jsonl"
    index_path = args.out_dir / f"kitscenes_{args.split}_eval_index.csv"
    image_root: Optional[Path] = None
    written = 0
    short_future = 0
    missing_images: List[str] = []
    nav_counts: Counter[str] = Counter()
    index_rows = []
    with open(scenes_path, "w") as out:
        for n, sid in enumerate(sorted(population), 1):
            loader = ds.get_sensor_loader(sid)
            scene_path = Path(loader.scene_path)
            root_here = scene_path.parent.parent
            if image_root is None:
                image_root = root_here
            elif root_here != image_root:
                sys.exit(f"scenes live under different roots: {image_root} vs {root_here}")
            scene_rel = scene_path.relative_to(image_root).as_posix()
            cams = {c for _, c in QWEN_VIEWS} | set(ALPAMAYO_CAMERAS.values())
            try:
                cam_sizes = {c: camera_size(loader, c) for c in cams}
                tele = tele_crop_params(loader, ALPAMAYO_CAMERAS[ALPAMAYO_TELE_SLOT])
            except (KeyError, FileNotFoundError) as e:
                log(f"WARNING: scene {sid} skipped: calibration: {e}")
                continue
            on_disk = {c: set(os.listdir(scene_path / c)) if (scene_path / c).is_dir() else set()
                       for c in cams}
            p = poses[sid]
            for t0 in population[sid]:
                if not 0 <= t0 < len(p.timestamps_ns):
                    missing_images.append(f"{make_uid(sid, t0)} (t0 outside the scene)")
                    continue
                rec = build_record(sid, t0, p, r_fix, scene_rel, cam_sizes, tele, args)
                needed = [i["image"] for i in rec["messages"][0]["content"] if "image" in i]
                needed += [img for c in rec["alpamayo"]["cameras"].values() for img in c["images"]]
                absent = [x for x in needed
                          if Path(x).name not in on_disk[Path(x).parent.name]]
                if absent:
                    missing_images.append(f"{rec['meta_info']['token']} ({len(absent)} images, "
                                          f"e.g. {absent[0]})")
                    continue
                if not all(rec["trajectory"]["future_valid_mask_10hz"]):
                    short_future += 1
                nav_counts[rec["meta_info"]["intent"]] += 1
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                index_rows.append((rec["meta_info"]["token"], sid, t0, rec["meta_info"]["t0_timestamp_ns"]))
                written += 1
            if n % 10 == 0 or n == len(population):
                log(f"  {n}/{len(population)} scenes, {written:,} samples")

    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_uid", "scene_uuid", "frame_index", "t0_timestamp_ns"])
        w.writerows(index_rows)

    checks.append(f"written: {written:,} samples in {len({r[1] for r in index_rows})} scenes")
    if missing_images:
        checks.append(f"NOT written (missing images or frames): {len(missing_images):,}, "
                      f"e.g. {missing_images[:3]}")
    checks.append(f"samples with a future shorter than 5 s: {short_future:,}")
    checks.append(f"navigation commands: {dict(nav_counts)} (threshold {args.command_threshold} m)")

    ok = written == EXPECTED_SAMPLES and len({r[1] for r in index_rows}) == EXPECTED_SCENES
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "population_source": args.population,
        "kitscenes_root": str(Path(args.kitscenes_root).resolve()),
        "image_root": str(image_root.resolve()) if image_root else None,
        "scenes_file": scenes_path.name, "index_file": index_path.name,
        "samples": written, "scenes": len({r[1] for r in index_rows}),
        "expected": {"samples": EXPECTED_SAMPLES, "scenes": EXPECTED_SCENES},
        "full_population": ok,
        "settings": {"command_threshold_m": args.command_threshold,
                     "history_height": args.history_height, "current_height": args.current_height,
                     "forward_axis_deg": round(math.degrees(fwd), 2),
                     "qwen_views": dict(QWEN_VIEWS), "qwen_frame_offsets": QWEN_FRAME_OFFSETS,
                     "alpamayo_cameras": {str(k): v for k, v in ALPAMAYO_CAMERAS.items()},
                     "alpamayo_frame_offsets": ALPAMAYO_FRAME_OFFSETS,
                     "mirror": {"repo": MIRROR_REPO, "path": MIRROR_PATH,
                                "revision": args.mirror_revision} if args.population == "mirror" else None},
        "checks": checks,
        "missing": missing_images[:200],
        "skipped_scenes": failed,
        "sha256": {"scenes_file": sha256_file(scenes_path), "index_file": sha256_file(index_path)},
    }
    (args.out_dir / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"scene file : {scenes_path}")
    print(f"index      : {index_path}")
    print(f"image root : {image_root}")
    print(f"samples    : {written:,} in {manifest['scenes']} scenes "
          f"({'full population' if ok else 'NOT the full 117 / 11,035 population'})")
    for c in checks:
        log(f"  - {c}")
    return 0 if ok or args.no_strict else 2


if __name__ == "__main__":
    sys.exit(main())
