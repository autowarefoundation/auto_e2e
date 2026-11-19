"""Evaluate Alpamayo 1.5 on KITScenes with the same ADE/FDE metrics as Qwen-Drive-1.0.

Idea: reuse the Qwen-Drive scene JSONL you already converted from KITScenes, so both
models are scored on the SAME samples, the SAME t0, and the SAME ground-truth future.
Only the camera input is different (Alpamayo wants other cameras and 0.1 s frame spacing),
so images are fetched separately with the kitscenes API in `load_frames()` below.

Cameras: by default the model gets its standard 4-camera profile [0, 1, 2, 6]
(cross-left, front-wide, cross-right, front-tele). KITScenes has no telephoto camera
(all cameras are ~87-88 deg wide), so slot 6 is synthesized by center-cropping a front
camera to a 30 deg horizontal FOV. The source is camera_base_front_center, the 16 Mpx
long-range camera (88.4 deg), so the crop keeps the most detail. No extra flags needed;
--no-tele gives the old 3-camera setup.

Run inside the Alpamayo 1.5 venv (https://github.com/NVlabs/alpamayo1.5), with the
kitscenes package also installed there, or pre-export the frames from the kitscenes venv.

    python eval_alpamayo15_kitscenes.py \
        --scenes kitscenes_test.jsonl \
        --out results_alpamayo15_4cam \
        --num-samples 6

Output: <out>/preds.jsonl (per-scene predictions + CoT) and metrics.json.
Re-running resumes: scenes already in preds.jsonl are skipped. A results folder made with
a different camera setup is refused, so 3- and 4-camera runs never get mixed.

Scoring a subset (e.g. only the keyframes where AutoE2E had HD map + route), no GPU:

    python eval_alpamayo15_kitscenes.py --scenes kitscenes_val.jsonl \
        --out results_alpamayo15_4cam --report-only \
        --sample-list outputs/autoe2e/metrics_kitscenes-epoch-5/tokens_map_route.txt

The table is then computed only on the listed tokens and written to metrics_subset.json
(metrics_common_subset.json with --common-set); metrics.json is left untouched.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

# torch / alpamayo1_5 are imported lazily in load_model() and the tensor helpers, so
# --list-cameras, --report-only and --check-frames also work in a lighter venv.
torch = None


def _torch():
    global torch
    if torch is None:
        import torch as _t
        torch = _t
    return torch


# ---------------------------------------------------------------------------
# Horizons: identical to the Qwen-Drive evaluation (10 Hz, step n = n * 0.1 s).
# Alpamayo predicts 64 steps (6.4 s); Qwen-Drive only 50 (5 s), so compare on 1-5 s.
# ---------------------------------------------------------------------------
HORIZONS = {"1s": 10, "2s": 20, "3s": 30, "4s": 40, "5s": 50}

# Alpamayo camera slots (index -> meaning) from load_physical_aiavdataset.py.
# Default 4-camera profile the model was trained and tested with: [0, 1, 2, 6].
#   0 cross_left_120fov, 1 front_wide_120fov, 2 cross_right_120fov, 6 front_tele_30fov
# KITScenes cameras (same names as in the Qwen-Drive converter). Slot 6 has no real
# KITScenes counterpart; it is a 30 deg center crop of TELE_SOURCE (see --tele-camera).
TELE_SLOT = 6
TELE_HFOV_DEG = 30.0            # Alpamayo front_tele_30fov
TELE_SOURCE_HFOV_DEG = 88.4     # camera_base_front_center (long-range, 5320x3032)
KIT_CAMERA_FOR_SLOT = {
    0: "camera_ring_front_left",
    1: "camera_ring_front",
    2: "camera_ring_front_right",
    TELE_SLOT: "camera_base_front_center",   # long-range camera, cropped to 30 deg
}
NUM_FRAMES = 4          # frames per camera
FRAME_OFFSETS = (-3, -2, -1, 0)   # KITScenes frames are 10 Hz: t0-0.3 .. t0
IMG_HW = (1080, 1920)   # Alpamayo's input size; the processor downsamples to 320x576 itself
IMAGE_ROOT: Path | None = None    # set from --image-root ($KITSCENES_ROOT/data)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def yaw_to_rot(yaw: np.ndarray) -> np.ndarray:
    """(T,) heading in rad -> (T, 3, 3) rotation about z (x fwd, y left, z up)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.zeros((len(yaw), 3, 3), dtype=np.float32)
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    rot[:, 2, 2] = 1.0
    return rot


def tele_crop_fraction(src_hfov_deg: float, dst_hfov_deg: float = TELE_HFOV_DEG) -> float:
    """Fraction of the image width that a pinhole camera with horizontal FOV src_hfov_deg
    must keep (centered) to show only dst_hfov_deg. 1.0 if the source is already narrow."""
    if src_hfov_deg <= dst_hfov_deg:
        return 1.0
    return math.tan(math.radians(dst_hfov_deg) / 2) / math.tan(math.radians(src_hfov_deg) / 2)


def _crop_center(img: Image.Image, width_frac: float) -> Image.Image:
    """Center crop keeping width_frac of the width, with the target 16:9 aspect (so the
    crop is a narrower FOV, not a stretch). Height is clipped to the image if needed."""
    if width_frac >= 1.0:
        return img
    th, tw = IMG_HW
    cw = img.width * width_frac
    ch = min(img.height, cw * th / tw)
    left, top = (img.width - cw) / 2, (img.height - ch) / 2
    return img.crop((round(left), round(top), round(left + cw), round(top + ch)))


def _frame_path_template(scene: dict) -> tuple[Path, int, int, str]:
    """From an image path already in the record, e.g.
    'test/<scene_id>/camera_ring_front/0000000040.jpg', return
    (scene_dir, file-name zero-pad width, frame number of that file, extension)."""
    root = IMAGE_ROOT
    if root is None:
        raise RuntimeError("IMAGE_ROOT is not set; pass --image-root")
    for item in scene["messages"][0]["content"]:
        if "image" in item:
            rel = Path(item["image"])
            return root / rel.parent.parent, len(rel.stem), int(rel.stem), rel.suffix
    raise ValueError(f"{scene['meta_info']['token']}: record has no image paths")


def _load_image(path: Path, crop_frac: float = 1.0):
    """Read an RGB image, optionally center-crop to a narrower FOV, then resize to cover
    IMG_HW and center-crop -> (3, H, W) uint8 tensor."""
    t = _torch()
    img = Image.open(path).convert("RGB")      # read-only access
    img = _crop_center(img, crop_frac)
    th, tw = IMG_HW
    s = max(tw / img.width, th / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.BICUBIC)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return t.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def _slot_crop(slot: int) -> float:
    return tele_crop_fraction(TELE_SOURCE_HFOV_DEG) if slot == TELE_SLOT else 1.0


def _frame_paths(scene: dict) -> list[tuple[int, list[Path]]]:
    """[(slot, [paths for t0-0.3 .. t0])] in slot order; raises if a frame is missing."""
    kf = int(scene["meta_info"]["frame_idx"])
    scene_dir, pad, _, ext = _frame_path_template(scene)
    out = []
    for slot in sorted(KIT_CAMERA_FOR_SLOT):
        cam_dir = scene_dir / KIT_CAMERA_FOR_SLOT[slot]
        paths = [cam_dir / f"{kf + off:0{pad}d}{ext}" for off in FRAME_OFFSETS]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"missing camera frame (slot {slot}): {p}")
        out.append((slot, paths))
    return out


def _check_paths(scene: dict) -> None:
    """Like load_frames() but only checks that the files exist (fast, no decoding)."""
    _frame_paths(scene)


def load_frames(scene: dict):
    """Return uint8 tensor (N_cams, NUM_FRAMES, 3, H, W), cameras ordered by slot index.

    Frames t0-0.3, t0-0.2, t0-0.1, t0 = KITScenes frames kf-3 .. kf of each camera,
    where kf is the keyframe the Qwen-Drive record was built for. The tele slot is a
    30 deg center crop of its source camera. Never writes anything.
    """
    t = _torch()
    cams = [
        t.stack([_load_image(p, _slot_crop(slot)) for p in paths])
        for slot, paths in _frame_paths(scene)
    ]
    return t.stack(cams)


def build_sample(scene: dict) -> dict:
    """Convert one Qwen-Drive scene record into Alpamayo 1.5 inputs + ground truth."""
    t = _torch()
    traj = scene["trajectory"]
    hist = np.asarray(traj["hist_traj_10hz"], dtype=np.float32)      # (16, 3) x, y, yaw; ends at 0
    fut = np.asarray(traj["future_traj_10hz"], dtype=np.float32)     # (50, 3)
    mask = np.asarray(
        traj.get("future_valid_mask_10hz", np.ones(len(fut))), dtype=bool
    )                                                                # (50,)

    hist_xyz = np.zeros((len(hist), 3), dtype=np.float32)
    hist_xyz[:, :2] = hist[:, :2]                                    # z = 0 (planar history)
    hist_rot = yaw_to_rot(hist[:, 2])

    frames = load_frames(scene)
    cam_idx = t.tensor(sorted(KIT_CAMERA_FOR_SLOT), dtype=t.int64)

    return {
        "image_frames": frames,
        "camera_indices": cam_idx,
        "ego_history_xyz": t.from_numpy(hist_xyz)[None, None],       # (1, 1, 16, 3)
        "ego_history_rot": t.from_numpy(hist_rot)[None, None],       # (1, 1, 16, 3, 3)
        "gt_xy": fut[:, :2],
        "gt_mask": mask,
    }


def camera_config() -> dict[str, str | float]:
    """Camera setup stored with every prediction, to keep result folders consistent."""
    cfg: dict[str, str | float] = {
        str(s): KIT_CAMERA_FOR_SLOT[s] for s in sorted(KIT_CAMERA_FOR_SLOT)
    }
    if TELE_SLOT in KIT_CAMERA_FOR_SLOT:
        cfg["tele_crop_frac"] = round(tele_crop_fraction(TELE_SOURCE_HFOV_DEG), 4)
    return cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(attn: str):
    t = _torch()
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=t.bfloat16, attn_implementation=attn
    ).to("cuda").eval()
    return model, helper.get_processor(model.tokenizer)


def predict(model, processor, sample: dict, num_samples: int, seed: int):
    """Return (K, 64, 2) predicted xy and K chain-of-causation strings.

    Samples one trajectory per call so it fits in 24 GB (RTX 3090);
    num_traj_samples=16 in one call needs ~40 GB. A 4th camera adds 4 more frames of
    vision tokens; if you hit OOM, lower max_generation_length or use a larger GPU.
    """
    t = _torch()
    from alpamayo1_5 import helper

    messages = helper.create_message(
        frames=sample["image_frames"].flatten(0, 1), camera_indices=sample["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
        },
        "cuda",
    )

    preds, cots = [], []
    with t.no_grad():
        for k in range(num_samples):
            t.cuda.manual_seed_all(seed + k)
            with t.autocast("cuda", dtype=t.bfloat16):
                pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )
            preds.append(pred_xyz.float().cpu().numpy()[0, 0, 0, :, :2])  # (64, 2)
            cots.append(str(extra["cot"][0]))
    return np.stack(preds), cots


# ---------------------------------------------------------------------------
# Metrics: same definitions as for Qwen-Drive
# ---------------------------------------------------------------------------
def scene_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, mask: np.ndarray) -> dict:
    """pred_xy (K, >=50, 2), gt_xy (50, 2), mask (50,). Returns per-horizon errors.

    Same rule as Qwen-Drive's eval_kitscenes.py: a keyframe is scored at horizon T only
    if its ground truth is valid for ALL of the first T*10 steps; otherwise that horizon
    is skipped for this keyframe (so n drops at 4 s / 5 s for short futures).
    ADE@T = mean L2 over steps 1..n; FDE@T = L2 at step n.
    first = sample 1; mean = averaged trajectory; min = best of the K samples.
    """
    pred_xy = np.asarray(pred_xy)[:, : len(gt_xy)]
    err = np.linalg.norm(pred_xy - gt_xy[None], axis=-1)                    # (K, 50)
    err_mean = np.linalg.norm(pred_xy.mean(axis=0) - gt_xy, axis=-1)       # (50,) mean trajectory
    out = {}
    valid = np.asarray(mask, dtype=bool)
    for name, n in HORIZONS.items():
        if n > len(valid) or not valid[:n].all():
            continue          # keyframe not scored at this horizon
        ade_k = err[:, :n].mean(axis=1)
        fde_k = err[:, n - 1]
        out[f"minADE@{name}"] = float(ade_k.min())
        out[f"ADE_first@{name}"] = float(ade_k[0])
        out[f"ADE_mean@{name}"] = float(err_mean[:n].mean())
        out[f"minFDE@{name}"] = float(fde_k.min())
        out[f"FDE_first@{name}"] = float(fde_k[0])
        out[f"FDE_mean@{name}"] = float(err_mean[n - 1])
    return out


def print_table(agg: dict, n_scenes: int, k: int) -> None:
    """Same layout as the Qwen-Drive-1.0 KITScenes report."""
    def v(key):
        return agg.get(key, {}).get("mean", math.nan)

    print(f"\nScenes evaluated: {n_scenes}   K = {k}")
    print("horizon    n |   ADE first  FDE first | ADE mean   FDE mean |  minADE  minFDE")
    for h in HORIZONS:
        n = agg.get(f"ADE_first@{h}", {}).get("n", 0)
        print(f"{h:>7} {n:4d} | {v(f'ADE_first@{h}'):11.3f} {v(f'FDE_first@{h}'):10.3f} |"
              f" {v(f'ADE_mean@{h}'):8.3f} {v(f'FDE_mean@{h}'):10.3f} |"
              f" {v(f'minADE@{h}'):7.3f} {v(f'minFDE@{h}'):7.3f}")


def aggregate(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r["metrics"]})
    agg = {}
    for k in keys:
        vals = [r["metrics"][k] for r in rows if k in r["metrics"]]
        agg[k] = {"mean": float(np.mean(vals)), "n": len(vals)}
    return agg


# ---------------------------------------------------------------------------
def main():
    global IMAGE_ROOT, TELE_SOURCE_HFOV_DEG

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="Qwen-Drive KITScenes scene JSONL")
    ap.add_argument("--out", default="results_alpamayo15_4cam")
    ap.add_argument("--num-samples", type=int, default=6, help="K for minADE/minFDE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="flash_attention_2", help="or 'sdpa'")
    ap.add_argument("--limit", type=int, default=0, help="only first N scenes (debug)")
    ap.add_argument("--image-root", default=os.environ.get("KITSCENES_ROOT", "") + "/data",
                    help="same --image-root as for Qwen-Drive ($KITSCENES_ROOT/data)")
    ap.add_argument("--tele-camera", default=KIT_CAMERA_FOR_SLOT[TELE_SLOT],
                    help="KITScenes camera cropped to 30 deg for Alpamayo slot 6 (front "
                         "tele). Default: camera_base_front_center (long-range camera)")
    ap.add_argument("--tele-source-hfov", type=float, default=TELE_SOURCE_HFOV_DEG,
                    help="horizontal FOV in deg of --tele-camera (88.4 long-range, "
                         "87.1 ring cameras); <= 30 means no crop")
    ap.add_argument("--no-tele", action="store_true",
                    help="old 3-camera setup [0, 1, 2] without slot 6")
    ap.add_argument("--list-cameras", action="store_true",
                    help="print the camera folders of the first scene and exit")
    ap.add_argument("--check-frames", action="store_true",
                    help="only verify all needed frames exist (no GPU), then exit")
    ap.add_argument("--report-only", action="store_true",
                    help="only print the table from existing preds.jsonl (no GPU)")
    ap.add_argument("--common-set", action="store_true",
                    help="score only keyframes with a full 5 s ground truth, so every "
                         "horizon uses the same keyframes")
    ap.add_argument("--sample-list", default=None,
                    help="score only the tokens in this file, one '<scene>-<frame>' per line "
                         "(e.g. AutoE2E's tokens_map_route.txt); writes metrics_subset.json")
    args = ap.parse_args()

    IMAGE_ROOT = Path(args.image_root)
    TELE_SOURCE_HFOV_DEG = args.tele_source_hfov
    if args.no_tele:
        KIT_CAMERA_FOR_SLOT.pop(TELE_SLOT)
    else:
        KIT_CAMERA_FOR_SLOT[TELE_SLOT] = args.tele_camera

    scenes = [json.loads(scene) for scene in open(args.scenes) if scene.strip()]
    if args.limit:
        scenes = scenes[: args.limit]

    if args.list_cameras:
        scene_dir = _frame_path_template(scenes[0])[0]
        print(scene_dir)
        for d in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            files = sorted(d.iterdir())
            print(f"  {d.name:35s} {len(files):5d} files  e.g. {files[0].name if files else '-'}")
        return

    cams = camera_config()
    print(f"cameras: {cams}")

    if args.check_frames:
        bad = 0
        for s in scenes:
            try:
                load_frames(s) if s is scenes[0] else _check_paths(s)
            except FileNotFoundError as e:
                bad += 1
                print(e)
        print(f"{len(scenes) - bad}/{len(scenes)} scenes have all frames")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "preds.jsonl"

    done = {}
    if pred_path.exists():
        for line in pred_path.open():
            r = json.loads(line)
            done[r["token"]] = r

    # Never mix camera setups in one results folder (old rows have no "cameras" key =
    # the previous 3-camera script).
    if done and not args.report_only:
        old = next(iter(done.values())).get("cameras", "3-camera run (no camera info)")
        if old != cams:
            raise SystemExit(f"{pred_path} was made with cameras {old}, now {cams}. "
                             f"Use a different --out.")

    if args.report_only:
        report(scenes, done, out_dir, args.common_set, args.sample_list)
        return

    model, processor = load_model(args.attn)

    with pred_path.open("a") as f:
        for i, scene in enumerate(scenes):
            token = scene["meta_info"]["token"]
            if token in done:
                continue
            sample = build_sample(scene)
            pred_xy, cots = predict(model, processor, sample, args.num_samples, args.seed)
            metrics = scene_metrics(pred_xy, sample["gt_xy"], sample["gt_mask"])
            row = {
                "token": token,
                "scene_token": scene["meta_info"].get("scene_token"),
                "cameras": cams,
                "pred_xy": np.round(pred_xy, 3).tolist(),
                "cot": cots,
                "metrics": metrics,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[token] = row
            msg = ", ".join(
                f"{k}={metrics[k]:.2f}" for k in ("minADE@5s", "minFDE@5s") if k in metrics
            )
            print(f"[{i + 1}/{len(scenes)}] {token}  {msg}")

    report(scenes, done, out_dir, args.common_set, args.sample_list)


def load_sample_list(path: str) -> set[str]:
    """Tokens '<scene>-<frame>' one per line (blank lines and '#' comments ignored)."""
    return {sample.strip() for sample in open(path) if sample.strip() and not sample.startswith("#")}


def report(scenes: list[dict], done: dict, out_dir: Path, common_set: bool = False,
           sample_list: str | None = None) -> None:
    """Recompute all metrics from the stored predictions (no GPU needed).

    common_set: score only keyframes whose ground truth is valid for the longest horizon,
    so every row uses the same keyframes (same as --common-set in eval_kitscenes.py).
    sample_list: score only the tokens listed in this file (e.g. the keyframes where
    AutoE2E had HD map + route); the result goes to *_subset.json.
    """
    n_max = max(HORIZONS.values())
    wanted = None
    if sample_list:
        wanted = load_sample_list(sample_list)
        scene_tokens = {s["meta_info"]["token"] for s in scenes}
        in_scenes = wanted & scene_tokens
        with_pred = {t for t in in_scenes if t in done}
        print(f"sample list {sample_list}: {len(wanted)} tokens, "
              f"{len(in_scenes)} in the scene file, {len(with_pred)} with a prediction")
        if len(in_scenes) < len(wanted):
            print(f"WARNING: {len(wanted) - len(in_scenes)} listed tokens are not in --scenes, "
                  f"e.g. {sorted(wanted - scene_tokens)[:3]}")
        if len(with_pred) < len(in_scenes):
            print(f"WARNING: {len(in_scenes) - len(with_pred)} listed tokens have no prediction "
                  f"yet, e.g. {sorted(in_scenes - with_pred)[:3]}")
    rows, skipped = [], 0
    for s in scenes:
        token = s["meta_info"]["token"]
        if wanted is not None and token not in wanted:
            continue
        r = done.get(token)
        if r is None:
            continue
        traj = s["trajectory"]
        gt = np.asarray(traj["future_traj_10hz"], dtype=np.float32)[:, :2]
        mask = np.asarray(traj.get("future_valid_mask_10hz", np.ones(len(gt))), dtype=bool)
        if common_set and (len(mask) < n_max or not mask[:n_max].all()):
            skipped += 1
            continue
        rows.append({"metrics": scene_metrics(np.asarray(r["pred_xy"]), gt, mask)})
    if common_set:
        print(f"common set: {len(rows)} keyframes with a full {n_max / 10:g} s ground truth "
              f"({skipped} skipped)")
    agg = aggregate(rows)
    name = ("metrics_common" if common_set else "metrics") + ("_subset" if wanted is not None else "")
    (out_dir / f"{name}.json").write_text(json.dumps(agg, indent=2))
    k = len(next(iter(done.values()))["pred_xy"]) if done else 0
    if done:
        print(f"cameras: {next(iter(done.values())).get('cameras', '3-camera run')}")
    print_table(agg, len(rows), k)


if __name__ == "__main__":
    main()
