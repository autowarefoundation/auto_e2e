#!/usr/bin/env python3
"""Run Alpamayo-1.5 on KITScenes val (117 scenes, 11,035 samples) and compute statistics.

Uses the scene file of prepare_kitscenes_e2e.py, so Alpamayo is scored on the same samples,
the same t0 and the same ground truth as Qwen-Drive-1.0 (eval_qwendrive_kitscenes.py).

Inputs per sample (from the record's "alpamayo" block)
    cameras  slot 0 cross-left  <- camera_ring_front_left
             slot 1 front-wide  <- camera_ring_front
             slot 2 cross-right <- camera_ring_front_right
             slot 6 front-tele  <- camera_base_front_center, centre crop to 30 deg HFOV
                                   around the principal point (KITScenes has no tele camera)
             4 frames each at 10 Hz (t0-0.3 s .. t0), resized/cropped to 1080x1920
    ego history  16 poses at 10 Hz (xyz + 3x3 rotation) in the t0 ego frame
    no navigation command
Output: 64 waypoints (6.4 s); the first 50 (5 s) are scored.

Run inside the Alpamayo 1.5 venv (https://github.com/NVlabs/alpamayo1.5):

    python e2e_eval/eval_alpamayo15_kitscenes.py \
        --data-dir data_e2e/kitscenes_val \
        --out outputs/alpamayo15 --num-samples 6

    # several GPUs: one process per GPU, then score once
    CUDA_VISIBLE_DEVICES=0 python ... --num-shards 4 --shard-index 0 &
    ...
    python e2e_eval/eval_alpamayo15_kitscenes.py --data-dir ... --out outputs/alpamayo15 --report-only

Re-running resumes: samples already in <out>/preds*.jsonl are skipped. A folder made with a
different camera setup is refused. stdout is only the table:

    Model           Selection         1s             2s             3s             5s
    Alpamayo-1.5    top-1 (sample 0)  ADE / FDE      ...
    Alpamayo-1.5    best of 3         ADE / FDE      ...

Problems go to stderr as WARNING lines; the full checks are in <out>/report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from accuracy_benchmark.kitscenes.common import (  # noqa: E402
    DEFAULT_BEST_OF, add_mirror_arg, evaluate, load_ground_truths, load_index, normalize_uid,
    resolve_data,
)

MODEL = "Alpamayo-1.5"
HF_MODEL = "nvidia/Alpamayo-1.5-10B"
IMG_HW = (1080, 1920)        # Alpamayo input size; its processor downsamples to 320x576
TELE_SLOT = "6"
torch = None                 # imported lazily so --report-only / --check-frames need no torch


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _torch():
    global torch
    if torch is None:
        import torch as _t
        torch = _t
    return torch


# --------------------------------------------------------------------------- data
def load_records(scenes: Path, wanted: List[str]) -> Dict[str, dict]:
    want = set(wanted)
    recs = {}
    with open(scenes) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = normalize_uid(rec["meta_info"]["token"])
            if uid in want:
                if "alpamayo" not in rec:
                    sys.exit(f"{scenes}: records have no 'alpamayo' block; "
                             "re-create the scene file with prepare_kitscenes_e2e.py")
                recs[uid] = rec
    return recs


def camera_setup(rec: dict, no_tele: bool) -> Dict[str, dict]:
    cams = {s: c for s, c in rec["alpamayo"]["cameras"].items()}
    if no_tele:
        cams.pop(TELE_SLOT, None)
    return dict(sorted(cams.items(), key=lambda kv: int(kv[0])))


def camera_config(rec: dict, no_tele: bool) -> dict:
    """Stored with every prediction so result folders never mix camera setups."""
    cfg = {s: c["camera"] for s, c in camera_setup(rec, no_tele).items()}
    if TELE_SLOT in cfg:
        cfg["tele_width_frac"] = rec["alpamayo"]["cameras"][TELE_SLOT]["crop"]["width_frac"]
    return cfg


def _crop(img, crop: Optional[dict]):
    """Narrow-FOV crop centred on the principal point, 16:9, clipped to the image."""
    if not crop or crop["width_frac"] >= 1.0:
        return img
    th, tw = IMG_HW
    cw = img.width * crop["width_frac"]
    ch = min(img.height, cw * th / tw)
    cx, cy = img.width * crop["center_u"], img.height * crop["center_v"]
    left = min(max(cx - cw / 2, 0), img.width - cw)
    top = min(max(cy - ch / 2, 0), img.height - ch)
    return img.crop((round(left), round(top), round(left + cw), round(top + ch)))


def load_image(path: Path, crop: Optional[dict] = None):
    """RGB -> optional tele crop -> resize to cover 1080x1920 -> centre crop -> (3, H, W) uint8."""
    from PIL import Image
    t = _torch()
    img = _crop(Image.open(path).convert("RGB"), crop)
    th, tw = IMG_HW
    s = max(tw / img.width, th / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.BICUBIC)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return t.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def image_paths(rec: dict, image_root: Path, no_tele: bool) -> List[Path]:
    return [image_root / p for c in camera_setup(rec, no_tele).values() for p in c["images"]]


def build_inputs(rec: dict, image_root: Path, no_tele: bool) -> dict:
    t = _torch()
    cams = camera_setup(rec, no_tele)
    frames = t.stack([
        t.stack([load_image(image_root / p, c.get("crop")) for p in c["images"]]) for c in cams.values()
    ])                                                                    # (N_cam, 4, 3, H, W)
    xyz = np.asarray(rec["alpamayo"]["ego_history_xyz"], dtype=np.float32)   # (16, 3)
    rot = np.asarray(rec["alpamayo"]["ego_history_rot"], dtype=np.float32)   # (16, 3, 3)
    return {
        "image_frames": frames,
        "camera_indices": t.tensor([int(s) for s in cams], dtype=t.int64),
        "ego_history_xyz": t.from_numpy(xyz)[None, None],                    # (1, 1, 16, 3)
        "ego_history_rot": t.from_numpy(rot)[None, None],                    # (1, 1, 16, 3, 3)
    }


# --------------------------------------------------------------------------- model
def load_model(attn: str):
    t = _torch()
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(HF_MODEL, dtype=t.bfloat16, attn_implementation=attn).to("cuda").eval()
    return model, helper.get_processor(model.tokenizer)


def _rollout(model, data: dict, n: int, seed: int):
    """One model call returning n trajectories: (n, 64, 2) xy and n reasoning strings."""
    t = _torch()
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    with t.autocast("cuda", dtype=t.bfloat16):
        xyz, _rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data, top_p=0.98, temperature=0.6, num_traj_samples=n,
            max_generation_length=256, return_extra=True)
    xy = xyz.float().cpu().numpy()                     # (batch=1, sets=1, n, 64, 3)
    xy = xy.reshape(-1, xy.shape[-2], xy.shape[-1])[:n, :, :2]
    cots = [str(c) for c in np.asarray(extra["cot"], dtype=object).reshape(-1)][:n]
    if len(xy) != n:
        raise RuntimeError(f"asked for {n} trajectories, the model returned {len(xy)}")
    return xy, cots + [""] * (n - len(cots))


def predict(model, processor, inputs: dict, num_samples: int, seed: int, batch_traj: bool):
    """(K, 64, 2) predicted xy in the t0 ego frame, K reasoning strings and the mode used.

    sequential: one trajectory per call (fits 24 GB); sample k uses seed + k.
    batched:    all K in one call (num_traj_samples=K): the images and prompt are processed
                once and the K trajectories are generated in parallel. Needs more GPU memory;
                on out-of-memory this sample falls back to sequential."""
    t = _torch()
    from alpamayo1_5 import helper

    messages = helper.create_message(frames=inputs["image_frames"].flatten(0, 1),
                                     camera_indices=inputs["camera_indices"])
    tokenized = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                              continue_final_message=True, return_dict=True,
                                              return_tensors="pt")
    data = helper.to_device({"tokenized_data": tokenized,
                             "ego_history_xyz": inputs["ego_history_xyz"],
                             "ego_history_rot": inputs["ego_history_rot"]}, "cuda")
    with t.no_grad():
        if batch_traj and num_samples > 1:
            try:
                xy, cots = _rollout(model, data, num_samples, seed)
                return xy, cots, "batched"
            except t.cuda.OutOfMemoryError:
                t.cuda.empty_cache()
        preds, cots = [], []
        for k in range(num_samples):
            xy, c = _rollout(model, data, 1, seed + k)
            preds.append(xy[0])
            cots.append(c[0])
    return np.stack(preds), cots, "sequential"


# --------------------------------------------------------------------------- predictions
def read_predictions(out_dir: Path) -> Dict[str, dict]:
    done = {}
    for path in sorted(out_dir.glob("preds*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if not line.endswith("\n"):
                    break                                 # incomplete last line of a killed run
                r = json.loads(line)
                uid = normalize_uid(r["token"])
                if uid and uid not in done:
                    done[uid] = r
    return done


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="prepare_kitscenes_e2e.py output dir (scene file, index, image root)")
    ap.add_argument("--scenes", type=Path, default=None)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="directory the image paths are relative to (default: from prepare_manifest.json)")
    ap.add_argument("--out", type=Path, default=Path("outputs/alpamayo15"))
    ap.add_argument("--num-samples", type=int, default=6, help="trajectories per sample (>= --best-of)")
    ap.add_argument("--best-of", type=int, default=DEFAULT_BEST_OF)
    add_mirror_arg(ap)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-traj", action="store_true",
                    help="generate all --num-samples trajectories in one call (num_traj_samples=K): "
                         "faster, more GPU memory; falls back per sample on out-of-memory. "
                         "Use a new --out: the random draws differ from a run without it")
    ap.add_argument("--attn", default="flash_attention_2", help="or 'sdpa'")
    ap.add_argument("--no-tele", action="store_true", help="3 cameras [0, 1, 2], no synthesized tele")
    ap.add_argument("--num-shards", type=int, default=1, help="split the samples over N processes")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: at most N new samples")
    ap.add_argument("--check-frames", action="store_true", help="only check that all images exist")
    ap.add_argument("--report-only", action="store_true", help="only score existing predictions")
    args = ap.parse_args()

    scenes, index_path, root = resolve_data(args.data_dir, args.scenes, args.index)
    image_root = args.image_root or root
    index = load_index(index_path)
    recs = load_records(scenes, index)
    log(f"index: {len(index):,} samples, {len(recs):,} found in {scenes.name}")
    if not recs:
        sys.exit("no index sample is in the scene file")
    first = recs[next(iter(recs))]
    cams = camera_config(first, args.no_tele)
    log(f"cameras: {cams}")

    if args.check_frames or not args.report_only:
        if image_root is None:
            sys.exit("pass --image-root (or --data-dir with prepare_manifest.json)")

    if args.check_frames:
        assert image_root is not None
        bad = [(uid, p) for uid, r in recs.items() for p in image_paths(r, image_root, args.no_tele)
               if not p.exists()]
        log(f"{len(recs) - len({u for u, _ in bad}):,}/{len(recs):,} samples have all images")
        for uid, p in bad[:10]:
            log(f"  missing: {p} ({uid})")
        if bad:
            return 1
        frames = build_inputs(first, image_root, args.no_tele)["image_frames"]   # decode one sample
        log(f"decoded one sample: image_frames {tuple(frames.shape)} {frames.dtype}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    done = read_predictions(args.out)
    if done:
        old = next(iter(done.values())).get("cameras")
        if old is not None and old != cams:
            sys.exit(f"{args.out} holds predictions made with cameras {old}, now {cams}; use another --out")
        old_batch = bool(next(iter(done.values())).get("batch_traj", False))
        if not args.report_only and old_batch != args.batch_traj:
            sys.exit(f"{args.out} holds predictions made {'with' if old_batch else 'without'} --batch-traj; "
                     f"{'add' if old_batch else 'drop'} it or use another --out")

    if not args.report_only:
        if args.num_samples < 1:
            sys.exit("--num-samples must be at least 1")
        if args.num_samples < args.best_of:
            log(f"--num-samples {args.num_samples} < --best-of {args.best_of}: "
                + ("only top-1 is reported" if args.num_samples == 1 else f"reporting best of {args.num_samples}"))
            args.best_of = args.num_samples
        mine = [u for i, u in enumerate(index) if i % args.num_shards == args.shard_index and u in recs]
        todo = [u for u in mine if u not in done]
        if args.limit:
            todo = todo[: args.limit]
        log(f"shard {args.shard_index}/{args.num_shards}: {len(mine):,} samples, "
            f"{len(mine) - len([u for u in mine if u not in done]):,} done, {len(todo):,} to run")
        if todo:
            import time
            assert image_root is not None
            model, processor = load_model(args.attn)
            name = "preds.jsonl" if args.num_shards == 1 else f"preds.shard{args.shard_index}-of-{args.num_shards}.jsonl"
            fallbacks, t_start = 0, time.time()
            with open(args.out / name, "a") as fh:
                for i, uid in enumerate(todo, 1):
                    rec = recs[uid]
                    xy, cots, mode = predict(model, processor, build_inputs(rec, image_root, args.no_tele),
                                             args.num_samples, args.seed, args.batch_traj)
                    if args.batch_traj and mode != "batched":
                        fallbacks += 1
                        if fallbacks == 1:
                            log("WARNING: out of GPU memory with --batch-traj; this sample (and any "
                                "other that does not fit) was run one trajectory per call")
                    row = {"token": uid, "scene_token": rec["meta_info"]["scene_token"],
                           "cameras": cams, "seed": args.seed, "batch_traj": args.batch_traj,
                           "traj_mode": mode, "pred_xy": np.round(xy, 3).tolist(), "cot": cots}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    done[uid] = row
                    if i % 25 == 0 or i == len(todo):
                        rate = (time.time() - t_start) / i
                        log(f"  [{i}/{len(todo)}] {uid}  {rate:.1f} s/sample, "
                            f"ETA {rate * (len(todo) - i) / 3600:.1f} h"
                            + (f", {fallbacks} out-of-memory fallbacks" if fallbacks else ""))
        if args.num_shards > 1:
            log("shard finished; score all shards with --report-only once every shard is done")
            return 0

    gts = load_ground_truths(scenes, index, args.mirror_gt, args.data_dir)
    # gt = gts[0]
    preds = {uid: np.asarray(r["pred_xy"], dtype=np.float64) for uid, r in done.items()}
    max_k = max((len(p) for p in preds.values()), default=0)
    if args.report_only and 0 < max_k < args.best_of:     # e.g. a folder run with --num-samples 1
        log(f"predictions have at most {max_k} trajectories per sample: "
            + ("only top-1 is reported" if max_k == 1 else f"reporting best of {max_k}"))
        args.best_of = max_k
    n_batched = sum(r.get("traj_mode") == "batched" for r in done.values())
    sampling = (f"Trajectories: {n_batched:,} samples with all trajectories in one call (seed {args.seed}), "
                f"{len(done) - n_batched:,} with one call per trajectory (trajectory k: seed {args.seed} + k)."
                if n_batched else f"One call per trajectory; trajectory k uses seed {args.seed} + k.")
    return evaluate(MODEL, preds, gts, index, args.out, args.best_of,
                    extra_checks=[f"Cameras: {cams}"],
                    notes=["Inputs: 4 cameras x 4 frames at 10 Hz (front-tele synthesized by cropping "
                           "camera_base_front_center to 30 deg), 1.5 s ego history, no navigation command.",
                           sampling])


if __name__ == "__main__":
    sys.exit(main())
