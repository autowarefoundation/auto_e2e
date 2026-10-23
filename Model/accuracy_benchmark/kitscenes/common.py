"""Shared definitions for the KITScenes val end-to-end planning evaluation.

Used by eval_qwendrive_kitscenes.py and eval_alpamayo15_kitscenes.py so that both models
are scored by exactly the same code, on the same samples, against the same ground truth.

Population
    117 scenes / 11,035 samples of the official KITScenes val split, as packed by the
    AutoE2E v3.5 mirror (build job kitscenes-v35-val-e0d55e79). prepare_kitscenes_e2e.py
    writes the list to kitscenes_val_eval_index.csv.

Metric (same definition as Autoware auto_e2e's KITScenes evaluator)
    10 Hz, ego frame of the keyframe (x forward, y left), positions only.
    ADE@T = mean L2 error over future steps 1..T*10, FDE@T = L2 error at step T*10,
    averaged over samples. Horizons 1, 2, 3, 5 s. A sample is scored at horizon T only if
    its ground truth is valid for all of the first T*10 steps.

Rows
    top-1 (sample 0)  the first predicted trajectory (neither model outputs a confidence)
    best of K         minADE and minFDE over the first K trajectories, each minimised
                      independently (oracle: uses the ground truth). Default K = 3.

Only numpy is needed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HZ = 10
FUTURE_STEPS = 50                   # 5 s at 10 Hz
HISTORY_STEPS = 16                  # 1.5 s at 10 Hz incl. t0
HORIZONS: Tuple[int, ...] = (1, 2, 3, 5)
DEFAULT_BEST_OF = 3
EXPECTED_SCENES = 117
EXPECTED_SAMPLES = 11_035

_UID = re.compile(r"^kitscenes-v1-(?P<scene>[0-9a-f-]{36})-f(?P<frame>\d+)$")
_SHORT = re.compile(r"^(?P<scene>[0-9a-f-]{36})[-,:](?P<frame>\d+)$")


# --------------------------------------------------------------------------- tokens
def make_uid(scene_id: str, frame_idx: int) -> str:
    """Canonical sample id, identical to the AutoE2E mirror: kitscenes-v1-<scene>-f000040."""
    return f"kitscenes-v1-{scene_id}-f{int(frame_idx):06d}"


def normalize_uid(text) -> Optional[str]:
    """Accepts 'kitscenes-v1-<scene>-f000040', '<scene>-40', '<scene>,40', '<scene>:40'."""
    text = str(text).strip().strip('"')
    for pattern in (_UID, _SHORT):
        m = pattern.match(text)
        if m:
            return make_uid(m["scene"], int(m["frame"]))
    return None


def parse_uid(uid: str) -> Tuple[str, int]:
    m = _UID.match(uid)
    if not m:
        raise ValueError(f"not a sample uid: {uid!r}")
    return m["scene"], int(m["frame"])


def load_index(path: Path) -> List[str]:
    """Sample uids from kitscenes_val_eval_index.csv (column sample_uid), in file order.
    Also accepts a JSON list or one token per line."""
    raw = Path(path).read_text()
    items: Optional[list] = None
    try:
        payload = json.loads(raw)
        items = payload["sample_uids"] if isinstance(payload, dict) else list(payload)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    if items is None:
        first = raw.splitlines()[0] if raw.strip() else ""
        if "sample_uid" in first.split(","):
            items = [r["sample_uid"] for r in csv.DictReader(raw.splitlines())]
        else:
            items = [line for line in raw.splitlines() if line.strip() and not line.startswith("#")]
    uids, bad = [], []
    for item in items:
        uid = normalize_uid(item)
        (uids.append(uid) if uid else bad.append(item))
    if bad:
        sys.exit(f"could not parse {len(bad)} entries of {path}, e.g. {bad[:3]}")
    if len(set(uids)) != len(uids):
        sys.exit(f"{path} contains duplicate samples")
    return uids


def resolve_data(data_dir: Optional[Path], scenes: Optional[Path], index: Optional[Path]
                 ) -> Tuple[Path, Path, Optional[Path]]:
    """(scene file, index csv, image root) from prepare_kitscenes_e2e.py's output dir,
    with explicit --scenes / --index taking precedence."""
    manifest = {}
    if data_dir is not None:
        mpath = Path(data_dir) / "prepare_manifest.json"
        if not mpath.exists():
            sys.exit(f"{mpath} not found -- is --data-dir the prepare_kitscenes_e2e.py output?")
        manifest = json.loads(mpath.read_text())
        scenes = scenes or Path(data_dir) / manifest["scenes_file"]
        index = index or Path(data_dir) / manifest["index_file"]
    if scenes is None or index is None:
        sys.exit("pass --data-dir, or both --scenes and --index")
    root = manifest.get("image_root")
    return Path(scenes), Path(index), Path(root) if root else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- ground truth
class GroundTruth:
    """Future trajectory, validity mask and t0 velocity per sample.

    Two sources, both in the ego frame at t0 (x forward, y left), 10 Hz, t0+0.1 s first:
      poses     from the scene file of prepare_kitscenes_e2e.py (KITScenes poses.txt)
      mirror    trajectory_xy_m of the AutoE2E v3.5 mirror (extract_mirror_gt.py)
    The t0 velocity (for the constant-velocity reference) always comes from the scene file,
    because it is a model input, not part of the ground truth."""

    def __init__(self, key: str, label: str) -> None:
        self.key, self.label = key, label
        self.future: Dict[str, np.ndarray] = {}     # uid -> (50, 2)
        self.valid: Dict[str, np.ndarray] = {}      # uid -> (50,) bool
        self.velocity: Dict[str, np.ndarray] = {}   # uid -> (2,) vx, vy at t0

    @classmethod
    def from_scene_file(cls, path: Path, wanted: Optional[Iterable[str]] = None) -> "GroundTruth":
        wanted_set = set(wanted) if wanted is not None else None
        gt = cls("poses", "poses.txt (KITScenes ego poses)")
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = normalize_uid(rec["meta_info"]["token"])
                if uid is None or (wanted_set is not None and uid not in wanted_set):
                    continue
                traj = rec["trajectory"]
                fut = np.asarray(traj["future_traj_10hz"], dtype=np.float64)[:, :2]
                gt.future[uid] = fut
                gt.valid[uid] = np.asarray(
                    traj.get("future_valid_mask_10hz", np.ones(len(fut))), dtype=bool)
                gt.velocity[uid] = np.asarray(traj["ego_status"]["ego_velocity"][:2], dtype=np.float64)
        return gt

    @classmethod
    def from_mirror_file(cls, path: Path, velocity_from: "GroundTruth",
                         wanted: Optional[Iterable[str]] = None) -> "GroundTruth":
        """kitscenes_val_gt_mirror.jsonl from extract_mirror_gt.py; first 50 steps (5 s) used."""
        wanted_set = set(wanted) if wanted is not None else None
        gt = cls("mirror", "trajectory_xy_m (AutoE2E v3.5 mirror)")
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = normalize_uid(rec["sample_uid"])
                if uid is None or (wanted_set is not None and uid not in wanted_set):
                    continue
                xy = np.asarray(rec["trajectory_xy_m"], dtype=np.float64).reshape(-1, 2)[:FUTURE_STEPS]
                valid = np.asarray(rec["valid"], dtype=bool)[:FUTURE_STEPS]
                if len(xy) < FUTURE_STEPS:                       # pad a short path as invalid
                    pad = FUTURE_STEPS - len(xy)
                    xy = np.vstack([xy, np.zeros((pad, 2))])
                    valid = np.concatenate([valid, np.zeros(pad, dtype=bool)])
                gt.future[uid], gt.valid[uid] = xy, valid
                if uid in velocity_from.velocity:
                    gt.velocity[uid] = velocity_from.velocity[uid]
        return gt


MIRROR_GT_FILE = "kitscenes_val_gt_mirror.jsonl"


def add_mirror_arg(ap) -> None:
    ap.add_argument("--mirror-gt", default="auto",
                    help="kitscenes_val_gt_mirror.jsonl from extract_mirror_gt.py, to also score against "
                         "the mirror's trajectory_xy_m. 'auto' (default): use <data-dir>/"
                         f"{MIRROR_GT_FILE} if it exists; 'none': poses.txt only")


def load_ground_truths(scenes: Path, index: List[str], mirror_gt: str,
                       data_dir: Optional[Path]) -> List[GroundTruth]:
    """[poses.txt ground truth, + mirror ground truth if requested / found]."""
    gts = [GroundTruth.from_scene_file(scenes, index)]
    if mirror_gt == "none":
        return gts
    if mirror_gt == "auto":
        path = Path(data_dir) / MIRROR_GT_FILE if data_dir is not None else None
        if path is None or not path.exists():
            print(f"note: no {MIRROR_GT_FILE} in --data-dir; scoring against poses.txt only "
                  "(run extract_mirror_gt.py to add trajectory_xy_m)", file=sys.stderr)
            return gts
    else:
        path = Path(mirror_gt)
        if not path.exists():
            sys.exit(f"--mirror-gt {path} not found")
    gts.append(GroundTruth.from_mirror_file(path, gts[0], index))
    return gts


def compare_ground_truths(a: GroundTruth, b: GroundTruth) -> str:
    """Mean |a - b| at each horizon, over samples present and valid in both."""
    parts = []
    common = set(a.future) & set(b.future)
    for sec in HORIZONS:
        i = sec * HZ - 1
        d = [np.linalg.norm(a.future[u][i] - b.future[u][i]) for u in common
             if a.valid[u][: i + 1].all() and b.valid[u][: i + 1].all()]
        if d:
            parts.append(f"{sec}s {np.mean(d):.3f} (max {np.max(d):.3f})")
    return (f"Ground truths compared, mean |{a.key} - {b.key}| in metres over {len(common):,} samples: "
            + ", ".join(parts))


# --------------------------------------------------------------------- scoring
def score_sample(trajs: np.ndarray, gt: np.ndarray, valid: np.ndarray, best_of: int) -> Dict[str, float]:
    """trajs (K, >=50, >=2), gt (50, 2), valid (50,) -> {'ADE_1s_first': .., ...}."""
    trajs = np.asarray(trajs, dtype=np.float64)
    if trajs.ndim == 2:
        trajs = trajs[None]
    n = min(trajs.shape[1], gt.shape[0])
    err = np.linalg.norm(trajs[:, :n, :2] - gt[None, :n, :2], axis=-1)     # (K, n)
    out: Dict[str, float] = {}
    for sec in HORIZONS:
        steps = sec * HZ
        if steps > n or not valid[:steps].all():
            continue
        ade, fde = err[:, :steps].mean(axis=1), err[:, steps - 1]
        out[f"ADE_{sec}s_first"], out[f"FDE_{sec}s_first"] = float(ade[0]), float(fde[0])
        if len(ade) >= best_of:
            out[f"ADE_{sec}s_best{best_of}"] = float(ade[:best_of].min())
            out[f"FDE_{sec}s_best{best_of}"] = float(fde[:best_of].min())
        out[f"ADE_{sec}s_min"], out[f"FDE_{sec}s_min"] = float(ade.min()), float(fde.min())
    return out


def constant_velocity(vel: np.ndarray, steps: int = FUTURE_STEPS) -> np.ndarray:
    t = np.arange(1, steps + 1, dtype=np.float64) / HZ
    return np.stack([vel[0] * t, vel[1] * t], axis=-1)[None]               # (1, steps, 2)


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for r in rows for k in r})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        agg[k] = {"mean": float(np.mean(vals)), "n": len(vals)}
    return agg


# --------------------------------------------------------------------- output
def _cell(agg: dict, sec: int, sel: str) -> str:
    a, f = agg.get(f"ADE_{sec}s_{sel}"), agg.get(f"FDE_{sec}s_{sel}")
    return "-" if a is None or f is None else f"{a['mean']:.3f} / {f['mean']:.3f}"


def result_rows(model: str, agg: dict, best_of: int) -> List[Tuple[str, str, List[str]]]:
    rows = [(model, "top-1 (sample 0)", [_cell(agg, h, "first") for h in HORIZONS])]
    if best_of > 1 and any(f"ADE_{h}s_best{best_of}" in agg for h in HORIZONS):
        rows.append((model, f"best of {best_of}", [_cell(agg, h, f"best{best_of}") for h in HORIZONS]))
    return rows


def format_table(rows: List[Tuple[str, str, List[str]]]) -> str:
    lines = ["\t".join(["Model", "Selection"] + [f"{h}s" for h in HORIZONS])]
    lines += ["\t".join([m, s] + cells) for m, s, cells in rows]
    return "\n".join(lines)


def build_checks(agg: dict, cv_agg: dict, requested: int, found: int, k_values: List[int],
                 best_of: int, extra: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """Returns (all check lines, the subset that are problems)."""
    lines, problems = [], []
    pop = f"Population: {found:,} of {requested:,} listed samples scored"
    if not (found == requested == EXPECTED_SAMPLES):
        pop += (f" -- NOT the full {EXPECTED_SAMPLES:,}-sample population; compare models only on "
                "the same samples")
        problems.append(pop)
    lines.append(pop)
    lines.append("Scored per horizon: " + ", ".join(
        f"{h}s {agg.get(f'ADE_{h}s_first', {}).get('n', 0):,}" for h in HORIZONS))
    lines.append(f"Trajectory samples per keyframe: {k_values}")
    if best_of > 1 and k_values and min(k_values) < best_of:
        msg = (f"Best of {best_of}: not computed for keyframes with fewer than {best_of} trajectories "
               f"(run with at least {best_of} samples)")
        lines.append(msg)
        problems.append(msg)
    if "ADE_1s_first" in cv_agg:
        # Reference only: constant velocity (velocity at t0 from the last two history poses)
        # scored against this ground truth -- the floor a model should beat.
        lines.append("Constant-velocity reference: " + ", ".join(
            f"{h}s {cv_agg[f'ADE_{h}s_first']['mean']:.3f} / {cv_agg[f'FDE_{h}s_first']['mean']:.3f}"
            for h in HORIZONS if f"ADE_{h}s_first" in cv_agg))
    lines.extend(extra)
    return lines, problems


def _score_against(predictions: Dict[str, np.ndarray], gt: GroundTruth, index: List[str],
                   best_of: int) -> dict:
    rows, cv_rows, per_sample, k_values, missing_gt = [], [], [], set(), 0
    for uid in index:
        trajs = predictions.get(uid)
        if trajs is None:
            continue
        if uid not in gt.future:
            missing_gt += 1
            continue
        trajs = np.asarray(trajs, dtype=np.float64)
        if trajs.ndim == 2:
            trajs = trajs[None]
        if not np.isfinite(trajs).all():
            continue
        k_values.add(int(trajs.shape[0]))
        m = score_sample(trajs, gt.future[uid], gt.valid[uid], best_of)
        rows.append(m)
        if uid in gt.velocity:
            cv_rows.append(score_sample(constant_velocity(gt.velocity[uid]), gt.future[uid],
                                        gt.valid[uid], best_of))
        per_sample.append({"sample_uid": uid, "num_trajectories": int(trajs.shape[0]),
                           **{k: round(v, 4) for k, v in m.items()}})
    return {"rows": rows, "agg": aggregate(rows), "cv_agg": aggregate(cv_rows),
            "per_sample": per_sample, "k_values": sorted(k_values), "missing_gt": missing_gt}


def markdown_report(model: str, sections: Sequence[dict], best_of: int,
                    general: Sequence[str] = (), notes: Sequence[str] = ()) -> str:
    hdr = "| Model | Selection | " + " | ".join(f"{h}s" for h in HORIZONS) + " |"
    sep = "|---|---|" + "---|" * len(HORIZONS)
    L = [f"# {model} on KITScenes val (117 scenes, 11,035 samples)", "",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
         "Values are ADE / FDE in metres (10 Hz, ego frame at t0, positions only).", ""]
    for sec in sections:
        L += [f"## Ground truth: {sec['label']}", "", hdr, sep]
        L += [f"| {m} | {s} | " + " | ".join(c) + " |" for m, s, c in sec["table_rows"]]
        L += ["", "Checks:", ""] + [f"- {c}" for c in sec["checks"]] + [""]
    if general:
        L += ["## Ground-truth comparison", ""] + [f"- {g}" for g in general] + [""]
    L += ["## Definitions", "",
          "- top-1 (sample 0): the first predicted trajectory; the model outputs no confidence score."]
    if best_of > 1:
        L += [f"- best of {best_of}: minADE and minFDE over the first {best_of} trajectories, each "
              "minimised independently. Oracle (uses the ground truth)."]
    L += [
          "- poses.txt: future ego poses from KITScenes poses.txt in the t0 ego frame "
          "(prepare_kitscenes_e2e.py).",
          "- trajectory_xy_m: the future path stored in the AutoE2E v3.5 mirror shards "
          "(extract_mirror_gt.py), first 5 s.",
          "- Both models are scored by the same code against the same ground truths."]
    L += [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


def evaluate(model: str, predictions: Dict[str, np.ndarray], gts: Sequence[GroundTruth],
             index: List[str], out_dir: Optional[Path], best_of: int = DEFAULT_BEST_OF,
             extra_checks: Sequence[str] = (), notes: Sequence[str] = ()) -> int:
    """Score predictions (uid -> (K, T, 2)) against one or more ground truths.

    stdout: one results table per ground truth (a 'Ground truth: ...' line above each when
    there is more than one). stderr: one WARNING line per problem.
    out_dir: results.tsv / per_sample.csv (first ground truth), results_<key>.tsv /
    per_sample_<key>.csv (others), report.md and metrics.json (all).
    Returns 0, or 1 if no sample could be scored against the first ground truth."""
    if isinstance(gts, GroundTruth):
        gts = [gts]
    multi = len(gts) > 1
    sections = []
    for i, gt in enumerate(gts):
        res = _score_against(predictions, gt, index, best_of)
        if not res["rows"]:
            msg = (f"no sample could be scored against {gt.label} -- check that the predictions, "
                   "the ground-truth file and the index refer to the same samples")
            if i == 0:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 1
            print(f"WARNING: {msg}", file=sys.stderr)
            continue
        extra = list(extra_checks) if i == 0 else []
        if res["missing_gt"]:
            extra.append(f"{res['missing_gt']:,} predicted samples have no ground truth in this source "
                         "(not scored)")
        checks, problems = build_checks(res["agg"], res["cv_agg"], len(index), len(res["rows"]),
                                        res["k_values"], best_of, extra)
        table_rows = result_rows(model, res["agg"], best_of)
        sections.append({**res, "key": gt.key, "label": gt.label, "checks": checks,
                         "problems": problems, "table_rows": table_rows,
                         "table": format_table(table_rows)})

    general = [compare_ground_truths(gts[0], g) for g in gts[1:]]

    for j, sec in enumerate(sections):
        if multi:
            print(("" if j == 0 else "\n") + f"Ground truth: {sec['label']}")
        print(sec["table"])
    for sec in sections:
        for p in sec["problems"]:
            print(f"WARNING: {'[' + sec['key'] + '] ' if multi else ''}{p}", file=sys.stderr)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, sec in enumerate(sections):
            suffix = "" if j == 0 else f"_{sec['key']}"
            (out_dir / f"results{suffix}.tsv").write_text(sec["table"] + "\n")
            fields = list(dict.fromkeys(k for r in sec["per_sample"] for k in r))
            with open(out_dir / f"per_sample{suffix}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(sec["per_sample"])
        (out_dir / "report.md").write_text(markdown_report(model, sections, best_of, general, notes))
        (out_dir / "metrics.json").write_text(json.dumps(
            {"model": model, "horizons_s": list(HORIZONS), "best_of": best_of,
             "ground_truth_comparison": general,
             "ground_truths": {sec["key"]: {
                 "label": sec["label"],
                 "population": {"requested": len(index), "scored": len(sec["rows"])},
                 "checks": sec["checks"], "metrics": sec["agg"], "constant_velocity": sec["cv_agg"]}
                 for sec in sections}}, indent=2))
    return 0
