#!/usr/bin/env python3
"""Statistics for Qwen-Drive-1.0 on KITScenes val (117 scenes, 11,035 samples).

1. Prepare the data once (prepare_kitscenes_e2e.py), then run Qwen-Drive on the scene file
   with its own runner, in the Qwen-Drive repo / venv:

       python scripts/run_planning.py \
           --model <Qwen-Drive-1.0-4B dir> \
           --scenes data_e2e/kitscenes_val/kitscenes_val_e2e.jsonl \
           --image-root <image_root from prepare_manifest.json> \
           --output outputs/qwendrive/predictions.jsonl \
           --num-samples 6 --seed 0

   (resumes by default; --num-shards / --shard-index split the work over GPUs, then
   concatenate the shard outputs.)

2. Score:

       python e2e_eval/eval_qwendrive_kitscenes.py \
           --predictions outputs/qwendrive/predictions.jsonl \
           --data-dir data_e2e/kitscenes_val \
           --out outputs/qwendrive/metrics

stdout is only the table:

    Model           Selection         1s             2s             3s             5s
    Qwen-Drive-1.0  top-1 (sample 0)  ADE / FDE      ...
    Qwen-Drive-1.0  best of 3         ADE / FDE      ...

Problems (incomplete population, ground-truth mismatch, too few samples for best of 3) are
printed as WARNING lines on stderr; the full checks are in <out>/report.md. Metric and row
definitions: kitscenes_e2e_common.py. Only numpy is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from accuracy_benchmark.kitscenes.common import (  # noqa: E402
    DEFAULT_BEST_OF, GroundTruth, add_mirror_arg, evaluate, load_ground_truths, load_index, normalize_uid,
    resolve_data,
)

MODEL = "Qwen-Drive-1.0"


def load_predictions(path: Path, wanted: set, gt: GroundTruth):
    """uid -> (K, 50, 2) from run_planning.py's predictions.jsonl; plus check lines."""
    preds, dup, outside, gt_diff, checked, truncated = {}, 0, 0, 0, 0, False
    with open(path) as fh:
        for line in fh:
            if not line.endswith("\n"):
                truncated = True          # last line of an interrupted run
                break
            rec = json.loads(line)
            uid = normalize_uid(rec["token"])
            if uid is None or uid not in wanted:
                outside += 1
                continue
            if uid in preds:
                dup += 1
                continue
            preds[uid] = np.asarray(rec["trajectories"], dtype=np.float64)[..., :2]
            # The runner copies the scene file's ground truth into each prediction; if it
            # differs, the predictions were made from a different scene file.
            fut = rec.get("future_trajectory")
            if fut is not None and uid in gt.future:
                checked += 1
                if not np.allclose(np.asarray(fut, dtype=np.float64)[:, :2], gt.future[uid], atol=1e-3):
                    gt_diff += 1
    checks = []
    if outside:
        checks.append(f"{outside:,} predictions are not in the index (ignored)")
    if dup:
        checks.append(f"{dup:,} duplicate predictions (first one kept)")
    if truncated:
        checks.append("last line of the predictions file is incomplete (ignored)")
    if gt_diff:
        checks.append(f"WARNING: {gt_diff:,} of {checked:,} predictions carry a different ground truth "
                      "than the scene file -- were they made from another scene file?")
    return preds, checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True, help="run_planning.py output (jsonl)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="prepare_kitscenes_e2e.py output dir (scene file + index)")
    ap.add_argument("--scenes", type=Path, default=None, help="scene file (default: from --data-dir)")
    ap.add_argument("--index", type=Path, default=None, help="index csv (default: from --data-dir)")
    ap.add_argument("--out", type=Path, default=None, help="directory for results.tsv, report.md, ...")
    ap.add_argument("--best-of", type=int, default=DEFAULT_BEST_OF)
    add_mirror_arg(ap)
    args = ap.parse_args()

    scenes, index_path, _ = resolve_data(args.data_dir, args.scenes, args.index)
    index = load_index(index_path)
    gts = load_ground_truths(scenes, index, args.mirror_gt, args.data_dir)
    gt = gts[0]
    preds, checks = load_predictions(args.predictions, set(index), gt)
    for c in checks:
        if c.startswith("WARNING"):
            print(c, file=sys.stderr)
    return evaluate(MODEL, preds, gts, index, args.out, args.best_of, extra_checks=checks,
                    notes=["Inputs: 3 cameras (front, front-left, front-right) x 4 frames at 2 Hz, "
                           "1.5 s ego history, navigation command derived from the ground-truth future."])


if __name__ == "__main__":
    sys.exit(main())
