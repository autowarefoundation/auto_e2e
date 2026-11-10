#!/usr/bin/env python3
"""Score Qwen-Drive predictions on KITScenes with ADE/FDE at 1, 2, 3, 4 and 5 s.

Input is the ``predictions.jsonl`` written by Qwen-Drive's ``scripts/run_planning.py``
for a scene file built by ``kitscenes_to_qwendrive.py``. Only numpy is needed.

Metric definition (same as Autoware auto_e2e's KITScenes evaluator,
``Model/evaluation/kitscenes_benchmark.py::compute_displacement_metrics``):

* 10 Hz, ego frame of the keyframe (x forward, y left), positions only
* ADE_T = mean L2 error over future steps 1..T*10, FDE_T = L2 error at step T*10
* averaged over samples (keyframes)

Differences you can control:

* ``--horizons`` picks the horizons in seconds (default 1 2 3 4 5; auto_e2e reports 3 5).
* A keyframe is scored at horizon T only when the ground truth is valid for all of the
  first T*10 steps (auto_e2e requires a full target too). Records that only reach 3 s
  count for 1-3 s but not 4-5 s; the report says how many were used per horizon.
  ``--common-set`` instead scores every horizon on the same keyframes (those valid up to
  the longest horizon), so the numbers across horizons come from one sample set.
* Qwen-Drive may draw several trajectories per keyframe (``--num-samples``). Reported:
  ``first`` = sample 0 (the single-trajectory number, comparable to a model that outputs
  one trajectory), ``mean`` = averaged over samples, ``min`` = best sample (oracle,
  needs the ground truth, not comparable to single-output models).
* ``--sample-list`` restricts scoring to a fixed keyframe list so two models are scored on
  exactly the same samples. It accepts an auto_e2e benchmark manifest (JSON with
  ``sample_uids`` like ``kitscenes-v1-<scene>-f000040``), a JSON list, or a text/CSV file
  with tokens ``<scene>-<frame>`` or ``<scene>,<frame>`` per line.

Usage:
    python eval_kitscenes.py --predictions outputs/kitscenes_test/predictions.jsonl \
        --scenes scenes/kitscenes_test.jsonl --output outputs/kitscenes_test/metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HZ = 10
HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)   # seconds; overridden by --horizons
_UID = re.compile(r"^kitscenes-v1-(?P<scene>[0-9a-f-]{36})-f(?P<frame>\d+)$")
_TOKEN = re.compile(r"^(?P<scene>[0-9a-f-]{36})[-,](?P<frame>\d+)$")


def normalize_token(text: str) -> str | None:
    text = text.strip().strip('"')
    for pattern in (_UID, _TOKEN):
        m = pattern.match(text)
        if m:
            return f"{m['scene']}-{int(m['frame'])}"
    return None


def load_sample_list(path: Path) -> set[str]:
    raw = path.read_text()
    items: list[str]
    try:
        payload = json.loads(raw)
        items = payload["sample_uids"] if isinstance(payload, dict) else list(payload)
    except (json.JSONDecodeError, KeyError, TypeError):
        items = [line for line in raw.splitlines() if line.strip() and not line.startswith("#")]
    tokens, bad = set(), []
    for item in items:
        tok = normalize_token(str(item))
        (tokens.add(tok) if tok else bad.append(item))
    if bad:
        sys.exit(f"could not parse {len(bad)} entries of {path}, e.g. {bad[:3]}")
    return tokens


def load_meta(path: Path | None) -> dict[str, dict]:
    """token -> intent / speed from the scene file (optional, for breakdowns)."""
    if path is None:
        return {}
    meta = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            m = rec["meta_info"]
            meta[m["token"]] = {
                "intent": m.get("intent", "UNKNOWN"),
                "speed": rec["trajectory"].get("current_state", {}).get("speed"),
            }
    return meta


def score_record(trajs: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """trajs [S, >=50, >=2], gt [50, >=2], valid [50] -> metrics for each full horizon."""
    err = np.linalg.norm(trajs[:, : gt.shape[0], :2] - gt[None, :, :2], axis=-1)  # [S, 50]
    row: dict[str, float] = {}
    for sec in HORIZONS:
        steps = sec * HZ
        if steps > gt.shape[0] or not valid[:steps].all():
            continue
        ade = err[:, :steps].mean(axis=1)
        fde = err[:, steps - 1]
        for name, values in (("ADE", ade), ("FDE", fde)):
            row[f"{name}_{sec}s_first"] = float(values[0])
            row[f"{name}_{sec}s_mean"] = float(values.mean())
            row[f"{name}_{sec}s_min"] = float(values.min())
    return row


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for sec in HORIZONS:
        scored = [r for r in rows if f"ADE_{sec}s_first" in r]
        out[f"num_scored_{sec}s"] = len(scored)
        for key in (f"{m}_{sec}s_{v}" for m in ("ADE", "FDE") for v in ("first", "mean", "min")):
            if scored:
                out[key] = round(float(np.mean([r[key] for r in scored])), 4)
    return out


def format_table(overall: dict) -> str:
    """Compact overview: one row per horizon, metres."""
    lines = ["", f"{'horizon':>7} {'n':>5} | {'ADE first':>9} {'FDE first':>9} | "
                 f"{'ADE mean':>8} {'FDE mean':>8} | {'minADE':>7} {'minFDE':>7}"]
    for sec in HORIZONS:
        if f"ADE_{sec}s_first" not in overall:
            continue
        lines.append(
            f"{sec:>6}s {overall[f'num_scored_{sec}s']:>5} | "
            f"{overall[f'ADE_{sec}s_first']:>9.3f} {overall[f'FDE_{sec}s_first']:>9.3f} | "
            f"{overall[f'ADE_{sec}s_mean']:>8.3f} {overall[f'FDE_{sec}s_mean']:>8.3f} | "
            f"{overall[f'ADE_{sec}s_min']:>7.3f} {overall[f'FDE_{sec}s_min']:>7.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    global HORIZONS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--scenes", type=Path, default=None,
                   help="the scene JSONL, for per-command and per-speed breakdowns")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="score only these keyframes (auto_e2e manifest, JSON list or text)")
    p.add_argument("--output", type=Path, default=None, help="directory for summary + per-sample CSV")
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS),
                   help="horizons in whole seconds (max 5 = the 50 predicted poses)")
    p.add_argument("--common-set", action="store_true",
                   help="score all horizons on the keyframes valid up to the longest horizon")
    args = p.parse_args(argv)

    if any(h < 1 or h * HZ > 50 for h in args.horizons):
        sys.exit("--horizons must be between 1 and 5 seconds")
    HORIZONS = tuple(sorted(set(args.horizons)))

    wanted = load_sample_list(args.sample_list) if args.sample_list else None
    meta = load_meta(args.scenes)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped: defaultdict[str, int] = defaultdict(int)
    with open(args.predictions) as fh:
        for line in fh:
            if not line.endswith("\n"):
                skipped["truncated_line"] += 1
                break
            rec = json.loads(line)
            token = rec["token"]
#            if wanted is not None and token not in wanted:
            if wanted is not None and (normalize_token(token) or token) not in wanted:
                continue
            if token in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(token)
            if rec.get("future_trajectory") is None:
                skipped["no_ground_truth"] += 1
                continue
            trajs = np.asarray(rec["trajectories"], dtype=np.float64)
            if trajs.ndim == 2:
                trajs = trajs[None]
            gt = np.asarray(rec["future_trajectory"], dtype=np.float64)
            valid = np.asarray(rec.get("future_valid") or np.ones(len(gt)), dtype=bool)
            if not (np.isfinite(trajs).all() and np.isfinite(gt).all()):
                skipped["non_finite"] += 1
                continue
            row = {"token": token, "num_samples": trajs.shape[0],
                   "num_valid_future": int(valid.sum())}
            row.update(meta.get(token, {}))
            if args.common_set and not valid[: max(HORIZONS) * HZ].all():
                skipped[f"future_shorter_than_{max(HORIZONS)}s"] += 1
                continue
            metrics = score_record(trajs, gt, valid)
            if not metrics:
                skipped[f"future_shorter_than_{min(HORIZONS)}s"] += 1
                continue
            row.update(metrics)
            rows.append(row)

    if not rows:
        sys.exit("no scorable predictions")

    summary: dict[str, Any] = {"overall": summarize(rows)}
    summary["overall"]["num_samples_per_keyframe"] = sorted({r["num_samples"] for r in rows})
    if wanted is not None:
        missing = sorted(wanted - seen)
        summary["sample_list"] = {"requested": len(wanted), "found": len(wanted) - len(missing),
                                  "missing_examples": missing[:5]}
    if meta:
        by_cmd = defaultdict(list)
        for r in rows:
            by_cmd[r.get("intent", "UNKNOWN")].append(r)
        summary["by_command"] = {k: summarize(v) for k, v in sorted(by_cmd.items())}
        by_speed = defaultdict(list)
        for r in rows:
            s = r.get("speed")
            if s is None:
                continue
            bucket = "0-2 m/s" if s < 2 else "2-8 m/s" if s < 8 else "8-14 m/s" if s < 14 else ">=14 m/s"
            by_speed[bucket].append(r)
        summary["by_speed"] = {k: summarize(v) for k, v in by_speed.items()}
    summary["skipped"] = dict(skipped)

    summary["horizons_s"] = list(HORIZONS)
    summary["common_set"] = bool(args.common_set)
    text = json.dumps(summary, indent=2)
    print(text)
    print(format_table(summary["overall"]), file=sys.stderr)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(text)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "token", k))
        with open(args.output / "per_sample.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
