#!/usr/bin/env python3
"""Extract the AutoE2E v3.5 mirror ground truth (trajectory_xy_m) for KITScenes val.

The mirror (huggingface.co/AutowareFoundation/auto_e2e, datasets/kitscenes-val/v3.5) stores
each sample's future path in its shards as <sample_uid>.trajectory_xy.npz:
    trajectory_xy_m   (64, 2) float32   x, y in metres, ego frame at t0, t0+0.1 s .. t0+6.4 s
    trajectory_valid  (64,)   uint8
This script reads only that file from the 117 val shards (build job kitscenes-v35-val-e0d55e79)
and writes one line per sample:

    {"sample_uid": "kitscenes-v1-<scene>-f000040", "trajectory_xy_m": [[x, y], ...], "valid": [1, ...]}

The scorers use it with --mirror-gt to report statistics against both ground truths:
poses.txt (from prepare_kitscenes_e2e.py) and trajectory_xy_m (this file).

    python extract_mirror_gt.py --data-dir kitscenes_val --delete-shards

Shards are downloaded one at a time (about 8 GB in total; images are in the same files).
With --delete-shards each shard is removed right after reading, so only one is on disk at a
time. --shards-dir reads shards you already have locally instead of downloading.
Re-running continues: shards already read are listed in <out>.progress.json.
Needs numpy and huggingface_hub (not needed with --shards-dir and --manifest).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXPECTED_SAMPLES, load_index, resolve_data  # noqa: E402

REPO, PATH = "AutowareFoundation/auto_e2e", "datasets/kitscenes-val/v3.5"
VAL_JOB = "kitscenes-v35-val-e0d55e79"
JOB_RE = re.compile(r"/(kitscenes-v35-[a-z]+-[0-9a-f]{8})-")
SUFFIX = ".trajectory_xy.npz"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def val_shards(manifest: dict) -> list:
    """Shard file names of the val partitions, smallest first."""
    val = {p["partition_id"] for p in manifest["partitions"]
           if (m := JOB_RE.search(p.get("source_uri", ""))) and m.group(1) == VAL_JOB}
    entries = [e for e in manifest["shard_entries"] if e["name"].split("-train-")[0] in val]
    return [e["name"] for e in sorted(entries, key=lambda e: e.get("byte_size", 0))]


def read_shard(path: Path) -> dict:
    """{sample_uid: (xy (64, 2), valid (64,))}, streaming; image members are skipped unread."""
    out = {}
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(SUFFIX)):
                continue
            fh = tf.extractfile(m)
            if fh is None:          # only for non-regular members; isfile() already excludes them
                continue
            uid = Path(m.name).name[: -len(SUFFIX)]
            with np.load(io.BytesIO(fh.read()), allow_pickle=False) as z:
                xy = np.asarray(z["trajectory_xy_m"], dtype=np.float64).reshape(-1, 2)
                valid = np.asarray(z["trajectory_valid"], dtype=np.uint8).reshape(-1)
            out[uid] = (xy, valid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="prepare_kitscenes_e2e.py output (for the index)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: <data-dir>/kitscenes_val_gt_mirror.jsonl)")
    ap.add_argument("--cache", type=Path, default=Path("auto_e2e_meta"), help="download directory")
    ap.add_argument("--revision", default=None, help="pin the auto_e2e commit")
    ap.add_argument("--shards-dir", type=Path, default=None, help="read local shards instead of downloading")
    ap.add_argument("--manifest", type=Path, default=None, help="local shards/manifest.json")
    ap.add_argument("--delete-shards", action="store_true", help="delete each downloaded shard after reading")
    args = ap.parse_args()

    _, index_path, _ = resolve_data(args.data_dir, None, None)
    wanted = set(load_index(index_path))
    out_path = args.out or args.data_dir / "kitscenes_val_gt_mirror.jsonl"
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")

    def fetch(rel: str) -> Path:
        if args.shards_dir is not None and rel.startswith("shards/") and rel != "shards/manifest.json":
            return args.shards_dir / Path(rel).name
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(REPO, f"{PATH}/{rel}", repo_type="model",
                                    revision=args.revision, local_dir=args.cache))

    manifest_path = args.manifest or fetch("shards/manifest.json")
    shards = val_shards(json.loads(Path(manifest_path).read_text()))
    log(f"{len(shards)} val shards in the mirror manifest; index has {len(wanted):,} samples")

    done = set(json.loads(progress_path.read_text())) if progress_path.exists() and part_path.exists() else set()
    if not done and part_path.exists():
        part_path.unlink()
    found = set()
    if part_path.exists():
        keep = 0                                    # drop a half-written last line of a killed run
        with open(part_path, "rb") as fh:
            for line in fh:
                if not line.endswith(b"\n"):
                    break
                found.add(json.loads(line)["sample_uid"])
                keep += len(line)
        os.truncate(part_path, keep)
    if done:
        log(f"continuing: {len(done)} shards already read, {len(found):,} samples")

    with open(part_path, "a") as out:
        for i, name in enumerate(shards, 1):
            if name in done:
                continue
            path = fetch(f"shards/{name}")
            samples = read_shard(path)
            for uid, (xy, valid) in samples.items():
                if uid in wanted and uid not in found:
                    out.write(json.dumps({"sample_uid": uid,
                                          "trajectory_xy_m": np.round(xy, 5).tolist(),
                                          "valid": valid.tolist()}) + "\n")
                    found.add(uid)
            out.flush()
            os.fsync(out.fileno())
            done.add(name)
            progress_path.write_text(json.dumps(sorted(done)))
            if args.delete_shards and args.shards_dir is None:
                path.unlink(missing_ok=True)
            log(f"  [{i}/{len(shards)}] {name}: {len(samples)} samples, {len(found):,} total")

    missing = wanted - found
    os.replace(part_path, out_path)
    progress_path.unlink(missing_ok=True)
    print(f"wrote {out_path}: {len(found):,} samples"
          + ("" if not missing else f"; MISSING {len(missing):,} index samples, e.g. {sorted(missing)[:3]}"))
    return 0 if len(found) == len(wanted) == EXPECTED_SAMPLES else 2


if __name__ == "__main__":
    sys.exit(main())
