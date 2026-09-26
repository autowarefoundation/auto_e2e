#!/usr/bin/env python3
"""Pack the KITScenes train corpus one scene at a time, without staging it all.

Motivation. `data_processing` needs its `raw_data` to hold extracted scenes, and
the KITScenes archives are ~3 GB each: extracting the 533-scene train split at
once needs well over 2 TB, which does not fit on a typical workstation. Streaming
the *download* does not help, because each archive still lands on disk before it
can be packed.

The packed output, on the other hand, is small. This script therefore streams the
*packing*: extract one archive, pack it, delete the extracted copy, move to the
next. Peak disk stays at roughly one scene instead of the whole corpus, so the
frozen split becomes reachable on a machine that cannot hold the raw data.

One scene per partition is the intended shape, not a workaround: calibration and
map state are scene-scoped, and `data_processing` raises "KITScenes partition
contains scenes with different calibration; pack one scene per partition" when a
partition mixes them.

With `--fetch` it also downloads each archive and deletes it once packed, so the
archives never all exist at once either. That matters: streaming the packing alone
still needs the ~1.6 TB of compressed archives on disk, which is more than a laptop
has. Fetching one at a time brings peak usage down to a single archive plus a single
extraction — roughly 8 GB — for the whole 533-scene corpus.

Resumable: a scene whose output already carries a `manifest.json` is skipped, so
the run can be interrupted and restarted.

Usage:
    export PYTHONPATH=<repo>/Model:<repo>
    python Platform/scripts/pack_kitscenes_corpus.py \\
        --tar-src /data/KITScenes-Multimodal/data/train \\
        --work-root /data/_staging \\
        --out-root /data/_shards

Validate the result before training with `verify_kitscenes_corpus.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Pinned by `data_processing`, which raises if the packed revision differs.
KITSCENES_SOURCE_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"

# `data_processing` defaults to DATASET_PACK_VERSION, but the KITScenes navigation
# path is versioned separately as KITSCENES_NAVIGATION_DATASET_VERSION, and it is
# that one the frozen validation manifest carries. Packing with the default and
# then training fails with "validation manifest dataset version does not match
# packed shards".
#
# Read from the manifest rather than pinned here: the contract is re-cut as the
# packed data changes, and a corpus packed against a stale version has to be
# repacked, not re-checked.
REPO_ROOT = Path(__file__).resolve().parents[2]


def frozen_dataset_version() -> str:
    """The packed dataset version the current frozen split expects."""
    from training.dataset_policy import KITSCENES_TRAINING_POLICY

    manifest = REPO_ROOT / "Model" / "training" / (
        KITSCENES_TRAINING_POLICY.validation_manifest
    )
    return str(json.loads(manifest.read_text())["dataset_version"])

# Abort after this many CONSECUTIVE failures. One bad archive should not end a
# multi-hour run, but a systematic breakage should stop it immediately.
MAX_CONSECUTIVE_FAILURES = 3


def free_gb(path: Path) -> float:
    """Free space, in GiB, on the filesystem holding ``path``."""
    return shutil.disk_usage(path).free / 1024**3


def extract_scene(tar_path: Path, dest_dir: Path) -> str:
    """Extract one scene archive under ``dest_dir`` and return its scene ID.

    ``-C`` is passed BEFORE the archive: in GNU tar it is positional and applies
    to the operands that follow it, so trailing it silently extracts into the
    current working directory while still returning 0.

    tar's stderr is propagated rather than swallowed: a failure that returns a
    non-zero code with no message is indistinguishable from a successful run that
    produced nothing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(dest_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tar exited {result.returncode}: {result.stderr.strip()[:300]}"
        )
    scene_id = tar_path.stem
    if not (dest_dir / scene_id).is_dir():
        produced = sorted(p.name for p in dest_dir.iterdir() if p.is_dir())
        raise RuntimeError(
            f"archive did not produce {scene_id}/, found {produced[:5]}"
        )
    return scene_id


def list_scene_archives(repo_id: str, revision: str, split: str = "train") -> list[str]:
    """Repository paths of every scene archive in ``split``, sorted."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=False)
    prefix = f"data/{split}/"
    return sorted(
        sibling.rfilename for sibling in info.siblings
        if sibling.rfilename.startswith(prefix)
        and sibling.rfilename.endswith(".tar")
    )


def fetch_archive(repo_id: str, revision: str, filename: str, dest_dir: Path) -> Path:
    """Download one archive into ``dest_dir`` and return its path.

    ``local_dir`` keeps the file out of the shared HF cache, so deleting it after
    packing actually reclaims the space instead of leaving a cached copy behind.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest_dir),
    )
    return Path(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar-src",
                    help="Directory holding already-downloaded archives. Omit with "
                         "--fetch to download them one at a time instead")
    ap.add_argument("--fetch", action="store_true",
                    help="Download each archive, pack it, then delete it. Keeps peak "
                         "disk at one archive plus one extraction")
    ap.add_argument("--repo-id", default="KIT-MRT/KITScenes-Multimodal")
    ap.add_argument("--source-revision", default=KITSCENES_SOURCE_REVISION,
                    help="Dataset revision to fetch; must match what packing pins")
    ap.add_argument("--work-root", required=True,
                    help="Staging root; emptied after each scene")
    ap.add_argument("--out-root", required=True,
                    help="Destination for the packed partitions")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--dataset-version", default=None,
                    help="Must match the frozen validation manifest "
                         "(default: read from it)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Pack only the first N archives (0 = all)")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="Stop when free space falls below this")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="Do not delete extracted scenes (disables streaming)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate the arguments BEFORE importing the pipeline: those imports are heavy
    # and need PYTHONPATH set, so doing them first turns a plain "you forgot an
    # argument" into an ImportError that points somewhere else entirely.
    if not args.fetch and not args.tar_src:
        print("error: pass --tar-src, or --fetch to download the archives",
              file=sys.stderr)
        return 1

    from flytekit.types.directory import FlyteDirectory

    from Platform.pipelines.workflows import Dataset, data_processing

    if args.dataset_version is None:
        args.dataset_version = frozen_dataset_version()

    if args.fetch:
        scene_sources = list_scene_archives(args.repo_id, args.source_revision)
        origin = f"{args.repo_id}@{args.source_revision[:7]}"
    else:
        tar_src = Path(args.tar_src)
        if not tar_src.is_dir():
            print(f"error: {tar_src} is not a directory", file=sys.stderr)
            return 1
        scene_sources = [str(path) for path in sorted(tar_src.glob("*.tar"))]
        origin = str(tar_src)
    if args.limit:
        scene_sources = scene_sources[: args.limit]
    if not scene_sources:
        print(f"error: no .tar archives found in {origin}", file=sys.stderr)
        return 1

    work_root = Path(args.work_root)
    out_root = Path(args.out_root)
    # `data_processing` builds KitScenesDataset(data_root=raw, split="train"),
    # so the staging tree must expose exactly that layout.
    stage_scenes = work_root / "data" / "train"
    stage_scenes.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"# {len(scene_sources)} archive(s) from {origin}"
          + ("  (fetching one at a time)" if args.fetch else ""))
    print(f"#   staging: {stage_scenes} (emptied after each scene)")
    print(f"#   output : {out_root}")
    print(f"#   version: {args.dataset_version}")
    print(f"#   free   : {free_gb(out_root):.0f} GiB\n")

    packed: list[str] = []
    failed: list[dict] = []
    consecutive = 0
    started = time.time()

    archive_dir = work_root / "archives"
    for index, source in enumerate(scene_sources, 1):
        scene_id = Path(source).stem
        dest = out_root / scene_id
        if (dest / "manifest.json").exists():
            packed.append(str(dest))
            print(f"  [{index}/{len(scene_sources)}] {scene_id[:8]} already packed, skipping")
            continue

        if free_gb(work_root) < args.min_free_gb:
            print(
                f"\naborting: {free_gb(out_root):.0f} GiB free, below the "
                f"{args.min_free_gb:.0f} GiB floor. Free space and rerun; the "
                "run resumes where it stopped.",
                file=sys.stderr,
            )
            break

        done = index - 1
        eta = ""
        if done:
            rate = (time.time() - started) / done
            eta = f"  ETA {(len(scene_sources) - done) * rate / 3600:.1f} h"
        print(f"  [{index}/{len(scene_sources)}] {scene_id[:8]} "
              + ("fetching ..." if args.fetch else "extracting ..."),
              end="", flush=True)

        staged = stage_scenes / scene_id
        archive: Path | None = None
        try:
            if args.fetch:
                archive = fetch_archive(
                    args.repo_id, args.source_revision, source, archive_dir
                )
                print(" extracting ...", end="", flush=True)
            else:
                archive = Path(source)
            extract_scene(archive, stage_scenes)
            if args.fetch:
                # Delete the archive as soon as it is unpacked: holding it until the
                # end of the scene would double peak usage for no reason.
                archive.unlink(missing_ok=True)
                archive = None
            print(" packing ...", end="", flush=True)
            out = data_processing.task_function(
                raw_data=FlyteDirectory(str(work_root)),
                dataset=Dataset.KITSCENES,
                source_revision=KITSCENES_SOURCE_REVISION,
                dataset_version=args.dataset_version,
                hz=10,
                image_size=args.image_size,
                episodes=1,
                world_model=False,
                group_ids=[scene_id],
            )
            src = Path(str(getattr(out, "path", out)))
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            samples = json.loads(
                (dest / "manifest.json").read_text()
            ).get("total_samples", "?")
            packed.append(str(dest))
            consecutive = 0
            print(f" ok ({samples} samples){eta}")
        except Exception as error:  # noqa: BLE001 - reported, then counted
            consecutive += 1
            failed.append(
                {"scene": scene_id, "error": f"{type(error).__name__}: {error}"[:400]}
            )
            print(f" FAILED {type(error).__name__}: {str(error)[:120]}")
            traceback.print_exc(limit=3)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"\naborting: {consecutive} consecutive failures, which "
                    "suggests a systematic problem rather than a bad archive.",
                    file=sys.stderr,
                )
                break
        finally:
            # These deletes are what keep peak disk at one scene.
            if not args.keep_extracted and staged.exists():
                shutil.rmtree(staged, ignore_errors=True)
            if args.fetch and archive is not None:
                archive.unlink(missing_ok=True)

    index_path = out_root / "shards_index.json"
    index_path.write_text(json.dumps({"packed": packed, "failed": failed}, indent=2))
    print(f"\n# packed {len(packed)} · failed {len(failed)}")
    print(f"# index -> {index_path}")
    for item in failed[:5]:
        print(f"#   {item['scene'][:8]}  {item['error'][:110]}")
    if args.fetch:
        shutil.rmtree(archive_dir, ignore_errors=True)
    print("\n# Validate before training:")
    print(f"#   python Platform/scripts/verify_kitscenes_corpus.py "
          f"--shards-root {out_root}")
    return 0 if packed else 1


if __name__ == "__main__":
    raise SystemExit(main())
