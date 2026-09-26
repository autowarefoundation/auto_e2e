#!/usr/bin/env python3
"""Check a packed KITScenes corpus against the frozen split, before training.

`train_il` already validates the packed corpus against the frozen split named by
`KITSCENES_TRAINING_POLICY.validation_manifest` and aborts on any mismatch, but it
does so after start-up, once the shards have been scanned. On a corpus that took
hours to pack that is a late and expensive place to find out.

The manifest is read from the policy rather than named here, because the frozen
contract is re-cut as the packed data changes: a checker pinned to one snapshot
would keep passing a corpus that training has already started rejecting.

This runs the same comparison beforehand and reports every check, so a corpus
can be fixed before a run rather than during one. It reads tar headers and
`manifest.json` members only; no camera payload is decoded.

Checks performed, all against the frozen manifest:

    packed partitions        == available_scene_count
    empty partitions         == excluded_empty_scene_count
    eligible groups          == eligible_group_count
    group UID digest         == eligible_group_uid_digest
    samples                  == eligible_sample_count
    sample UID digest        == eligible_sample_uid_digest
    packed dataset version   == dataset_version
    packed contract digest   == packed_contract_digest

When they all pass, `--validation_scope full` will select the frozen holdout, so
the resulting ADE/FDE can be compared with anyone else's run over the same split.

Usage:
    export PYTHONPATH=<repo>/Model:<repo>
    python Platform/scripts/verify_kitscenes_corpus.py --shards-root /data/_shards

Exit codes: 0 all checks pass · 1 corpus does not match · 2 nothing to check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def frozen_manifest_path() -> Path:
    """Locate the split manifest ``train_il`` will validate this corpus against.

    Resolved through the training policy, so the checker and the trainer cannot
    disagree about which snapshot is current.
    """
    from training.dataset_policy import KITSCENES_TRAINING_POLICY

    return REPO_ROOT / "Model" / "training" / (
        KITSCENES_TRAINING_POLICY.validation_manifest
    )


class Check(NamedTuple):
    """One comparison between the packed corpus and the frozen manifest."""

    name: str
    actual: Any
    expected: Any

    @property
    def ok(self) -> bool:
        return self.actual == self.expected


def read_partitions(root: Path) -> tuple[dict[str, dict], list[str], int, set[str]]:
    """Scan partition directories exactly as ``train_il`` does at start-up.

    Returns the manifests keyed by directory, the non-empty directories, the
    number of empty ones, and the set of packed dataset versions.
    """
    manifests: dict[str, dict] = {}
    non_empty: list[str] = []
    empty = 0
    versions: set[str] = set()
    for path in sorted(root.iterdir()):
        manifest_path = path / "manifest.json"
        if not (path.is_dir() and manifest_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text())
        manifests[str(path)] = manifest
        version = manifest.get("dataset_version")
        if version:
            versions.add(str(version))
        if int(manifest.get("total_samples", 0)) <= 0:
            empty += 1
        else:
            non_empty.append(str(path))
    return manifests, non_empty, empty, versions


def build_checks(
    frozen: dict,
    *,
    partition_count: int,
    empty_count: int,
    group_uids: tuple[str, ...],
    group_digest: str,
    sample_count: int,
    sample_digest: str,
    dataset_version: str,
    contract_digest: str,
) -> list[Check]:
    """Pair every packed quantity with what the frozen manifest requires."""
    return [
        Check("packed partitions", partition_count,
              frozen["available_scene_count"]),
        Check("empty partitions", empty_count,
              frozen["excluded_empty_scene_count"]),
        Check("eligible groups", len(group_uids),
              frozen["eligible_group_count"]),
        Check("group UID digest", group_digest,
              frozen["eligible_group_uid_digest"]),
        Check("samples", sample_count, frozen["eligible_sample_count"]),
        Check("sample UID digest", sample_digest,
              frozen["eligible_sample_uid_digest"]),
        Check("dataset version", dataset_version, frozen["dataset_version"]),
        Check("contract digest", contract_digest,
              frozen["packed_contract_digest"]),
    ]


def format_check(check: Check) -> str:
    actual, expected = str(check.actual), str(check.expected)
    if len(actual) > 20:
        actual, expected = actual[:16] + "...", expected[:16] + "..."
    mark = "PASS" if check.ok else "FAIL"
    return f"  [{mark}] {check.name:<24} {actual:>20}  expected {expected}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards-root", required=True,
                    help="Directory holding the packed partitions")
    ap.add_argument("--manifest", default=None,
                    help="Frozen split manifest to compare against "
                         "(default: the one the training policy names)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from data_parsing.pre_extracted import discover_split_inventory
    from training.dataset_policy import group_uid_digest

    from Platform.pipelines.training_checkpoint import stable_digest

    root = Path(args.shards_root)
    if not root.is_dir():
        print(f"error: {root} is not a directory")
        return 2
    manifest_path = (
        Path(args.manifest) if args.manifest else frozen_manifest_path()
    )
    frozen = json.loads(manifest_path.read_text())

    manifests, non_empty, empty, versions = read_partitions(root)
    if not manifests:
        print(f"error: no partitions with a manifest.json under {root}")
        return 2

    print(f"corpus : {root}")
    print(f"  partitions with a manifest : {len(manifests)}")
    print(f"  empty                      : {empty}")
    print(f"  non-empty                  : {len(non_empty)}")
    print(f"  dataset version            : {sorted(versions) or ['(none)']}")
    if not non_empty:
        print("error: every partition is empty")
        return 2

    print("\nscanning shard headers (meta.json members only) ...", flush=True)
    try:
        inventory = discover_split_inventory(non_empty)
    except ValueError as error:
        # Expected while a corpus is still being packed, e.g. "requires metadata
        # for at least two split groups". Not a failure of the corpus itself.
        missing = frozen["available_scene_count"] - len(manifests)
        print(f"\n  not analysable yet: {error}")
        print(f"  {missing} of {frozen['available_scene_count']} partitions still "
              "missing; keep packing and run this again.")
        return 1

    contract_digests = {
        stable_digest(manifest.get("contracts")) for manifest in manifests.values()
    }
    checks = build_checks(
        frozen,
        partition_count=len(manifests),
        empty_count=empty,
        group_uids=inventory.group_uids,
        group_digest=group_uid_digest(inventory.group_uids),
        sample_count=inventory.sample_count,
        sample_digest=inventory.sample_uid_digest,
        dataset_version=next(iter(versions), ""),
        contract_digest=(
            next(iter(contract_digests)) if len(contract_digests) == 1
            else f"AMBIGUOUS({len(contract_digests)})"
        ),
    )

    print(f"\n== against {manifest_path.name} ==")
    for check in checks:
        print(format_check(check))

    failures = [check for check in checks if not check.ok]
    if failures:
        print(f"\n== not comparable yet: {len(failures)} check(s) failed.")
        missing = frozen["available_scene_count"] - len(manifests)
        if missing > 0:
            print(f"   {missing} partition(s) still to pack.")
        print("   Until they all pass, --validation_scope full will abort and the "
              "resulting metrics are not comparable with the frozen split.")
        return 1

    print("\n== corpus matches the frozen split.")
    print(f"   Holdout: {frozen['validation_group_count']} scenes / "
          f"{frozen['validation_sample_count']} samples, group digest "
          f"{frozen['validation_group_uid_digest'][:16]}...")
    index_path = root / "shards_index_verified.json"
    index_path.write_text(json.dumps(
        {"packed": sorted(manifests)}, indent=2))
    print(f"   Shard list -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
