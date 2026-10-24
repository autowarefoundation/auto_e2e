"""Extract Drive-JEPA commands from the published AutoE2E route inputs.

This is a maneuver-label adaptation (100 m lookahead), not NAVSIM's native
20 m centerline algorithm. It does not access poses or target trajectories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from accuracy_benchmark.drive_jepa.external_kitscenes import (
    EXPECTED_VAL_SAMPLE_UID_SHA256,
    file_sha256,
    published_digest_ids,
    write_json,
)

MIRROR_REVISION = "7262c151599c1b5e4a99f0da9c3196ff9ff615fe"
MIRROR_REPO = "AutowareFoundation/auto_e2e"
PREFIX = "datasets/kitscenes-val/v3.5/shards/"
COMMAND_ORDER = ("left", "straight", "right", "unknown")
KNOWN_MANEUVERS = {
    "left",
    "straight",
    "right",
    "unknown",
    "u_turn",
    "merge",
    "exit",
    "destination",
}


def route_command(metadata):
    """Map published route intent to one-hot NAVSIM slots; no zero fallback."""
    for field in ("route_valid", "map_valid"):
        if not isinstance(metadata.get(field), bool):
            raise ValueError(f"Navigation metadata requires boolean {field}")
    maneuver = metadata.get("route_maneuver")
    if maneuver not in KNOWN_MANEUVERS:
        raise ValueError(f"Unsupported published maneuver: {maneuver!r}")
    category = (
        maneuver if metadata["route_valid"] and metadata["map_valid"] else "unknown"
    )
    if category not in COMMAND_ORDER:
        category = (
            "unknown"  # Merge/exit/U-turn labels do not specify a NAVSIM direction.
        )
    command = [0.0] * 4
    command[COMMAND_ORDER.index(category)] = 1.0
    return command, category


def extract_shard(path):
    records = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            suffix = ".navigation_meta.json"
            if not member.name.endswith(suffix):
                continue
            if not member.isfile() or member.size > 128 * 1024:
                raise ValueError("Invalid navigation metadata member")
            uid = member.name[: -len(suffix)]
            if "/" in uid or uid in records:
                raise ValueError(f"Invalid or duplicated sample UID {uid}")
            payload = archive.extractfile(member).read()
            metadata = json.loads(payload)
            if metadata.get("schema_version") != "sample_navigation_v2":
                raise ValueError("Unexpected navigation metadata schema")
            if metadata.get("maneuver_lookahead_m") != 100.0:
                raise ValueError(
                    "Published maneuver lookahead differs from pinned policy"
                )
            command, category = route_command(metadata)
            records[uid] = {
                "command": command,
                "category": category,
                "route_maneuver": metadata["route_maneuver"],
                "route_valid": metadata["route_valid"],
                "map_valid": metadata["map_valid"],
                "sample_timestamp_ns": metadata["sample_timestamp_ns"],
                "route_id": metadata["route_id"],
                "scene_navigation_sha256": metadata["scene_navigation_sha256"],
                "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            }
    if not records:
        raise ValueError(f"No route metadata in {path}")
    return records


def prepare(args):
    from huggingface_hub import hf_hub_download

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(
        hf_hub_download(
            MIRROR_REPO,
            PREFIX + "manifest.json",
            revision=MIRROR_REVISION,
            local_dir=str(args.output / "mirror"),
        )
    )
    manifest = json.loads(manifest_path.read_text())
    partitions = [
        p for p in manifest["partitions"] if "kitscenes-v35-val-" in p["source_uri"]
    ]
    if len(partitions) != 117 or sum(p["sample_count"] for p in partitions) != 11035:
        raise ValueError("Mirror val partition identity differs")
    partition_counts = {p["partition_id"]: p["sample_count"] for p in partitions}
    entries = [
        entry
        for entry in manifest["shard_entries"]
        if any(entry["name"].startswith(pid + "-") for pid in partition_counts)
    ]
    if len(entries) != 117:
        raise ValueError("Expected one shard per official val partition")
    baseline_timestamps = {}
    for path in sorted(args.baseline.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            baseline_timestamps.update(
                zip(data["sample_uid"].tolist(), data["timestamp_ns"].tolist())
            )
    if (
        len(baseline_timestamps) != 11035
        or published_digest_ids(baseline_timestamps) != EXPECTED_VAL_SAMPLE_UID_SHA256
    ):
        raise ValueError(
            "Baseline sample identity is not the published official val population"
        )

    def fetch(entry):
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("Unsafe shard filename")
        path = Path(
            hf_hub_download(
                MIRROR_REPO,
                PREFIX + name,
                revision=MIRROR_REVISION,
                local_dir=str(args.output / "mirror"),
            )
        )
        if path.stat().st_size != entry["byte_size"]:
            raise ValueError(f"Shard size mismatch: {name}")
        records = extract_shard(path)
        pid = next(pid for pid in partition_counts if name.startswith(pid + "-"))
        if len(records) != partition_counts[pid]:
            raise ValueError(f"Shard sample count mismatch: {name}")
        return name, file_sha256(path), records

    records, shard_digests = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, digest, shard_records) in enumerate(
            pool.map(fetch, entries), 1
        ):
            if set(records) & set(shard_records):
                raise ValueError("Duplicate route sample UIDs")
            records.update(shard_records)
            shard_digests[name] = digest
            print(
                f"Route metadata: {index}/117 shards, {len(records)} samples",
                flush=True,
            )
    if set(records) != set(baseline_timestamps):
        raise ValueError("Route command sample set differs from baseline")
    timestamp_error = max(
        abs(records[uid]["sample_timestamp_ns"] - baseline_timestamps[uid])
        for uid in records
    )
    if timestamp_error != 0:
        raise ValueError(
            "Published route timestamps differ from the audited sample timestamps"
        )
    commands = {uid: record["command"] for uid, record in sorted(records.items())}
    write_json(args.output / "commands.json", commands)
    write_json(args.output / "sample_route_provenance.json", records)
    summary = {
        "schema_version": "autoe2e_route_maneuver_to_navsim_command_v1",
        "mirror_repo": MIRROR_REPO,
        "mirror_revision": MIRROR_REVISION,
        "manifest_sha256": file_sha256(manifest_path),
        "command_order": COMMAND_ORDER,
        "mapping": "valid published left/straight/right -> matching one-hot; invalid or ambiguous -> unknown one-hot",
        "command_policy": "autoe2e_published_route_maneuver_100m_v1",
        "native_navsim_policy": "route centerline at 20m with +/-2m lateral threshold; not reproduced by this adaptation",
        "route_provenance": "Published AutoE2E map-matched scene routes, reconstructed offline from scene traces; no new reconstruction from future evaluation targets",
        "uses_future_ground_truth_for_command_extraction": False,
        "scene_count": 117,
        "sample_count": len(records),
        "published_sample_uid_sha256": published_digest_ids(records),
        "command_counts": dict(Counter(r["category"] for r in records.values())),
        "original_maneuver_counts": dict(
            Counter(r["route_maneuver"] for r in records.values())
        ),
        "invalid_route_sample_count": sum(
            not r["route_valid"] for r in records.values()
        ),
        "maximum_route_to_raw_timestamp_error_ns": timestamp_error,
        "commands_sha256": file_sha256(args.output / "commands.json"),
        "source_shards_sha256": shard_digests,
    }
    write_json(args.output / "commands.provenance.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".benchmark/route_inputs"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".benchmark/population"),
        help="Identity-only NPZs produced by prepare_inputs; predictions are not needed",
    )
    parser.add_argument("--workers", type=int, default=4)
    prepare(parser.parse_args())
