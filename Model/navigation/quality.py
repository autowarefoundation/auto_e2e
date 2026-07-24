"""Versioned KITScenes route-quality policy and corpus audit."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import RouteQuality


NAVIGATION_QUALITY_AUDIT_VERSION = "navigation_quality_audit_v1"
PACKED_NAVIGATION_QUALITY_AUDIT_VERSION = (
    "packed_navigation_quality_audit_v1"
)


@dataclasses.dataclass(frozen=True)
class NavigationQualityPolicy:
    policy_id: str = "kitscenes_route_quality_v1"
    minimum_matched_pose_ratio: float = 0.80
    maximum_p95_distance_m: float = 5.0
    maximum_p95_heading_error_rad: float = math.radians(45.0)
    require_zero_unresolved_discontinuities: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("quality policy_id must not be empty")
        if not 0.0 <= self.minimum_matched_pose_ratio <= 1.0:
            raise ValueError("minimum_matched_pose_ratio must be in [0,1]")
        limits = (
            self.maximum_p95_distance_m,
            self.maximum_p95_heading_error_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in limits):
            raise ValueError("quality limits must be finite and positive")

    def decision(self, quality: RouteQuality) -> tuple[bool, tuple[str, ...]]:
        reasons = []
        if "no_lane_sequence" in quality.failure_reasons:
            reasons.append("no_lane_sequence")
        if quality.matched_pose_ratio < self.minimum_matched_pose_ratio:
            reasons.append("matched_pose_ratio_below_threshold")
        if quality.p95_lateral_distance_m > self.maximum_p95_distance_m:
            reasons.append("p95_distance_above_threshold")
        if (
            quality.p95_heading_error_rad
            > self.maximum_p95_heading_error_rad
        ):
            reasons.append("p95_heading_error_above_threshold")
        if (
            self.require_zero_unresolved_discontinuities
            and quality.unresolved_discontinuities
        ):
            reasons.append("unresolved_discontinuity")
        return not reasons, tuple(reasons)

    def contract(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT_NAVIGATION_QUALITY_POLICY = NavigationQualityPolicy()


def load_packed_navigation_quality(
    shard_dirs: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and hash-check scene quality artifacts from packed partitions."""
    records = []
    identities = []
    for value in sorted(str(Path(path)) for path in shard_dirs):
        root = Path(value).resolve()
        manifest_path = root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError(
                f"packed manifest is not an object: {manifest_path}"
            )
        if int(manifest.get("total_samples", 0)) <= 0:
            continue
        navigation = manifest.get("navigation")
        if not isinstance(navigation, Mapping):
            raise ValueError(
                f"packed partition has no navigation summary: {root}"
            )
        scenes = navigation.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(
                f"packed partition has no navigation scenes: {root}"
            )
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise ValueError("navigation scene summary must be an object")
            scene_id = str(scene.get("scene_id", ""))
            relative = Path(str(scene.get("path", "")))
            destination = (root / relative).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(
                    "navigation quality path escapes packed partition"
                )
            quality_path = destination / "navigation_quality.json"
            quality_bytes = quality_path.read_bytes()
            hashes = scene.get("hashes")
            if not isinstance(hashes, Mapping):
                raise ValueError("navigation scene summary has no hashes")
            expected_hash = hashes.get("navigation_quality.json")
            actual_hash = hashlib.sha256(quality_bytes).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError(
                    "navigation quality hash mismatch for "
                    f"scene {scene_id!r}"
                )
            quality = json.loads(quality_bytes)
            if not isinstance(quality, dict):
                raise ValueError(
                    "navigation quality artifact must be an object"
                )
            if str(quality.get("scene_id", "")) != scene_id:
                raise ValueError(
                    "navigation quality scene ID differs from manifest"
                )
            if (
                quality.get("quality_policy")
                != DEFAULT_NAVIGATION_QUALITY_POLICY.contract()
            ):
                raise ValueError(
                    "navigation quality artifact uses another policy"
                )
            records.append(quality)
            identities.append({
                "manifest_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
                "navigation_quality_sha256": actual_hash,
                "partition_id": manifest.get("partition_id"),
                "scene_id": scene_id,
                "sample_count": int(manifest["total_samples"]),
            })
    if not records:
        raise ValueError("packed partitions contain no navigation quality")
    return records, identities


def _quality_from_mapping(value: Mapping[str, Any]) -> RouteQuality:
    return RouteQuality(
        matched_pose_ratio=float(value["matched_pose_ratio"]),
        median_lateral_distance_m=float(
            value["median_lateral_distance_m"]
        ),
        p95_lateral_distance_m=float(value["p95_lateral_distance_m"]),
        median_heading_error_rad=float(value["median_heading_error_rad"]),
        p95_heading_error_rad=float(value["p95_heading_error_rad"]),
        shortest_path_fill_count=int(
            value.get("shortest_path_fill_count", 0)
        ),
        shortest_path_fill_length_m=float(
            value.get("shortest_path_fill_length_m", 0.0)
        ),
        adjacent_transition_count=int(
            value.get("adjacent_transition_count", 0)
        ),
        unresolved_discontinuities=int(
            value.get("unresolved_discontinuities", 0)
        ),
        failure_reasons=tuple(value.get("failure_reasons", ())),
    )


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("quality distribution requires finite values")
    return {
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def audit_navigation_quality(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: NavigationQualityPolicy = DEFAULT_NAVIGATION_QUALITY_POLICY,
) -> dict[str, Any]:
    """Aggregate scene quality and verify persisted route validity."""
    if not records:
        raise ValueError("navigation quality audit requires scene records")
    normalized = sorted(
        (dict(record) for record in records),
        key=lambda record: str(record.get("scene_id", "")),
    )
    scene_ids = [str(record.get("scene_id", "")) for record in normalized]
    if any(not scene_id for scene_id in scene_ids):
        raise ValueError("navigation quality record has no scene_id")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("navigation quality audit has duplicate scene IDs")

    qualities = []
    decisions = []
    failure_counts: collections.Counter[str] = collections.Counter()
    sample_count = 0
    for record in normalized:
        raw_quality = record.get("quality")
        if not isinstance(raw_quality, Mapping):
            raise ValueError("navigation quality record has no quality object")
        quality = _quality_from_mapping(raw_quality)
        accepted, reasons = policy.decision(quality)
        persisted_valid = bool(record.get("route_valid", False))
        if persisted_valid != accepted:
            raise ValueError(
                "persisted route validity differs from quality policy for "
                f"scene {record['scene_id']!r}"
            )
        qualities.append(quality)
        failure_counts.update(reasons)
        sample_count += int(record.get("sample_count", 0))
        decisions.append({
            "accepted": accepted,
            "failure_reasons": list(reasons),
            "route_confidence": float(
                record.get("route_confidence", 0.0)
            ),
            "scene_id": str(record["scene_id"]),
            "sample_count": int(record.get("sample_count", 0)),
        })

    accepted_count = sum(
        int(decision["accepted"]) for decision in decisions
    )
    canonical_records = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "schema_version": NAVIGATION_QUALITY_AUDIT_VERSION,
        "policy": policy.contract(),
        "source_records_sha256": hashlib.sha256(
            canonical_records
        ).hexdigest(),
        "scene_count": len(decisions),
        "accepted_scene_count": accepted_count,
        "excluded_scene_count": len(decisions) - accepted_count,
        "accepted_scene_fraction": accepted_count / len(decisions),
        "sample_count": sample_count,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "distributions": {
            "matched_pose_ratio": _distribution(
                [quality.matched_pose_ratio for quality in qualities]
            ),
            "median_lateral_distance_m": _distribution(
                [
                    quality.median_lateral_distance_m
                    for quality in qualities
                ]
            ),
            "p95_lateral_distance_m": _distribution(
                [
                    quality.p95_lateral_distance_m
                    for quality in qualities
                ]
            ),
            "median_heading_error_rad": _distribution(
                [
                    quality.median_heading_error_rad
                    for quality in qualities
                ]
            ),
            "p95_heading_error_rad": _distribution(
                [
                    quality.p95_heading_error_rad
                    for quality in qualities
                ]
            ),
            "route_confidence": _distribution(
                [
                    float(record.get("route_confidence", 0.0))
                    for record in normalized
                ]
            ),
        },
        "scenes": decisions,
    }


def audit_packed_navigation_quality(
    shard_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Audit one-scene KITScenes partitions and bind decisions to hashes."""
    records, identities = load_packed_navigation_quality(shard_dirs)
    identities = sorted(
        identities,
        key=lambda value: (
            str(value.get("partition_id", "")),
            str(value.get("scene_id", "")),
        ),
    )
    partition_ids = [
        str(identity.get("partition_id", "")) for identity in identities
    ]
    if any(not partition_id for partition_id in partition_ids):
        raise ValueError(
            "navigation quality audit requires partition IDs"
        )
    duplicate_partitions = sorted(
        partition_id
        for partition_id, count in collections.Counter(
            partition_ids
        ).items()
        if count != 1
    )
    if duplicate_partitions:
        raise ValueError(
            "navigation quality audit requires exactly one scene per "
            f"partition: {duplicate_partitions[:3]}"
        )

    records_by_scene = {
        str(record.get("scene_id", "")): record for record in records
    }
    for identity in identities:
        scene_id = str(identity["scene_id"])
        record = records_by_scene[scene_id]
        if int(record.get("sample_count", -1)) != int(
            identity["sample_count"]
        ):
            raise ValueError(
                "navigation quality sample count differs from manifest for "
                f"scene {scene_id!r}"
            )

    audit = audit_navigation_quality(records)
    partition_by_scene = {
        str(identity["scene_id"]): str(identity["partition_id"])
        for identity in identities
    }
    decisions = []
    for decision in audit["scenes"]:
        scene_id = str(decision["scene_id"])
        decisions.append({
            **decision,
            "partition_id": partition_by_scene[scene_id],
        })
    accepted_partition_ids = sorted(
        decision["partition_id"]
        for decision in decisions
        if decision["accepted"]
    )
    excluded_partition_ids = sorted(
        decision["partition_id"]
        for decision in decisions
        if not decision["accepted"]
    )
    return {
        **audit,
        "schema_version": PACKED_NAVIGATION_QUALITY_AUDIT_VERSION,
        "quality_audit_schema_version": (
            NAVIGATION_QUALITY_AUDIT_VERSION
        ),
        "packed_artifacts": identities,
        "accepted_partition_ids": accepted_partition_ids,
        "excluded_partition_ids": excluded_partition_ids,
        "scenes": decisions,
    }


def verify_packed_navigation_quality_audit(
    report: Mapping[str, Any],
    shard_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Recompute a packed audit and reject stale or edited reports."""
    if not isinstance(report, Mapping):
        raise ValueError("navigation quality audit must be an object")
    expected = audit_packed_navigation_quality(shard_dirs)
    actual_bytes = json.dumps(
        dict(report),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected_bytes = json.dumps(
        expected,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if actual_bytes != expected_bytes:
        raise ValueError(
            "navigation quality audit differs from packed artifacts"
        )
    return expected
