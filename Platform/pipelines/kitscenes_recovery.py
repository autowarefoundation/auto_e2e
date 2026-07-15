"""Contracts for reusing immutable KITScenes raw and label artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


RECOVERY_MANIFEST_SCHEMA = "kitscenes_recovery_manifest_v1"
KNOWN_MISSING_TRAIN_SCENE = "0aef5c74-debd-67ee-c41a-72bb6c82b221"
AUDITED_LABEL_COUNT = 4_598
AUDITED_EMPTY_SCENE_COUNT = 129
_SCENE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _canonical_artifact_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": int(entry["index"]),
        "scene_id": str(entry["scene_id"]),
        "raw_uri": str(entry["raw_uri"]),
        "label_uri": str(entry["label_uri"]),
    }


def artifact_set_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered scene/raw/label tuples as compact, newline-ended JSONL."""
    digest = hashlib.sha256()
    for entry in entries:
        line = json.dumps(
            _canonical_artifact_record(entry),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest.update(line.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_s3_directory_uri(uri: str, field: str) -> None:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field} must be an S3 directory URI, got {uri!r}")


def validate_recovery_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_artifact_set_sha256: str,
    expected_dataset: str,
    expected_source_revision: str,
    expected_scene_ids: Sequence[str],
    expected_label_count: int | None = None,
    expected_empty_scene_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate a recovery manifest before constructing mapped pack nodes."""
    if manifest.get("schema_version") != RECOVERY_MANIFEST_SCHEMA:
        raise ValueError(
            "unsupported recovery manifest schema "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"recovery dataset mismatch: {manifest.get('dataset')!r}"
        )
    if manifest.get("source_revision") != expected_source_revision:
        raise ValueError(
            "recovery source revision mismatch: "
            f"{manifest.get('source_revision')!r}"
        )
    if manifest.get("split") != "train":
        raise ValueError(
            f"recovery manifest must use the train split, got "
            f"{manifest.get('split')!r}"
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("recovery manifest entries must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    seen_raw: set[str] = set()
    seen_labels: set[str] = set()
    for expected_index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"recovery entry {expected_index} must be an object"
            )
        required = {
            "index",
            "scene_id",
            "raw_uri",
            "label_uri",
            "expected_label_count",
        }
        missing = required - set(raw_entry)
        if missing:
            raise ValueError(
                f"recovery entry {expected_index} is missing {sorted(missing)}"
            )

        entry = {
            **_canonical_artifact_record(raw_entry),
            "expected_label_count": int(
                raw_entry["expected_label_count"]
            ),
        }
        if entry["index"] != expected_index:
            raise ValueError(
                "recovery entry indices must be contiguous and ordered: "
                f"position={expected_index} index={entry['index']}"
            )
        scene_id = entry["scene_id"]
        if not _SCENE_ID_RE.fullmatch(scene_id):
            raise ValueError(f"invalid KITScenes scene id {scene_id!r}")
        if entry["expected_label_count"] < 0:
            raise ValueError(
                f"negative label count for scene {scene_id}: "
                f"{entry['expected_label_count']}"
            )
        _require_s3_directory_uri(entry["raw_uri"], "raw_uri")
        _require_s3_directory_uri(entry["label_uri"], "label_uri")

        for value, seen, field in (
            (scene_id, seen_scenes, "scene_id"),
            (entry["raw_uri"], seen_raw, "raw_uri"),
            (entry["label_uri"], seen_labels, "label_uri"),
        ):
            if value in seen:
                raise ValueError(
                    f"duplicate recovery {field} at index {expected_index}: "
                    f"{value}"
                )
            seen.add(value)
        normalized.append(entry)

    actual_label_count = sum(
        entry["expected_label_count"] for entry in normalized
    )
    actual_empty_scene_count = sum(
        entry["expected_label_count"] == 0 for entry in normalized
    )
    if (
        expected_label_count is not None
        and actual_label_count != expected_label_count
    ):
        raise ValueError(
            "recovery label total differs from the audited artifact set: "
            f"expected={expected_label_count} actual={actual_label_count}"
        )
    if (
        expected_empty_scene_count is not None
        and actual_empty_scene_count != expected_empty_scene_count
    ):
        raise ValueError(
            "recovery empty-scene count differs from the audited artifact set: "
            f"expected={expected_empty_scene_count} "
            f"actual={actual_empty_scene_count}"
        )

    expected = list(expected_scene_ids)
    actual = [entry["scene_id"] for entry in normalized]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            "recovery scenes differ from the pinned official inventory: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={missing} unexpected={unexpected}"
        )

    computed_digest = artifact_set_sha256(normalized)
    manifest_digest = str(manifest.get("artifact_set_sha256", ""))
    if manifest_digest != computed_digest:
        raise ValueError(
            "recovery manifest's artifact-set digest is invalid: "
            f"declared={manifest_digest!r} computed={computed_digest}"
        )
    if expected_artifact_set_sha256 != computed_digest:
        raise ValueError(
            "recovery artifact-set digest differs from the launch input: "
            f"expected={expected_artifact_set_sha256} "
            f"computed={computed_digest}"
        )
    return normalized


def load_recovery_manifest(
    path: str | Path,
    **validation_kwargs: Any,
) -> list[dict[str, Any]]:
    with Path(path).open() as stream:
        manifest = json.load(stream)
    return validate_recovery_manifest(manifest, **validation_kwargs)
