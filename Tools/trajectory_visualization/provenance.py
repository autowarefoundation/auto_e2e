"""Validation for immutable dataset and overlay publication manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from Platform.pipelines.overlay import OVERLAY_SCHEMA


_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_SHA256_LENGTH = 64


def _read_manifest(path: str | Path, label: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(f"{label} manifest exceeds the 8 MiB limit")
    payload = source.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} manifest must be a JSON object")
    return document, hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _single_entry(
    entries: Any,
    *,
    key: str,
    value: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} entries must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get(key) == value
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label} manifest must contain exactly one entry for {value!r}"
        )
    return matches[0]


def validate_report_provenance(
    *,
    dataset_manifest_path: str | Path,
    overlay_manifest_path: str | Path,
    shard_path: str | Path,
    shard_sha256: str,
    overlay_path: str | Path,
    overlay_sha256: str,
    sample_count: int,
    base_seeds: Sequence[int],
) -> dict[str, Any]:
    """Validate report inputs against both immutable publication manifests."""
    dataset_manifest, dataset_manifest_sha256 = _read_manifest(
        dataset_manifest_path,
        "dataset",
    )
    overlay_manifest, overlay_manifest_sha256 = _read_manifest(
        overlay_manifest_path,
        "overlay",
    )
    if dataset_manifest.get("schema_version") != "v2":
        raise ValueError("dataset manifest must use schema_version v2")
    if overlay_manifest.get("schema_version") != "v1":
        raise ValueError("overlay manifest must use schema_version v1")
    if dataset_manifest.get("status") != "ready":
        raise ValueError("dataset manifest is not ready")
    if overlay_manifest.get("status") != "ready":
        raise ValueError("overlay manifest is not ready")

    expected_dataset_digest = _sha256(
        overlay_manifest.get("dataset_manifest_sha256"),
        "overlay dataset_manifest_sha256",
    )
    if expected_dataset_digest != dataset_manifest_sha256:
        raise ValueError("overlay and dataset manifest digests differ")

    dataset = str(dataset_manifest.get("dataset", ""))
    version = str(dataset_manifest.get("version", ""))
    if not dataset or not version:
        raise ValueError("dataset manifest has no dataset/version identity")
    if (
        overlay_manifest.get("dataset") != dataset
        or overlay_manifest.get("version") != version
    ):
        raise ValueError("overlay and dataset publication identities differ")
    if overlay_manifest.get("overlay_binary_schema") != OVERLAY_SCHEMA:
        raise ValueError("overlay manifest has an unsupported binary schema")

    shard = Path(shard_path)
    overlay = Path(overlay_path)
    shard_entry = _single_entry(
        dataset_manifest.get("shard_entries"),
        key="name",
        value=shard.name,
        label="dataset shard",
    )
    rig = shard_entry.get("rig")
    if not isinstance(rig, dict):
        raise ValueError("dataset shard has no rig artifact")
    rig_sha256 = _sha256(rig.get("sha256"), "dataset shard rig sha256")
    expected_rig_key = f"{dataset}/{version}/rig/{rig_sha256}.json"
    if rig.get("key") != expected_rig_key:
        raise ValueError("dataset shard rig key is not canonical")
    overlay_entry = _single_entry(
        overlay_manifest.get("shards"),
        key="shard",
        value=shard.name,
        label="overlay shard",
    )
    if int(shard_entry.get("byte_size", -1)) != shard.stat().st_size:
        raise ValueError("local shard size differs from dataset manifest")
    if _sha256(overlay_entry.get("sha256"), "overlay shard sha256") != (
        overlay_sha256
    ):
        raise ValueError("local overlay SHA-256 differs from overlay manifest")
    if int(overlay_entry.get("byte_size", -1)) != overlay.stat().st_size:
        raise ValueError("local overlay size differs from overlay manifest")
    if int(overlay_entry.get("sample_count", -1)) != sample_count:
        raise ValueError("overlay sample count differs from overlay manifest")
    raw_seeds = overlay_entry.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("overlay shard entry has no seed list")
    manifest_seeds = tuple(int(seed) for seed in raw_seeds)
    if manifest_seeds != tuple(base_seeds):
        raise ValueError("overlay seeds differ from overlay manifest")
    if overlay_manifest.get("seeds") != raw_seeds:
        raise ValueError("overlay set and shard seed lists differ")

    model_artifact_sha256 = _sha256(
        overlay_manifest.get("model_artifact_sha256"),
        "model_artifact_sha256",
    )
    request_identity = _sha256(
        overlay_manifest.get("request_identity"),
        "request_identity",
    )
    cache_identity = _sha256(
        overlay_manifest.get("cache_identity"),
        "cache_identity",
    )
    registered_model_name = str(
        overlay_manifest.get("registered_model_name", "")
    )
    run_id = str(overlay_manifest.get("run_id", ""))
    model_version = int(overlay_manifest.get("model_version", 0))
    if not registered_model_name or not run_id or model_version < 1:
        raise ValueError("overlay manifest has incomplete model identity")
    sampler = str(overlay_manifest.get("sampler", ""))
    inference_contract_version = str(
        overlay_manifest.get("inference_contract_version", "")
    )
    noise_policy_version = str(
        overlay_manifest.get("noise_policy_version", "")
    )
    num_inference_steps = int(
        overlay_manifest.get("num_inference_steps", 0)
    )
    if (
        not sampler
        or not inference_contract_version
        or not noise_policy_version
        or num_inference_steps < 1
    ):
        raise ValueError("overlay manifest has incomplete inference identity")
    return {
        "dataset": {
            "name": dataset,
            "version": version,
            "manifest_name": Path(dataset_manifest_path).name,
            "manifest_sha256": dataset_manifest_sha256,
            "shard_key": str(shard_entry.get("key", "")),
            "shard_content_identity": _sha256(
                shard_entry.get("content_identity"),
                "shard_content_identity",
            ),
            "rig_key": expected_rig_key,
            "rig_sha256": rig_sha256,
            "local_shard_sha256": _sha256(
                shard_sha256,
                "local shard sha256",
            ),
        },
        "model": {
            "registered_model_name": registered_model_name,
            "model_version": model_version,
            "run_id": run_id,
            "artifact_sha256": model_artifact_sha256,
        },
        "overlay": {
            "manifest_name": Path(overlay_manifest_path).name,
            "manifest_sha256": overlay_manifest_sha256,
            "s3_key": str(overlay_entry.get("s3_key", "")),
            "request_identity": request_identity,
            "cache_identity": cache_identity,
            "sampler": sampler,
            "num_inference_steps": num_inference_steps,
            "inference_contract_version": inference_contract_version,
            "noise_policy_version": noise_policy_version,
        },
    }
