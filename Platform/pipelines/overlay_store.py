"""S3/DynamoDB metadata contract for canonical trajectory overlays."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Mapping, Sequence


def shard_model_pk(dataset: str, version: str, shard: str) -> str:
    return f"SHARD#{dataset}#{version}#{shard}"


def model_sk(model_artifact_id: str) -> str:
    return f"MODEL#{model_artifact_id}"


def model_pk(model_artifact_id: str) -> str:
    return model_sk(model_artifact_id)


def model_version_pk(registered_model_name: str, model_version: str | int) -> str:
    return f"MODELVER#{registered_model_name}#{model_version}"


def overlay_set_pk(
    model_artifact_id: str,
    dataset: str,
    version: str,
) -> str:
    return f"OVLSET#{model_artifact_id}#{dataset}#{version}"


def _decimal(value: Any, default: float = 0.0) -> Decimal:
    if value in (None, "", "?"):
        value = default
    return Decimal(str(value))


def _sha256_hex(value: str, name: str) -> str:
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def canonical_container_digest(value: str) -> str:
    """Return an ECR-compatible canonical image digest."""
    return f"sha256:{_sha256_hex(value, 'container_image_digest')}"


def overlay_request_identity(
    *,
    model_artifact_id: str,
    dataset_manifest_digest: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    sampler: str,
    base_seeds: Sequence[int],
    overlay_schema: str,
    inference_contract_version: str,
    noise_policy_version: str,
) -> str:
    """Hash every caller-controlled determinant of an overlay request."""
    seeds = [int(seed) for seed in base_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("base_seeds must be non-empty and unique")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("base_seeds must fit non-negative signed int64")
    for name, value in (
        ("sampler", sampler),
        ("overlay_schema", overlay_schema),
        ("inference_contract_version", inference_contract_version),
        ("noise_policy_version", noise_policy_version),
    ):
        if not value:
            raise ValueError(f"{name} must be provided")

    payload = {
        "base_seeds": seeds,
        "container_image_digest": canonical_container_digest(
            container_image_digest
        ),
        "dataset_manifest_sha256": _sha256_hex(
            dataset_manifest_digest, "dataset_manifest_digest"
        ),
        "inference_contract_version": inference_contract_version,
        "model_artifact_sha256": _sha256_hex(
            model_artifact_id, "model_artifact_id"
        ),
        "model_inference_code_sha256": _sha256_hex(
            model_inference_code_digest, "model_inference_code_digest"
        ),
        "noise_policy_version": noise_policy_version,
        "overlay_schema": overlay_schema,
        "preprocessing_contract_sha256": _sha256_hex(
            preprocessing_contract_digest,
            "preprocessing_contract_digest",
        ),
        "sampler": sampler,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def overlay_cache_identity(
    request_identity: str,
    num_inference_steps: int,
) -> str:
    """Complete the request identity with the checkpoint-derived step count."""
    request_identity = _sha256_hex(request_identity, "request_identity")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    encoded = json.dumps(
        {
            "num_inference_steps": int(num_inference_steps),
            "request_identity": request_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def overlay_pointer_item(
    *,
    dataset: str,
    version: str,
    shard: str,
    model_artifact_id: str,
    s3_key: str,
    sha256: str,
    byte_size: int,
    sample_count: int,
    overlay_schema: str,
    dataset_manifest_digest: str,
    cache_identity: str,
    created_at: str,
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the pointer-only SHARD x MODEL item."""
    return {
        "pk": shard_model_pk(dataset, version, shard),
        "sk": model_sk(model_artifact_id),
        "s3_key": s3_key,
        "sha256": sha256,
        "byte_size": int(byte_size),
        "sample_count": int(sample_count),
        "overlay_schema": overlay_schema,
        "dataset_manifest_sha256": _sha256_hex(
            dataset_manifest_digest, "dataset_manifest_digest"
        ),
        "cache_identity": _sha256_hex(
            cache_identity, "cache_identity"
        ),
        "status": "ready",
        "created_at": created_at,
        "registered_model_name": str(
            model_metadata["registered_model_name"]
        ),
        "model_version": int(model_metadata["model_version"]),
        "run_id": str(model_metadata["run_id"]),
        "model_name": str(model_metadata.get("model_name", "")),
        "eval_ade": _decimal(model_metadata.get("eval_ade")),
        "eval_fde": _decimal(model_metadata.get("eval_fde")),
        "val_fraction": _decimal(model_metadata.get("val_fraction")),
    }


def model_profile_item(
    model_artifact_id: str,
    metadata: Mapping[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    return {
        "pk": model_pk(model_artifact_id),
        "sk": "META",
        "registered_model_name": str(metadata["registered_model_name"]),
        "model_version": int(metadata["model_version"]),
        "run_id": str(metadata["run_id"]),
        "model_name": str(metadata.get("model_name", "")),
        "eval_ade": _decimal(metadata.get("eval_ade")),
        "eval_fde": _decimal(metadata.get("eval_fde")),
        "eval_gate_pass": _decimal(metadata.get("eval_gate_pass")),
        "dataset": str(metadata["dataset"]),
        "dataset_version": str(metadata["dataset_version"]),
        "train_execution_id": str(metadata.get("train_execution_id", "")),
        "val_fraction": _decimal(metadata.get("val_fraction")),
        "created_at": created_at,
    }


def model_version_item(
    model_artifact_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pk": model_version_pk(
            str(metadata["registered_model_name"]),
            metadata["model_version"],
        ),
        "sk": "META",
        "run_id": str(metadata["run_id"]),
        "artifact_uri": str(metadata["artifact_uri"]),
        "checkpoint_sha256": model_artifact_id,
    }


def overlay_set_item(
    model_artifact_id: str,
    dataset: str,
    version: str,
    *,
    status: str,
    seeds: Sequence[int],
    overlay_schema: str,
    dataset_manifest_digest: str,
    request_identity: str,
    artifacts_bucket: str,
    created_at: str,
    cache_identity: str = "",
    n_shards: int = 0,
    n_samples: int = 0,
    manifest_key: str = "",
) -> dict[str, Any]:
    if status not in {"building", "ready", "deleted"}:
        raise ValueError(f"invalid overlay-set status {status!r}")
    if not artifacts_bucket:
        raise ValueError("artifacts_bucket must be provided")
    dataset_manifest_digest = _sha256_hex(
        dataset_manifest_digest, "dataset_manifest_digest"
    )
    request_identity = _sha256_hex(request_identity, "request_identity")
    if status == "ready":
        cache_identity = _sha256_hex(cache_identity, "cache_identity")
    elif cache_identity:
        cache_identity = _sha256_hex(cache_identity, "cache_identity")
    return {
        "pk": overlay_set_pk(model_artifact_id, dataset, version),
        "sk": "META",
        "status": status,
        "n_shards": int(n_shards),
        "n_samples": int(n_samples),
        "seeds": [int(seed) for seed in seeds],
        "manifest_key": manifest_key,
        "overlay_schema": overlay_schema,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "request_identity": request_identity,
        "cache_identity": cache_identity,
        "artifacts_bucket": artifacts_bucket,
        "created_at": created_at,
    }
