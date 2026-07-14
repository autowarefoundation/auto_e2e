"""S3/DynamoDB metadata contract for canonical trajectory overlays."""

from __future__ import annotations

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
    created_at: str,
    n_shards: int = 0,
    n_samples: int = 0,
    manifest_key: str = "",
) -> dict[str, Any]:
    if status not in {"building", "ready", "deleted"}:
        raise ValueError(f"invalid overlay-set status {status!r}")
    return {
        "pk": overlay_set_pk(model_artifact_id, dataset, version),
        "sk": "META",
        "status": status,
        "n_shards": int(n_shards),
        "n_samples": int(n_samples),
        "seeds": [int(seed) for seed in seeds],
        "manifest_key": manifest_key,
        "overlay_schema": overlay_schema,
        "created_at": created_at,
    }
