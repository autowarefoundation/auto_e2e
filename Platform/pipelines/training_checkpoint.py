"""Checkpoint contracts for resumable imitation-learning runs."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


CHECKPOINT_SCHEMA_VERSION = "il_checkpoint_v2"
_RESUME_REQUIRED_FIELDS = {
    "schema_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "epoch",
    "config",
    "training_state",
    "data_fingerprint",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def stable_digest(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def checkpoint_key(run_id: str, epoch: int) -> str:
    if not run_id or "/" in run_id:
        raise ValueError(f"invalid MLflow run id: {run_id!r}")
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    return f"imitation-learning/{run_id}/epoch-{epoch:04d}.pt"


def best_pointer_key(run_id: str, *, role: str = "best") -> str:
    if not run_id or "/" in run_id:
        raise ValueError(f"invalid MLflow run id: {run_id!r}")
    filenames = {
        "best": "best.json",
        "best_trajectory": "best-trajectory.json",
    }
    if role not in filenames:
        raise ValueError(f"invalid best checkpoint role: {role!r}")
    return f"imitation-learning/{run_id}/{filenames[role]}"


def metric_pair_is_better(
    ade: float,
    fde: float,
    best_ade: float,
    best_fde: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Rank checkpoints by ADE first and FDE only when ADE is tied."""
    if ade < best_ade - tolerance:
        return True
    return abs(ade - best_ade) <= tolerance and fde < best_fde - tolerance


def rescale_partial_accumulation_gradients(
    parameters,
    *,
    accumulation_steps: int,
    partial_count: int,
) -> None:
    """Convert a partial window's 1/N-scaled gradients to its own mean."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if not 0 < partial_count <= accumulation_steps:
        raise ValueError(
            "partial_count must be between 1 and accumulation_steps"
        )
    factor = accumulation_steps / partial_count
    if factor == 1.0:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing {sorted(missing)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG device count does not match this runtime"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def validate_resume_envelope(payload: Mapping[str, Any]) -> None:
    """Validate fields needed before interpreting a resume transition."""
    if not isinstance(payload, Mapping):
        raise ValueError("resume checkpoint must be a mapping")
    missing = _RESUME_REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(
            f"resume checkpoint is missing required fields: {sorted(missing)}"
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported resume checkpoint schema "
            f"{payload['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if not isinstance(payload["config"], Mapping):
        raise ValueError("resume checkpoint config must be a mapping")


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_data_fingerprint: str,
    allowed_config_changes: frozenset[str] = frozenset(),
    compatible_data_fingerprints: frozenset[str] = frozenset(),
) -> None:
    validate_resume_envelope(payload)
    saved_config = dict(payload["config"])
    requested_config = dict(expected_config)
    unknown_changes = allowed_config_changes - (
        saved_config.keys() | requested_config.keys()
    )
    if unknown_changes:
        raise ValueError(
            "resume config change allowlist contains unknown fields: "
            f"{sorted(unknown_changes)}"
        )
    for name in allowed_config_changes:
        saved_config.pop(name, None)
        requested_config.pop(name, None)
    if stable_digest(saved_config) != stable_digest(requested_config):
        raise ValueError("resume checkpoint model/training config does not match")
    accepted_fingerprints = {
        expected_data_fingerprint,
        *compatible_data_fingerprints,
    }
    if payload["data_fingerprint"] not in accepted_fingerprints:
        raise ValueError("resume checkpoint dataset fingerprint does not match")
    if int(payload["epoch"]) <= 0:
        raise ValueError("resume checkpoint epoch must be positive")


def upload_immutable_checkpoint(
    s3_client,
    *,
    bucket: str,
    key: str,
    path: str | Path,
) -> dict[str, Any]:
    """Create one immutable S3 checkpoint, accepting an identical retry."""
    from botocore.exceptions import ClientError

    checkpoint_path = Path(path)
    size = checkpoint_path.stat().st_size
    digest = sha256_file(checkpoint_path)
    metadata = {
        "sha256": digest,
        "checkpoint-schema": CHECKPOINT_SCHEMA_VERSION,
    }

    for attempt in range(4):
        try:
            with checkpoint_path.open("rb") as stream:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=stream,
                    ContentType="application/octet-stream",
                    Metadata=metadata,
                    IfNoneMatch="*",
                )
            created = True
            break
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            code = error.response.get("Error", {}).get("Code")
            if status == 409 or code == "ConditionalRequestConflict":
                if attempt == 3:
                    raise
                time.sleep(0.05 * (2**attempt))
                continue
            if status != 412 and code != "PreconditionFailed":
                raise
            existing = s3_client.head_object(Bucket=bucket, Key=key)
            existing_digest = existing.get("Metadata", {}).get("sha256")
            if (
                int(existing.get("ContentLength", -1)) != size
                or existing_digest != digest
            ):
                raise RuntimeError(
                    f"immutable checkpoint conflict at s3://{bucket}/{key}"
                ) from error
            created = False
            break

    return {
        "uri": f"s3://{bucket}/{key}",
        "sha256": digest,
        "size": size,
        "created": created,
    }


def validate_immutable_checkpoint_record(
    s3_client,
    record: Mapping[str, Any],
) -> None:
    """Verify a checkpoint record against immutable S3 object metadata."""
    uri = record.get("uri")
    if not isinstance(uri, str):
        raise ValueError("checkpoint record has no S3 URI")
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("checkpoint record has an invalid S3 URI")
    digest = validate_sha256(
        record.get("sha256"),
        field="checkpoint record sha256",
    )
    head = s3_client.head_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    if head.get("Metadata", {}).get("sha256") != digest:
        raise ValueError(
            "checkpoint record digest differs from immutable S3 metadata"
        )
    content_length = int(head.get("ContentLength", -1))
    if content_length <= 0:
        raise ValueError("checkpoint record points to an empty S3 object")
    if record.get("size") is not None and int(record["size"]) != content_length:
        raise ValueError(
            "checkpoint record size differs from immutable S3 object"
        )


def update_best_pointer(
    s3_client,
    *,
    bucket: str,
    run_id: str,
    role: str = "best",
    epoch: int,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    ade: float,
    fde: float,
    selection: Mapping[str, Any] | None = None,
    metric_contract: Mapping[str, Any] | None = None,
) -> str:
    """Update the versioned best pointer after a metric improvement."""
    key = best_pointer_key(run_id, role=role)
    checkpoint_sha256 = validate_sha256(
        checkpoint_sha256,
        field="checkpoint pointer sha256",
    )
    pointer = {
        "schema_version": (
            "best_checkpoint_pointer_v3"
            if metric_contract is not None
            else (
                "best_checkpoint_pointer_v2"
                if selection is not None
                else "best_checkpoint_pointer_v1"
            )
        ),
        "run_id": run_id,
        "checkpoint_role": role,
        "epoch": epoch,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "ade": ade,
        "fde": fde,
    }
    if selection is not None:
        policy_version = selection.get("policy_version")
        score = selection.get("score")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError(
                "checkpoint selection has no policy version"
            )
        if not isinstance(score, (int, float)):
            raise ValueError(
                "checkpoint selection has no numeric score"
            )
        pointer["selection"] = dict(selection)
    if metric_contract is not None:
        required = {
            "version",
            "horizon_seconds",
            "horizon_steps",
            "target_source",
            "aggregation",
        }
        missing = required - set(metric_contract)
        if missing:
            raise ValueError(
                "checkpoint metric contract is incomplete: "
                f"{sorted(missing)}"
            )
        if (
            float(metric_contract["horizon_seconds"]) <= 0.0
            or int(metric_contract["horizon_steps"]) <= 0
        ):
            raise ValueError(
                "checkpoint metric horizon must be positive"
            )
        pointer["metric_contract"] = dict(metric_contract)
    body = json.dumps(
        pointer,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={"sha256": hashlib.sha256(body).hexdigest()},
    )
    return f"s3://{bucket}/{key}"
