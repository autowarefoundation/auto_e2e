"""Checkpoint contracts for resumable imitation-learning runs."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION = "il_checkpoint_v2"


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


def checkpoint_key(run_id: str, epoch: int) -> str:
    if not run_id or "/" in run_id:
        raise ValueError(f"invalid MLflow run id: {run_id!r}")
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    return f"imitation-learning/{run_id}/epoch-{epoch:04d}.pt"


def best_pointer_key(run_id: str) -> str:
    if not run_id or "/" in run_id:
        raise ValueError(f"invalid MLflow run id: {run_id!r}")
    return f"imitation-learning/{run_id}/best.json"


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


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any],
    expected_data_fingerprint: str,
) -> None:
    required = {
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
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"resume checkpoint is missing required fields: {sorted(missing)}"
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported resume checkpoint schema "
            f"{payload['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if stable_digest(payload["config"]) != stable_digest(expected_config):
        raise ValueError("resume checkpoint model/training config does not match")
    if payload["data_fingerprint"] != expected_data_fingerprint:
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


def update_best_pointer(
    s3_client,
    *,
    bucket: str,
    run_id: str,
    epoch: int,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    ade: float,
    fde: float,
) -> str:
    """Update the versioned best pointer after a metric improvement."""
    key = best_pointer_key(run_id)
    body = json.dumps(
        {
            "schema_version": "best_checkpoint_pointer_v1",
            "run_id": run_id,
            "epoch": epoch,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_sha256": checkpoint_sha256,
            "ade": ade,
            "fde": fde,
        },
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
