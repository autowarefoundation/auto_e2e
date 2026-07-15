"""Flyte tasks for ops-only canonical trajectory-overlay precomputation."""

from __future__ import annotations

import json
import os
from typing import Any, List, Mapping, NamedTuple, Sequence

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from Platform.pipelines.inference import (
    INFERENCE_CONTRACT_VERSION,
    NOISE_POLICY_VERSION,
)
from Platform.pipelines.overlay import OVERLAY_SCHEMA


ECR_PREFIX = os.environ.get(
    "ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com"
)
EVAL_IMAGE = f"{ECR_PREFIX}/auto-e2e/eval:latest"
MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
OVERLAY_CACHE_VERSION = (
    f"overlay-{OVERLAY_SCHEMA}-{INFERENCE_CONTRACT_VERSION}-"
    f"{NOISE_POLICY_VERSION}"
)

ResolvedOverlayModel = NamedTuple(
    "ResolvedOverlayModel",
    checkpoint=FlyteFile,
    metadata=FlyteFile,
)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be provided")
    return value


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", {})
    return str(response.get("Error", {}).get("Code", ""))


def _is_not_found(exc: Exception) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return _error_code(exc) in {"PreconditionFailed", "412"} or status == 412


def _is_conditional_check_failed(exc: Exception) -> bool:
    return _error_code(exc) == "ConditionalCheckFailedException"


def _assert_s3_compatible(
    head: Mapping[str, Any],
    *,
    key: str,
    expected_metadata: Mapping[str, str],
    byte_size: int | None = None,
) -> None:
    metadata = head.get("Metadata", {})
    mismatches = [
        name
        for name, value in expected_metadata.items()
        if metadata.get(name) != value
    ]
    if byte_size is not None and int(head.get("ContentLength", -1)) != byte_size:
        mismatches.append("content-length")
    if mismatches:
        raise RuntimeError(
            "immutable overlay object exists with a different identity "
            f"({', '.join(sorted(mismatches))}): {key}"
        )


def _head_s3_compatible(
    s3,
    *,
    bucket: str,
    key: str,
    expected_metadata: Mapping[str, str],
) -> Mapping[str, Any] | None:
    from botocore.exceptions import ClientError

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_not_found(exc):
            return None
        raise
    _assert_s3_compatible(
        head, key=key, expected_metadata=expected_metadata
    )
    return head


def _put_s3_immutable(
    s3,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    metadata: Mapping[str, str],
    content_type: str,
    content_encoding: str | None = None,
) -> None:
    from botocore.exceptions import ClientError

    request: dict[str, Any] = {
        "Bucket": bucket,
        "Key": key,
        "Body": payload,
        "IfNoneMatch": "*",
        "ContentType": content_type,
        "CacheControl": "private, max-age=31536000, immutable",
        "Metadata": dict(metadata),
    }
    if content_encoding:
        request["ContentEncoding"] = content_encoding
    try:
        s3.put_object(**request)
        return
    except ClientError as exc:
        if not _is_precondition_failed(exc):
            raise

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_not_found(exc):
            raise RuntimeError(
                f"conditional overlay put failed but object disappeared: {key}"
            ) from exc
        raise
    _assert_s3_compatible(
        head,
        key=key,
        expected_metadata=metadata,
        byte_size=len(payload),
    )


def _get_dynamo_item(table, item: Mapping[str, Any]) -> dict[str, Any]:
    response = table.get_item(
        Key={"pk": item["pk"], "sk": item["sk"]},
        ConsistentRead=True,
    )
    existing = response.get("Item")
    if not existing:
        raise RuntimeError(
            "conditional DynamoDB write failed but the item disappeared: "
            f"{item['pk']} / {item['sk']}"
        )
    return existing


def _put_dynamo_immutable(
    table,
    item: Mapping[str, Any],
    *,
    identity_fields: Sequence[str],
) -> dict[str, Any]:
    from botocore.exceptions import ClientError

    try:
        table.put_item(
            Item=dict(item),
            ConditionExpression=(
                "attribute_not_exists(pk) AND attribute_not_exists(sk)"
            ),
        )
        return dict(item)
    except ClientError as exc:
        if not _is_conditional_check_failed(exc):
            raise

    existing = _get_dynamo_item(table, item)
    mismatches = [
        field
        for field in identity_fields
        if existing.get(field) != item.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "immutable DynamoDB item exists with a different identity "
            f"({', '.join(sorted(mismatches))}): "
            f"{item['pk']} / {item['sk']}"
        )
    return existing


def _publish_overlay_set_ready(table, item: Mapping[str, Any]) -> None:
    from botocore.exceptions import ClientError

    try:
        table.put_item(
            Item=dict(item),
            ConditionExpression=(
                "#status = :building "
                "AND request_identity = :request_identity "
                "AND dataset_manifest_sha256 = :dataset_manifest "
                "AND artifacts_bucket = :artifacts_bucket"
            ),
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":building": "building",
                ":request_identity": item["request_identity"],
                ":dataset_manifest": item["dataset_manifest_sha256"],
                ":artifacts_bucket": item["artifacts_bucket"],
            },
        )
        return
    except ClientError as exc:
        if not _is_conditional_check_failed(exc):
            raise

    existing = _get_dynamo_item(table, item)
    fields = (
        "status",
        "request_identity",
        "cache_identity",
        "dataset_manifest_sha256",
        "artifacts_bucket",
        "overlay_schema",
        "seeds",
        "n_shards",
        "n_samples",
        "manifest_key",
        "created_at",
    )
    mismatches = [
        field
        for field in fields
        if existing.get(field) != item.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "ready overlay set is immutable and differs in "
            f"{', '.join(sorted(mismatches))}: {item['pk']}"
        )


def _gate_token(item: Mapping[str, Any], model_artifact_id: str) -> str:
    return json.dumps(
        {
            "artifacts_bucket": item["artifacts_bucket"],
            "created_at": item["created_at"],
            "dataset_manifest_sha256": item[
                "dataset_manifest_sha256"
            ],
            "model_artifact_id": model_artifact_id,
            "overlay_schema": item["overlay_schema"],
            "request_identity": item["request_identity"],
            "status": item["status"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_gate(value: str) -> dict[str, str]:
    try:
        gate = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid overlay-set gate token") from exc
    required = {
        "artifacts_bucket",
        "created_at",
        "dataset_manifest_sha256",
        "model_artifact_id",
        "overlay_schema",
        "request_identity",
        "status",
    }
    if not isinstance(gate, dict) or any(
        not isinstance(gate.get(name), str) or not gate[name]
        for name in required
    ):
        raise ValueError("overlay-set gate token is incomplete")
    if gate["status"] not in {"building", "ready"}:
        raise ValueError("overlay-set gate is not writable")
    return gate


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    cache=True,
    cache_version="overlay-model-resolution-v1",
)
def resolve_overlay_model(
    registered_model_name: str,
    model_version: str,
) -> ResolvedOverlayModel:
    """Resolve an immutable registry version and download its checkpoint."""
    import json
    import tempfile
    from pathlib import Path

    import mlflow
    from mlflow.tracking import MlflowClient

    from Platform.pipelines.inference import sha256_file

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    registered = client.get_model_version(registered_model_name, model_version)
    run = client.get_run(registered.run_id)
    source = str(registered.source).rstrip("/")
    checkpoint_uri = source if source.endswith(".pt") else f"{source}/best.pt"

    output_dir = Path(tempfile.mkdtemp(prefix="overlay-model-"))
    checkpoint_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=checkpoint_uri,
            dst_path=str(output_dir),
        )
    )
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"registry source did not resolve best.pt: {checkpoint_uri}"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    advertised = (
        registered.tags.get("checkpoint_sha256")
        or run.data.tags.get("checkpoint_sha256")
        or run.data.params.get("model/checkpoint_sha256")
    )
    if advertised and advertised != checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 differs from MLflow metadata: "
            f"{checkpoint_sha256} != {advertised}"
        )

    params = run.data.params
    metrics = run.data.metrics
    metadata = {
        "registered_model_name": registered_model_name,
        "model_version": str(registered.version),
        "run_id": str(registered.run_id),
        "artifact_uri": source,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "model_name": params.get("model/backbone", registered_model_name),
        "eval_ade": metrics.get("eval/ade"),
        "eval_fde": metrics.get("eval/fde"),
        "eval_gate_pass": metrics.get("eval/gate_pass"),
        "dataset_source": params.get("data/dataset", ""),
        "dataset_version_source": params.get("data/dataset_version", ""),
        "train_execution_id": params.get("ctx/train_execution_id", ""),
        "val_fraction": params.get("train/val_fraction", "0"),
        "resolved_at": _utc_now(),
    }
    metadata_path = output_dir / "model-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return ResolvedOverlayModel(
        checkpoint=FlyteFile(str(checkpoint_path)),
        metadata=FlyteFile(str(metadata_path)),
    )


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def prepare_overlay_set(
    resolved_metadata: FlyteFile,
    dataset: str,
    dataset_version: str,
    dataset_manifest_digest: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str,
    base_seeds: List[int],
    sampler: str,
) -> str:
    """Write model coordinates and create or resume one compatible set."""
    from pathlib import Path

    import boto3

    from Platform.pipelines.inference import (
        INFERENCE_CONTRACT_VERSION,
        NOISE_POLICY_VERSION,
    )
    from Platform.pipelines.overlay_store import (
        model_profile_item,
        model_version_item,
        overlay_request_identity,
        overlay_set_item,
    )

    for name, value in (
        ("dataset", dataset),
        ("dataset_version", dataset_version),
        ("dataset_manifest_digest", dataset_manifest_digest),
        ("preprocessing_contract_digest", preprocessing_contract_digest),
        ("model_inference_code_digest", model_inference_code_digest),
        ("container_image_digest", container_image_digest),
        ("artifacts_bucket", artifacts_bucket),
        ("dynamo_table", dynamo_table),
    ):
        _required(value, name)
    if sampler != "model-default":
        raise ValueError(
            "only sampler='model-default' is implemented by policy inference"
        )
    metadata = json.loads(Path(resolved_metadata.download()).read_text())
    model_artifact_id = metadata["checkpoint_sha256"]
    request_identity = overlay_request_identity(
        model_artifact_id=model_artifact_id,
        dataset_manifest_digest=dataset_manifest_digest,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        sampler=sampler,
        base_seeds=base_seeds,
        overlay_schema=OVERLAY_SCHEMA,
        inference_contract_version=INFERENCE_CONTRACT_VERSION,
        noise_policy_version=NOISE_POLICY_VERSION,
    )
    created_at = _utc_now()
    table = boto3.resource("dynamodb", region_name=aws_region).Table(dynamo_table)
    profile_metadata = dict(metadata)
    profile_metadata.update({
        "dataset": metadata.get("dataset_source", ""),
        "dataset_version": metadata.get("dataset_version_source", ""),
    })
    profile = model_profile_item(
        model_artifact_id, profile_metadata, created_at=created_at
    )
    _put_dynamo_immutable(
        table,
        profile,
        identity_fields=(
            "pk",
            "sk",
            "registered_model_name",
            "model_version",
            "run_id",
        ),
    )
    version_item = model_version_item(model_artifact_id, metadata)
    _put_dynamo_immutable(
        table,
        version_item,
        identity_fields=(
            "pk",
            "sk",
            "run_id",
            "artifact_uri",
            "checkpoint_sha256",
        ),
    )
    set_item = overlay_set_item(
        model_artifact_id,
        dataset,
        dataset_version,
        status="building",
        seeds=base_seeds,
        overlay_schema=OVERLAY_SCHEMA,
        dataset_manifest_digest=dataset_manifest_digest,
        request_identity=request_identity,
        artifacts_bucket=artifacts_bucket,
        created_at=created_at,
    )
    existing = _put_dynamo_immutable(
        table,
        set_item,
        identity_fields=(
            "pk",
            "sk",
            "dataset_manifest_sha256",
            "request_identity",
            "artifacts_bucket",
            "overlay_schema",
            "seeds",
        ),
    )
    if existing.get("status") not in {"building", "ready"}:
        raise RuntimeError(
            f"overlay set cannot resume from status {existing.get('status')!r}"
        )
    return _gate_token(existing, model_artifact_id)


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="4", mem="16Gi", gpu="1"),
    cache=True,
    cache_version=OVERLAY_CACHE_VERSION,
    cache_serialize=True,
)
def precompute_overlay_partition(
    checkpoint: FlyteFile,
    model_metadata: FlyteFile,
    prepare_gate: str,
    shard_dir: FlyteDirectory,
    dataset: str,
    dataset_version: str,
    dataset_manifest_digest: str,
    preprocessing_contract_digest: str,
    model_inference_code_digest: str,
    container_image_digest: str,
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str,
    base_seeds: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    sampler: str = "model-default",
) -> FlyteFile:
    """Load one checkpoint once and write every tar overlay in one partition."""
    import json
    import tempfile
    from pathlib import Path

    import boto3
    import torch

    from data_parsing.pre_extracted import make_pre_extracted_loader
    from Platform.pipelines.inference import load_policy
    from Platform.pipelines.overlay import (
        overlay_s3_key,
        write_overlay,
    )
    from Platform.pipelines.overlay_precompute import (
        infer_loader_controls,
        planner_is_deterministic,
    )
    from Platform.pipelines.overlay_store import (
        canonical_container_digest,
        overlay_cache_identity,
        overlay_pointer_item,
        overlay_request_identity,
    )

    for name, value in (
        ("dataset_manifest_digest", dataset_manifest_digest),
        ("preprocessing_contract_digest", preprocessing_contract_digest),
        ("model_inference_code_digest", model_inference_code_digest),
        ("container_image_digest", container_image_digest),
        ("artifacts_bucket", artifacts_bucket),
    ):
        _required(value, name)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampler != "model-default":
        raise ValueError(
            "only sampler='model-default' is implemented by policy inference"
        )

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy("file_system")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint.download()
    metadata = json.loads(Path(model_metadata.download()).read_text())
    model, config, model_artifact_id = load_policy(checkpoint_path, device)
    if model_artifact_id != metadata["checkpoint_sha256"]:
        raise ValueError(
            "downloaded checkpoint differs from resolved model metadata"
        )
    gate = _parse_gate(prepare_gate)
    request_identity = overlay_request_identity(
        model_artifact_id=model_artifact_id,
        dataset_manifest_digest=dataset_manifest_digest,
        preprocessing_contract_digest=preprocessing_contract_digest,
        model_inference_code_digest=model_inference_code_digest,
        container_image_digest=container_image_digest,
        sampler=sampler,
        base_seeds=base_seeds,
        overlay_schema=OVERLAY_SCHEMA,
        inference_contract_version=INFERENCE_CONTRACT_VERSION,
        noise_policy_version=NOISE_POLICY_VERSION,
    )
    expected_gate = {
        "artifacts_bucket": artifacts_bucket,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "model_artifact_id": model_artifact_id,
        "overlay_schema": OVERLAY_SCHEMA,
        "request_identity": request_identity,
    }
    mismatches = [
        field
        for field, value in expected_gate.items()
        if gate.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "overlay-set preparation identity differs in "
            + ", ".join(sorted(mismatches))
        )
    metadata.update({
        "dataset": dataset,
        "dataset_version": dataset_version,
    })
    deterministic_planner = planner_is_deterministic(model)
    planner = model.Reactive_E2E.TrajectoryPlanner
    num_inference_steps = int(
        getattr(planner, "num_inference_steps", 1)
    )
    cache_identity = overlay_cache_identity(
        request_identity, num_inference_steps
    )
    actual_seeds = (
        [int(base_seeds[0])]
        if deterministic_planner
        else [int(seed) for seed in base_seeds]
    )
    container_image_digest = canonical_container_digest(
        container_image_digest
    )

    local_dir = Path(shard_dir.download()).resolve()
    tarfiles = sorted(local_dir.glob("*.tar"))
    if not tarfiles:
        raise FileNotFoundError(f"no tar shards in partition {local_dir}")

    s3 = boto3.client("s3", region_name=aws_region)
    table = boto3.resource("dynamodb", region_name=aws_region).Table(dynamo_table)
    output_dir = Path(tempfile.mkdtemp(prefix="overlay-partition-"))
    entries = []

    for tar_path in tarfiles:
        shard_name = tar_path.name
        key = overlay_s3_key(
            model_artifact_id, dataset, dataset_version, shard_name
        )
        object_identity = {
            "base-seeds": ",".join(map(str, actual_seeds)),
            "cache-identity": cache_identity,
            "dataset-manifest-digest": dataset_manifest_digest,
            "model-artifact-id": model_artifact_id,
            "overlay-schema": OVERLAY_SCHEMA,
            "request-identity": request_identity,
            "seed-count": str(len(actual_seeds)),
        }
        head = _head_s3_compatible(
            s3,
            bucket=artifacts_bucket,
            key=key,
            expected_metadata=object_identity,
        )

        if head is not None:
            object_metadata = head.get("Metadata", {})
            artifact_sha = str(object_metadata.get("sha256", ""))
            if len(artifact_sha) != 64:
                raise RuntimeError(f"overlay object has no valid SHA-256: {key}")
            sample_count = int(object_metadata.get("sample-count", "0"))
            if sample_count < 1:
                raise RuntimeError(
                    f"overlay object has no positive sample count: {key}"
                )
            byte_size = int(head["ContentLength"])
        else:
            current_batch_size = batch_size
            while True:
                try:
                    loader = make_pre_extracted_loader(
                        str(local_dir),
                        batch_size=current_batch_size,
                        num_workers=num_workers,
                        shuffle=0,
                        pin_memory=(device.type == "cuda"),
                        shard_files=[tar_path],
                    )
                    sample_uids, controls, v0, actual_seeds_tuple = (
                        infer_loader_controls(
                            model,
                            loader,
                            model_artifact_id=model_artifact_id,
                            dataset_manifest_digest=dataset_manifest_digest,
                            base_seeds=base_seeds,
                            device=device,
                        )
                    )
                    inferred_seeds = list(actual_seeds_tuple)
                    if inferred_seeds != actual_seeds:
                        raise RuntimeError(
                            "planner seed normalization changed during inference"
                        )
                    break
                except torch.cuda.OutOfMemoryError:
                    if current_batch_size == 1:
                        raise
                    current_batch_size = max(1, current_batch_size // 2)
                    torch.cuda.empty_cache()
                    print(
                        f"OOM on {shard_name}; retrying with "
                        f"batch_size={current_batch_size}"
                    )

            artifact = write_overlay(
                output_dir / "overlay.bin.gz",
                sample_uids,
                controls,
                v0,
                base_seeds=actual_seeds,
                deterministic_planner=deterministic_planner,
            )
            artifact_sha = artifact.sha256
            sample_count = artifact.sample_count
            byte_size = artifact.byte_size
            object_metadata = {
                **object_identity,
                "sha256": artifact.sha256,
                "sample-count": str(artifact.sample_count),
            }
            _put_s3_immutable(
                s3,
                bucket=artifacts_bucket,
                key=key,
                payload=artifact.path.read_bytes(),
                metadata=object_metadata,
                content_type="application/octet-stream",
                content_encoding="gzip",
            )

        created_at = gate["created_at"]
        pointer = overlay_pointer_item(
            dataset=dataset,
            version=dataset_version,
            shard=shard_name,
            model_artifact_id=model_artifact_id,
            s3_key=key,
            sha256=artifact_sha,
            byte_size=byte_size,
            sample_count=sample_count,
            overlay_schema=OVERLAY_SCHEMA,
            dataset_manifest_digest=dataset_manifest_digest,
            cache_identity=cache_identity,
            created_at=created_at,
            model_metadata=metadata,
        )
        _put_dynamo_immutable(
            table,
            pointer,
            identity_fields=(
                "pk",
                "sk",
                "s3_key",
                "sha256",
                "byte_size",
                "sample_count",
                "overlay_schema",
                "dataset_manifest_sha256",
                "cache_identity",
                "status",
            ),
        )
        entries.append({
            "shard": shard_name,
            "s3_key": key,
            "sha256": artifact_sha,
            "byte_size": byte_size,
            "sample_count": sample_count,
            "seeds": actual_seeds,
            "created_at": created_at,
        })

    result = {
        "model_artifact_id": model_artifact_id,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "dataset_manifest_digest": dataset_manifest_digest,
        "preprocessing_contract_digest": preprocessing_contract_digest,
        "model_inference_code_digest": model_inference_code_digest,
        "container_image_digest": container_image_digest,
        "request_identity": request_identity,
        "cache_identity": cache_identity,
        "sampler": sampler,
        "num_inference_steps": num_inference_steps,
        "inference_contract_version": INFERENCE_CONTRACT_VERSION,
        "noise_policy_version": NOISE_POLICY_VERSION,
        "overlay_schema": OVERLAY_SCHEMA,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_model": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
        "checkpoint_config": config,
        "entries": entries,
    }
    result_path = output_dir / "partition-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return FlyteFile(str(result_path))


@task(
    container_image=EVAL_IMAGE,
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="1", mem="2Gi"),
)
def finalize_overlay_set(
    model_metadata: FlyteFile,
    partition_results: List[FlyteFile],
    prepare_gate: str,
    dataset: str,
    dataset_version: str,
    dataset_manifest_digest: str,
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str,
) -> str:
    """Write the immutable audit manifest and flip the set to ready last."""
    import hashlib
    import json
    from pathlib import Path

    import boto3

    from Platform.pipelines.overlay_store import overlay_set_item

    metadata = json.loads(Path(model_metadata.download()).read_text())
    gate = _parse_gate(prepare_gate)
    results = [
        json.loads(Path(result.download()).read_text())
        for result in partition_results
    ]
    entries = [entry for result in results for entry in result["entries"]]
    if not entries:
        raise ValueError("overlay run produced no shard entries")
    entries.sort(key=lambda entry: entry["shard"])
    if len({entry["shard"] for entry in entries}) != len(entries):
        raise ValueError("duplicate shard names across overlay partitions")

    model_artifact_id = metadata["checkpoint_sha256"]
    expected_gate = {
        "artifacts_bucket": artifacts_bucket,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "model_artifact_id": model_artifact_id,
        "overlay_schema": OVERLAY_SCHEMA,
    }
    gate_mismatches = [
        field
        for field, value in expected_gate.items()
        if gate.get(field) != value
    ]
    if gate_mismatches:
        raise ValueError(
            "overlay finalizer gate differs in "
            + ", ".join(sorted(gate_mismatches))
        )

    result_fields = (
        "model_artifact_id",
        "dataset",
        "dataset_version",
        "dataset_manifest_digest",
        "preprocessing_contract_digest",
        "model_inference_code_digest",
        "container_image_digest",
        "request_identity",
        "cache_identity",
        "sampler",
        "num_inference_steps",
        "inference_contract_version",
        "noise_policy_version",
        "overlay_schema",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "gpu_model",
    )
    reference = {field: results[0].get(field) for field in result_fields}
    for index, result in enumerate(results):
        mismatches = [
            field
            for field in result_fields
            if result.get(field) != reference[field]
        ]
        if mismatches:
            raise ValueError(
                f"overlay partition {index} differs in "
                + ", ".join(sorted(mismatches))
            )
    expected_result = {
        "model_artifact_id": model_artifact_id,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "dataset_manifest_digest": dataset_manifest_digest,
        "request_identity": gate["request_identity"],
        "overlay_schema": OVERLAY_SCHEMA,
    }
    mismatches = [
        field
        for field, value in expected_result.items()
        if reference.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "overlay result identity differs in "
            + ", ".join(sorted(mismatches))
        )
    actual_seed_sets = {tuple(entry["seeds"]) for entry in entries}
    if len(actual_seed_sets) != 1:
        raise ValueError("overlay partitions used inconsistent seed sets")
    seeds = list(next(iter(actual_seed_sets)))
    output_sha256 = hashlib.sha256(
        json.dumps(
            [(entry["shard"], entry["sha256"]) for entry in entries],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    environment = results[0]
    created_at = gate["created_at"]
    manifest = {
        "schema_version": "v1",
        "status": "ready",
        "registered_model_name": metadata["registered_model_name"],
        "model_version": metadata["model_version"],
        "run_id": metadata["run_id"],
        "model_artifact_sha256": model_artifact_id,
        "dataset": dataset,
        "version": dataset_version,
        "dataset_manifest_sha256": dataset_manifest_digest,
        "request_identity": environment["request_identity"],
        "cache_identity": environment["cache_identity"],
        "n_shards": len(entries),
        "n_samples": sum(entry["sample_count"] for entry in entries),
        "seeds": seeds,
        "sampler": environment["sampler"],
        "num_inference_steps": environment["num_inference_steps"],
        "inference_contract_version": environment[
            "inference_contract_version"
        ],
        "noise_policy_version": environment["noise_policy_version"],
        "overlay_binary_schema": OVERLAY_SCHEMA,
        "preprocessing_contract_digest": environment[
            "preprocessing_contract_digest"
        ],
        "model_inference_code_digest": environment[
            "model_inference_code_digest"
        ],
        "container_image_digest": environment["container_image_digest"],
        "torch_version": environment["torch_version"],
        "cuda_version": environment["cuda_version"],
        "cudnn_version": environment["cudnn_version"],
        "gpu_model": environment["gpu_model"],
        "output_sha256": output_sha256,
        "created_at": created_at,
        "shards": entries,
    }
    manifest_key = (
        f"overlays_manifest/schema=v1/model={model_artifact_id}/"
        f"dataset={dataset}/version={dataset_version}/manifest.json"
    )
    s3 = boto3.client("s3", region_name=aws_region)
    manifest_payload = json.dumps(
        manifest, indent=2, sort_keys=True
    ).encode()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _put_s3_immutable(
        s3,
        bucket=artifacts_bucket,
        key=manifest_key,
        payload=manifest_payload,
        metadata={
            "cache-identity": environment["cache_identity"],
            "dataset-manifest-digest": dataset_manifest_digest,
            "manifest-sha256": manifest_sha256,
            "model-artifact-id": model_artifact_id,
            "output-sha256": output_sha256,
            "overlay-schema": OVERLAY_SCHEMA,
            "request-identity": environment["request_identity"],
        },
        content_type="application/json",
    )

    table = boto3.resource("dynamodb", region_name=aws_region).Table(dynamo_table)
    ready_item = overlay_set_item(
        model_artifact_id,
        dataset,
        dataset_version,
        status="ready",
        seeds=seeds,
        overlay_schema=OVERLAY_SCHEMA,
        dataset_manifest_digest=dataset_manifest_digest,
        request_identity=environment["request_identity"],
        cache_identity=environment["cache_identity"],
        artifacts_bucket=artifacts_bucket,
        created_at=created_at,
        n_shards=len(entries),
        n_samples=manifest["n_samples"],
        manifest_key=manifest_key,
    )
    _publish_overlay_set_ready(table, ready_item)
    return manifest_key
