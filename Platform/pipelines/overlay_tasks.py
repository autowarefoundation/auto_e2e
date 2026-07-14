"""Flyte tasks for ops-only canonical trajectory-overlay precomputation."""

from __future__ import annotations

import os
from typing import List, NamedTuple

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
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str,
    base_seeds: List[int],
) -> str:
    """Write model coordinates and mark the overlay set as building."""
    import json
    from pathlib import Path

    import boto3

    from Platform.pipelines.overlay_store import (
        model_profile_item,
        model_version_item,
        overlay_set_item,
    )

    _required(dataset_manifest_digest, "dataset_manifest_digest")
    _required(artifacts_bucket, "artifacts_bucket")
    metadata = json.loads(Path(resolved_metadata.download()).read_text())
    metadata.update({
        "dataset": dataset,
        "dataset_version": dataset_version,
        "dataset_manifest_digest": dataset_manifest_digest,
        "artifacts_bucket": artifacts_bucket,
    })
    created_at = _utc_now()
    table = boto3.resource("dynamodb", region_name=aws_region).Table(dynamo_table)
    model_artifact_id = metadata["checkpoint_sha256"]
    table.put_item(Item=model_profile_item(
        model_artifact_id, metadata, created_at=created_at
    ))
    table.put_item(Item=model_version_item(model_artifact_id, metadata))
    table.put_item(Item=overlay_set_item(
        model_artifact_id,
        dataset,
        dataset_version,
        status="building",
        seeds=base_seeds,
        overlay_schema=OVERLAY_SCHEMA,
        created_at=created_at,
    ))

    return model_artifact_id


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
    from botocore.exceptions import ClientError

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
    from Platform.pipelines.overlay_store import overlay_pointer_item

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
    if prepare_gate != model_artifact_id:
        raise ValueError("overlay-set preparation identity differs from checkpoint")
    metadata.update({
        "dataset": dataset,
        "dataset_version": dataset_version,
    })
    deterministic_planner = planner_is_deterministic(model)

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
        try:
            head = s3.head_object(Bucket=artifacts_bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            head = None

        if head is not None:
            object_metadata = head.get("Metadata", {})
            identity = {
                "model-artifact-id": model_artifact_id,
                "dataset-manifest-digest": dataset_manifest_digest,
                "overlay-schema": OVERLAY_SCHEMA,
            }
            if any(object_metadata.get(k) != v for k, v in identity.items()):
                raise RuntimeError(
                    f"immutable overlay key exists with different identity: {key}"
                )
            artifact_sha = object_metadata["sha256"]
            sample_count = int(object_metadata["sample-count"])
            byte_size = int(head["ContentLength"])
            actual_seeds = [
                int(value) for value in object_metadata["base-seeds"].split(",")
            ]
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
                    actual_seeds = list(actual_seeds_tuple)
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
            s3.upload_file(
                str(artifact.path),
                artifacts_bucket,
                key,
                ExtraArgs={
                    "ContentType": "application/octet-stream",
                    "ContentEncoding": "gzip",
                    "CacheControl": "private, max-age=31536000, immutable",
                    "Metadata": {
                        "sha256": artifact.sha256,
                        "sample-count": str(artifact.sample_count),
                        "seed-count": str(artifact.seed_count),
                        "base-seeds": ",".join(map(str, actual_seeds)),
                        "overlay-schema": OVERLAY_SCHEMA,
                        "model-artifact-id": model_artifact_id,
                        "dataset-manifest-digest": dataset_manifest_digest,
                    },
                },
            )

        created_at = _utc_now()
        table.put_item(Item=overlay_pointer_item(
            dataset=dataset,
            version=dataset_version,
            shard=shard_name,
            model_artifact_id=model_artifact_id,
            s3_key=key,
            sha256=artifact_sha,
            byte_size=byte_size,
            sample_count=sample_count,
            overlay_schema=OVERLAY_SCHEMA,
            created_at=created_at,
            model_metadata=metadata,
        ))
        entries.append({
            "shard": shard_name,
            "s3_key": key,
            "sha256": artifact_sha,
            "byte_size": byte_size,
            "sample_count": sample_count,
            "seeds": actual_seeds,
            "created_at": created_at,
        })

    planner = model.Reactive_E2E.TrajectoryPlanner
    result = {
        "model_artifact_id": model_artifact_id,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "dataset_manifest_digest": dataset_manifest_digest,
        "preprocessing_contract_digest": preprocessing_contract_digest,
        "model_inference_code_digest": model_inference_code_digest,
        "container_image_digest": container_image_digest,
        "sampler": sampler,
        "num_inference_steps": int(
            getattr(planner, "num_inference_steps", 1)
        ),
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
    dataset: str,
    dataset_version: str,
    dataset_manifest_digest: str,
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str,
) -> str:
    """Write the audit manifest and flip the set to ready last."""
    import hashlib
    import json
    from pathlib import Path

    import boto3

    from Platform.pipelines.overlay_store import overlay_set_item

    metadata = json.loads(Path(model_metadata.download()).read_text())
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
    created_at = _utc_now()
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
    s3.put_object(
        Bucket=artifacts_bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2, sort_keys=True).encode(),
        ContentType="application/json",
        CacheControl="private, max-age=31536000, immutable",
    )

    table = boto3.resource("dynamodb", region_name=aws_region).Table(dynamo_table)
    table.put_item(Item=overlay_set_item(
        model_artifact_id,
        dataset,
        dataset_version,
        status="ready",
        seeds=seeds,
        overlay_schema=OVERLAY_SCHEMA,
        created_at=created_at,
        n_shards=len(entries),
        n_samples=manifest["n_samples"],
        manifest_key=manifest_key,
    ))
    return manifest_key
