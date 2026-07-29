"""Flyte task for validation-only model metrics sliced by ODD intervals."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
from flytekit import Resources, task

from Platform.pipelines.odd_metric_projection import (
    METRIC_POLICY_VERSION,
    PROJECTION_POLICY_VERSION,
    PROJECTION_SCHEMA_VERSION,
    MetricSample,
    project_metric_samples,
    projection_cache_identity,
)
from Platform.pipelines.training_checkpoint import stable_digest


ECR_PREFIX = os.environ.get(
    "ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com"
)
DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)
MLFLOW_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
PROJECTION_TASK_ENV = {
    "AUTO_E2E_DATA_PREP_IMAGE": DATA_PREP_IMAGE,
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
}
MAX_JSON_BYTES = 64 << 20
MAX_PARQUET_BYTES = 128 << 20
MAX_OVERLAY_BYTES = 64 << 20
_SHA256_CHARS = frozenset("0123456789abcdef")


def _sha256_hex(value: str, name: str) -> str:
    normalized = str(value).removeprefix("sha256:")
    if (
        len(normalized) != 64
        or normalized.lower() != normalized
        or any(char not in _SHA256_CHARS for char in normalized)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _required(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be provided")
    return value


def _s3_location(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _verified_s3_object(
    client,
    *,
    bucket: str,
    key: str,
    expected_sha256: str,
    expected_size: int | None = None,
    maximum_bytes: int,
) -> bytes:
    expected_sha256 = _sha256_hex(expected_sha256, "expected_sha256")
    response = client.get_object(Bucket=bucket, Key=key)
    advertised_size = int(response.get("ContentLength", -1))
    if advertised_size < 0 or advertised_size > maximum_bytes:
        raise ValueError(f"S3 object exceeds size policy: s3://{bucket}/{key}")
    if expected_size is not None and advertised_size != int(expected_size):
        raise ValueError(f"S3 object advertised size differs: {key}")
    payload = response["Body"].read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"S3 object exceeds size policy: s3://{bucket}/{key}")
    if expected_size is not None and len(payload) != int(expected_size):
        raise ValueError(f"S3 object body size differs: {key}")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"S3 object digest differs for {key}: "
            f"expected={expected_sha256} actual={actual}"
        )
    return payload


def _json_object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _artifact(
    manifest: Mapping[str, Any],
    name: str,
    *,
    required_format: str,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("LabelSet manifest has no artifacts")
    value = artifacts.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"LabelSet manifest has no {name} artifact")
    artifact = dict(value)
    if (
        not artifact.get("key")
        or artifact.get("format") != required_format
        or int(artifact.get("byte_size", 0)) <= 0
    ):
        raise ValueError(f"LabelSet {name} artifact is invalid")
    _sha256_hex(str(artifact.get("sha256", "")), f"{name}.sha256")
    return artifact


def _read_parquet_rows(
    payload: bytes,
    *,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(payload), columns=list(columns))
    return [dict(row) for row in table.to_pylist()]


def _decode_shard_index(item: Mapping[str, Any], shard: str) -> dict[str, Any]:
    payload_attribute = item.get("payload")
    if not isinstance(payload_attribute, Mapping):
        raise ValueError(f"cached shard index has no payload: {shard}")
    compressed = payload_attribute.get("B")
    if not isinstance(compressed, (bytes, bytearray)):
        raise ValueError(f"cached shard index payload is not binary: {shard}")
    try:
        payload = gzip.decompress(compressed)
    except (EOFError, OSError) as exc:
        raise ValueError(f"cached shard index is not valid gzip: {shard}") from exc
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"cached shard index exceeds size policy: {shard}")
    index = _json_object(payload, f"shard index {shard}")
    samples = index.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"cached shard index has no sample list: {shard}")
    stored_shard = str(index.get("shard", ""))
    if stored_shard and stored_shard != shard:
        raise ValueError(f"cached shard index identifies another shard: {shard}")
    return index


def _cached_shard_index(
    client,
    *,
    table_name: str,
    dataset: str,
    version: str,
    shard: str,
) -> dict[str, Any]:
    response = client.get_item(
        TableName=table_name,
        Key={
            "pk": {"S": f"IDX#{dataset}#{version}#{shard}"},
            "sk": {"S": "META"},
        },
        ConsistentRead=True,
    )
    item = response.get("Item")
    if not isinstance(item, Mapping):
        raise ValueError(
            "model metric projection requires a materialized immutable shard "
            f"index: {dataset}/{version}/{shard}"
        )
    return _decode_shard_index(item, shard)


def _index_sample_timestamp(sample: Mapping[str, Any]) -> int:
    pose = sample.get("pose_current")
    if not isinstance(pose, Mapping) or "timestamp_ns" not in pose:
        raise ValueError(
            "evaluation sample has no exact pose timestamp; projection must "
            "not estimate an anchor from frame position"
        )
    timestamp = int(pose["timestamp_ns"])
    if timestamp < 0:
        raise ValueError("evaluation sample timestamp must be non-negative")
    return timestamp


def _index_sample_controls(
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    raw_future = sample.get("ego_future")
    if not isinstance(raw_future, list) or len(raw_future) != 64 * 2:
        raise ValueError("evaluation sample has no 64-step Ground Truth controls")
    ground_truth = np.asarray(raw_future, dtype=np.float64).reshape(64, 2)
    if not np.isfinite(ground_truth).all():
        raise ValueError("Ground Truth controls contain NaN or infinity")

    raw_now = sample.get("ego_now")
    if not isinstance(raw_now, list) or len(raw_now) != 4:
        raise ValueError("evaluation sample has no current ego state")
    initial_speed = float(raw_now[0])
    if not math.isfinite(initial_speed) or initial_speed < 0.0:
        raise ValueError("evaluation sample initial speed is invalid")
    return ground_truth, initial_speed


def _metric_samples_from_shard(
    *,
    overlay_payload: bytes,
    index: Mapping[str, Any],
    scene_uid: str,
    validation_group_uids: frozenset[str],
) -> list[MetricSample]:
    from Platform.pipelines.overlay import (
        decode_overlay,
        sample_uid_hash,
    )

    decoded = decode_overlay(overlay_payload)
    samples = index.get("samples")
    if not isinstance(samples, list):
        raise ValueError("shard index has no samples")
    if len(samples) != decoded.controls.shape[0]:
        raise ValueError("overlay and shard index sample counts differ")

    directory = dict(decoded.directory)
    if len(directory) != len(decoded.directory):
        raise ValueError("overlay directory contains duplicate hashes")
    observed_hashes: set[int] = set()
    output = []
    for raw_sample in samples:
        if not isinstance(raw_sample, Mapping):
            raise ValueError("shard index contains an invalid sample")
        sample_uid = str(raw_sample.get("sample_uid", ""))
        split_group_uid = str(raw_sample.get("split_group_uid", ""))
        if not sample_uid or not split_group_uid:
            raise ValueError("shard index sample identity is incomplete")
        uid_hash = sample_uid_hash(sample_uid)
        row = directory.get(uid_hash)
        if row is None:
            raise ValueError(f"overlay has no row for sample {sample_uid}")
        if uid_hash in observed_hashes:
            raise ValueError("sample UID hash collision in shard index")
        observed_hashes.add(uid_hash)

        ground_truth, initial_speed = _index_sample_controls(raw_sample)
        overlay_speed = float(decoded.v0[row])
        if not math.isclose(
            overlay_speed,
            initial_speed,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"overlay initial speed differs for sample {sample_uid}"
            )
        if split_group_uid not in validation_group_uids:
            continue
        output.append(
            MetricSample(
                sample_uid=sample_uid,
                scene_uid=scene_uid,
                split_group_uid=split_group_uid,
                sample_anchor_timestamp_ns=_index_sample_timestamp(raw_sample),
                predicted_controls=decoded.controls[row],
                ground_truth_controls=ground_truth,
                initial_speed_mps=initial_speed,
            )
        )
    if observed_hashes != set(directory):
        raise ValueError("overlay directory and shard index UID sets differ")
    return output


def _uid_digest(values: Sequence[str]) -> str:
    normalized = sorted(str(value) for value in values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("identity values must be non-empty and unique")
    if any(not value for value in normalized):
        raise ValueError("identity values must not be empty")
    return stable_digest(normalized)


def _validation_contract(
    training_metadata: Mapping[str, Any],
) -> tuple[frozenset[str], int, str, str, str]:
    training = training_metadata.get("training")
    validation = training_metadata.get("validation")
    if not isinstance(training, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("training metadata has no validation contract")
    split = training.get("validation_split")
    if not isinstance(split, Mapping):
        raise ValueError("training metadata has no frozen validation split")
    strategy = str(split.get("strategy", ""))
    split_id = str(split.get("split_id", ""))
    groups = split.get("validation_group_uids")
    if (
        strategy != "exact_group_fraction"
        or not split_id
        or not isinstance(groups, list)
    ):
        raise ValueError("model does not have an exact group validation split")
    normalized_groups = tuple(str(group) for group in groups)
    group_digest = _uid_digest(normalized_groups)
    if group_digest != str(split.get("validation_group_uid_digest", "")):
        raise ValueError("validation group digest differs from its manifest")
    sample_count = int(validation.get("sample_count", 0))
    sample_digest = _sha256_hex(
        str(validation.get("sample_uid_digest", "")),
        "validation.sample_uid_digest",
    )
    if sample_count <= 0:
        raise ValueError("validation sample count must be positive")
    return (
        frozenset(normalized_groups),
        sample_count,
        sample_digest,
        strategy,
        split_id,
    )


def _event_values(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observations_by_uid = {
        str(observation["observation_uid"]): observation
        for observation in observations
    }
    output = []
    for event in events:
        label_key = str(event["primary_event_key"])
        values = {
            str(value)
            for uid in event.get("observation_uids", [])
            for observation in [observations_by_uid.get(str(uid))]
            if observation is not None
            and str(observation.get("label_key", "")) == label_key
            for value in observation.get("values", [])
            if str(value)
        }
        output.append({
            **event,
            "primary_values": sorted(values),
        })
    return output


def _deterministic_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii")


def _deterministic_jsonl_gzip(rows: Sequence[Mapping[str, Any]]) -> bytes:
    plain = io.BytesIO()
    for row in rows:
        plain.write(json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii"))
        plain.write(b"\n")
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=compressed,
        compresslevel=6,
        mtime=0,
    ) as stream:
        stream.write(plain.getvalue())
    return compressed.getvalue()


def _projection_root(
    *,
    odd_dataset: str,
    odd_version: str,
    labelset_id: str,
    model_artifact_id: str,
    projection_identity: str,
) -> str:
    segments = {
        "odd_dataset": odd_dataset,
        "odd_version": odd_version,
        "labelset_id": labelset_id,
        "model_artifact_id": model_artifact_id,
        "projection_identity": projection_identity,
    }
    for name, value in segments.items():
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{name} must be one non-empty path segment")
    return (
        "odd_metric_projections/schema=v1/"
        f"dataset={odd_dataset}/version={odd_version}/"
        f"labelset={labelset_id}/model={model_artifact_id}/"
        f"projection={projection_identity}"
    )


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="12Gi"),
    environment=PROJECTION_TASK_ENV,
    retries=2,
)
def project_odd_model_metrics(
    overlay_manifest_key: str,
    overlay_manifest_sha256: str,
    evaluation_dataset_manifest_uri: str,
    evaluation_dataset_manifest_sha256: str,
    labelset_manifest_key: str,
    labelset_manifest_sha256: str,
    datasets_bucket: str,
    artifacts_bucket: str,
    dynamo_table: str,
    aws_region: str = "us-west-2",
) -> str:
    """Publish validation ADE/FDE slices without changing the LabelSet."""
    import tempfile

    import boto3
    import mlflow
    from mlflow.tracking import MlflowClient

    from Platform.pipelines.overlay_tasks import (
        _put_dynamo_immutable,
        _put_s3_immutable,
    )

    for name, value in (
        ("overlay_manifest_key", overlay_manifest_key),
        ("overlay_manifest_sha256", overlay_manifest_sha256),
        ("evaluation_dataset_manifest_uri", evaluation_dataset_manifest_uri),
        (
            "evaluation_dataset_manifest_sha256",
            evaluation_dataset_manifest_sha256,
        ),
        ("labelset_manifest_key", labelset_manifest_key),
        ("labelset_manifest_sha256", labelset_manifest_sha256),
        ("datasets_bucket", datasets_bucket),
        ("artifacts_bucket", artifacts_bucket),
        ("dynamo_table", dynamo_table),
        ("aws_region", aws_region),
    ):
        _required(value, name)

    overlay_manifest_sha256 = _sha256_hex(
        overlay_manifest_sha256,
        "overlay_manifest_sha256",
    )
    evaluation_dataset_manifest_sha256 = _sha256_hex(
        evaluation_dataset_manifest_sha256,
        "evaluation_dataset_manifest_sha256",
    )
    labelset_manifest_sha256 = _sha256_hex(
        labelset_manifest_sha256,
        "labelset_manifest_sha256",
    )
    s3 = boto3.client("s3", region_name=aws_region)
    dynamo = boto3.client("dynamodb", region_name=aws_region)

    overlay_manifest = _json_object(
        _verified_s3_object(
            s3,
            bucket=artifacts_bucket,
            key=overlay_manifest_key,
            expected_sha256=overlay_manifest_sha256,
            maximum_bytes=MAX_JSON_BYTES,
        ),
        "overlay manifest",
    )
    if overlay_manifest.get("status") != "ready":
        raise ValueError("overlay manifest is not ready")
    model_artifact_id = _sha256_hex(
        str(overlay_manifest.get("model_artifact_sha256", "")),
        "model_artifact_sha256",
    )
    evaluation_dataset = str(overlay_manifest.get("dataset", ""))
    evaluation_version = str(overlay_manifest.get("version", ""))
    overlay_entries = overlay_manifest.get("shards")
    if (
        not evaluation_dataset
        or not evaluation_version
        or not isinstance(overlay_entries, list)
        or not overlay_entries
    ):
        raise ValueError("overlay manifest identity is incomplete")
    if (
        str(overlay_manifest.get("dataset_manifest_sha256", ""))
        != evaluation_dataset_manifest_sha256
    ):
        raise ValueError(
            "overlay and evaluation dataset manifest digests differ"
        )

    evaluation_bucket, evaluation_key = _s3_location(
        evaluation_dataset_manifest_uri
    )
    if evaluation_bucket != datasets_bucket:
        raise ValueError("evaluation dataset manifest must use datasets_bucket")
    evaluation_manifest = _json_object(
        _verified_s3_object(
            s3,
            bucket=evaluation_bucket,
            key=evaluation_key,
            expected_sha256=evaluation_dataset_manifest_sha256,
            maximum_bytes=MAX_JSON_BYTES,
        ),
        "evaluation dataset manifest",
    )
    if (
        evaluation_manifest.get("status") != "ready"
        or evaluation_manifest.get("dataset") != evaluation_dataset
        or evaluation_manifest.get("version") != evaluation_version
        or int(evaluation_manifest.get("total_samples", -1))
        != int(overlay_manifest.get("n_samples", -2))
        or int(evaluation_manifest.get("shards", -1))
        != int(overlay_manifest.get("n_shards", -2))
    ):
        raise ValueError(
            "evaluation dataset and overlay inventory identities differ"
        )
    frequency_hz = int(evaluation_manifest.get("hz", 0))
    if frequency_hz != 10:
        raise ValueError("projection v1 requires a 10 Hz evaluation dataset")

    labelset_manifest = _json_object(
        _verified_s3_object(
            s3,
            bucket=datasets_bucket,
            key=labelset_manifest_key,
            expected_sha256=labelset_manifest_sha256,
            maximum_bytes=MAX_JSON_BYTES,
        ),
        "LabelSet manifest",
    )
    odd_dataset = str(labelset_manifest.get("dataset_name", ""))
    odd_version = str(labelset_manifest.get("dataset_version", ""))
    labelset_id = str(labelset_manifest.get("labelset_id", ""))
    labelset_dataset_manifest_sha256 = _sha256_hex(
        str(labelset_manifest.get("dataset_manifest_sha256", "")),
        "labelset.dataset_manifest_sha256",
    )
    if (
        labelset_manifest.get("status") != "ready"
        or labelset_manifest.get("publication_scope") != "full"
        or not odd_dataset
        or not odd_version
        or not labelset_id
    ):
        raise ValueError("LabelSet must be one complete full publication")

    scene_index_artifact = _artifact(
        labelset_manifest,
        "scene_index",
        required_format="json",
    )
    observations_artifact = _artifact(
        labelset_manifest,
        "observations_parquet",
        required_format="parquet",
    )
    events_artifact = _artifact(
        labelset_manifest,
        "events_parquet",
        required_format="parquet",
    )
    scene_index = _json_object(
        _verified_s3_object(
            s3,
            bucket=datasets_bucket,
            key=str(scene_index_artifact["key"]),
            expected_sha256=str(scene_index_artifact["sha256"]),
            expected_size=int(scene_index_artifact["byte_size"]),
            maximum_bytes=MAX_JSON_BYTES,
        ),
        "ODD scene index",
    )
    if (
        scene_index.get("labelset_id") != labelset_id
        or len(scene_index.get("scenes", []))
        != int(labelset_manifest.get("scene_count", -1))
    ):
        raise ValueError("ODD scene index differs from LabelSet manifest")
    scene_by_shard = {}
    for scene in scene_index["scenes"]:
        shard = str(scene.get("shard_name", ""))
        scene_uid = str(scene.get("scene_uid", ""))
        if not shard or not scene_uid or shard in scene_by_shard:
            raise ValueError("ODD scene index shard ownership is invalid")
        scene_by_shard[shard] = scene_uid

    observation_rows = _read_parquet_rows(
        _verified_s3_object(
            s3,
            bucket=datasets_bucket,
            key=str(observations_artifact["key"]),
            expected_sha256=str(observations_artifact["sha256"]),
            expected_size=int(observations_artifact["byte_size"]),
            maximum_bytes=MAX_PARQUET_BYTES,
        ),
        columns=(
            "observation_uid",
            "scene_uid",
            "label_key",
            "status",
            "values",
            "start_timestamp_ns",
            "end_timestamp_ns",
        ),
    )
    observations = [
        {
            **row,
            "key": row["label_key"],
        }
        for row in observation_rows
    ]
    raw_events = _read_parquet_rows(
        _verified_s3_object(
            s3,
            bucket=datasets_bucket,
            key=str(events_artifact["key"]),
            expected_sha256=str(events_artifact["sha256"]),
            expected_size=int(events_artifact["byte_size"]),
            maximum_bytes=MAX_PARQUET_BYTES,
        ),
        columns=(
            "event_uid",
            "scene_uid",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "primary_event_key",
            "observation_uids",
            "status",
        ),
    )
    events = _event_values(raw_events, observation_rows)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run_id = str(overlay_manifest.get("run_id", ""))
    if not run_id:
        raise ValueError("overlay manifest has no MLflow run ID")
    with tempfile.TemporaryDirectory(prefix="odd-metric-projection-") as tmp:
        metadata_path = MlflowClient().download_artifacts(
            run_id,
            "training/metadata.json",
            dst_path=tmp,
        )
        training_metadata = _json_object(
            Path(metadata_path).read_bytes(),
            "training metadata",
        )
    (
        validation_groups,
        expected_validation_count,
        expected_validation_digest,
        validation_strategy,
        validation_split_id,
    ) = _validation_contract(training_metadata)

    metric_samples = []
    seen_shards = set()
    for entry in sorted(
        overlay_entries,
        key=lambda value: str(value.get("shard", "")),
    ):
        shard = str(entry.get("shard", ""))
        if not shard or shard in seen_shards:
            raise ValueError("overlay manifest contains duplicate shard names")
        seen_shards.add(shard)
        scene_uid = scene_by_shard.get(shard)
        if scene_uid is None:
            raise ValueError(
                f"overlay shard has no LabelSet scene ownership: {shard}"
            )
        overlay_payload = _verified_s3_object(
            s3,
            bucket=artifacts_bucket,
            key=str(entry.get("s3_key", "")),
            expected_sha256=str(entry.get("sha256", "")),
            expected_size=int(entry.get("byte_size", 0)),
            maximum_bytes=MAX_OVERLAY_BYTES,
        )
        index = _cached_shard_index(
            dynamo,
            table_name=dynamo_table,
            dataset=evaluation_dataset,
            version=evaluation_version,
            shard=shard,
        )
        metric_samples.extend(_metric_samples_from_shard(
            overlay_payload=overlay_payload,
            index=index,
            scene_uid=scene_uid,
            validation_group_uids=validation_groups,
        ))
    if seen_shards != set(scene_by_shard):
        raise ValueError(
            "overlay and LabelSet scene shard inventories differ"
        )
    if len(metric_samples) != expected_validation_count:
        raise ValueError(
            "projected validation sample count differs from training: "
            f"expected={expected_validation_count} "
            f"actual={len(metric_samples)}"
        )

    records, summary = project_metric_samples(
        metric_samples,
        observations,
        events,
        frequency_hz=frequency_hz,
    )
    if summary["sample_uid_digest"] != expected_validation_digest:
        raise ValueError(
            "projected validation sample identity differs from training"
        )
    projection_identity = projection_cache_identity(
        model_artifact_sha256=model_artifact_id,
        overlay_manifest_sha256=overlay_manifest_sha256,
        evaluation_dataset_manifest_sha256=(
            evaluation_dataset_manifest_sha256
        ),
        labelset_manifest_sha256=labelset_manifest_sha256,
        labelset_dataset_manifest_sha256=(
            labelset_dataset_manifest_sha256
        ),
        validation_sample_uid_digest=expected_validation_digest,
    )
    report = {
        **summary,
        "status": "ready",
        "projection_id": projection_identity,
        "model": {
            "artifact_sha256": model_artifact_id,
            "registered_model_name": str(
                overlay_manifest.get("registered_model_name", "")
            ),
            "model_version": int(
                overlay_manifest.get("model_version", 0)
            ),
            "run_id": run_id,
        },
        "evaluation_dataset": {
            "dataset": evaluation_dataset,
            "version": evaluation_version,
            "manifest_uri": evaluation_dataset_manifest_uri,
            "manifest_sha256": evaluation_dataset_manifest_sha256,
            "overlay_manifest_key": overlay_manifest_key,
            "overlay_manifest_sha256": overlay_manifest_sha256,
            "overlay_cache_identity": str(
                overlay_manifest.get("cache_identity", "")
            ),
        },
        "labelset": {
            "dataset": odd_dataset,
            "version": odd_version,
            "labelset_id": labelset_id,
            "manifest_key": labelset_manifest_key,
            "manifest_sha256": labelset_manifest_sha256,
            "dataset_manifest_sha256": (
                labelset_dataset_manifest_sha256
            ),
        },
        "validation": {
            "strategy": validation_strategy,
            "split_id": validation_split_id,
            "group_count": len(validation_groups),
            "sample_count": expected_validation_count,
            "sample_uid_digest": expected_validation_digest,
        },
    }
    report_payload = _deterministic_json(report)
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    samples_payload = _deterministic_jsonl_gzip(records)
    samples_sha256 = hashlib.sha256(samples_payload).hexdigest()
    root = _projection_root(
        odd_dataset=odd_dataset,
        odd_version=odd_version,
        labelset_id=labelset_id,
        model_artifact_id=model_artifact_id,
        projection_identity=projection_identity,
    )
    report_key = f"{root}/report.json"
    samples_key = f"{root}/samples.jsonl.gz"
    object_identity = {
        "labelset-manifest-sha256": labelset_manifest_sha256,
        "model-artifact-sha256": model_artifact_id,
        "projection-id": projection_identity,
        "projection-policy": PROJECTION_POLICY_VERSION,
    }
    _put_s3_immutable(
        s3,
        bucket=artifacts_bucket,
        key=report_key,
        payload=report_payload,
        metadata={
            **object_identity,
            "sha256": report_sha256,
        },
        content_type="application/json",
    )
    _put_s3_immutable(
        s3,
        bucket=artifacts_bucket,
        key=samples_key,
        payload=samples_payload,
        metadata={
            **object_identity,
            "sha256": samples_sha256,
        },
        content_type="application/x-ndjson",
        content_encoding="gzip",
    )

    manifest = {
        "schema_version": "odd_model_metric_projection_manifest_v1",
        "status": "ready",
        "projection_id": projection_identity,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "model_artifact_sha256": model_artifact_id,
        "labelset_id": labelset_id,
        "labelset_manifest_sha256": labelset_manifest_sha256,
        "evaluation_dataset_manifest_sha256": (
            evaluation_dataset_manifest_sha256
        ),
        "validation_sample_uid_digest": expected_validation_digest,
        "sample_count": expected_validation_count,
        "artifacts": {
            "report": {
                "key": report_key,
                "sha256": report_sha256,
                "byte_size": len(report_payload),
                "content_type": "application/json",
            },
            "samples": {
                "key": samples_key,
                "sha256": samples_sha256,
                "byte_size": len(samples_payload),
                "content_type": "application/x-ndjson",
                "content_encoding": "gzip",
            },
        },
    }
    manifest_payload = _deterministic_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_key = f"{root}/manifest.json"
    _put_s3_immutable(
        s3,
        bucket=artifacts_bucket,
        key=manifest_key,
        payload=manifest_payload,
        metadata={
            **object_identity,
            "manifest-sha256": manifest_sha256,
        },
        content_type="application/json",
    )

    table = boto3.resource(
        "dynamodb",
        region_name=aws_region,
    ).Table(dynamo_table)
    pointer = {
        "pk": f"ODDPROJ#{odd_dataset}#{odd_version}#{labelset_id}",
        "sk": (
            f"MODEL#{model_artifact_id}#PROJECTION#{projection_identity}"
        ),
        "status": "ready",
        "projection_id": projection_identity,
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "labelset_id": labelset_id,
        "labelset_manifest_sha256": labelset_manifest_sha256,
        "model_artifact_id": model_artifact_id,
        "registered_model_name": report["model"]["registered_model_name"],
        "model_version": report["model"]["model_version"],
        "run_id": run_id,
        "evaluation_dataset": evaluation_dataset,
        "evaluation_version": evaluation_version,
        "evaluation_dataset_manifest_sha256": (
            evaluation_dataset_manifest_sha256
        ),
        "validation_sample_uid_digest": expected_validation_digest,
        "sample_count": expected_validation_count,
        "scene_count": int(summary["scene_count"]),
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "manifest_byte_size": len(manifest_payload),
        "report_key": report_key,
        "report_sha256": report_sha256,
        "report_byte_size": len(report_payload),
        "artifacts_bucket": artifacts_bucket,
    }
    _put_dynamo_immutable(
        table,
        pointer,
        identity_fields=tuple(sorted(pointer)),
    )
    return manifest_key
