#!/usr/bin/env python3
"""Recover and validate reusable KITScenes raw/label Flyte artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from Platform.pipelines.kitscenes_recovery import (  # noqa: E402
    KNOWN_MISSING_TRAIN_SCENE,
    RECOVERY_MANIFEST_SCHEMA,
    artifact_set_sha256,
)


DATASET = "KIT-MRT/KITScenes-Multimodal"
SOURCE_REVISION = "6fde0034446669e2ed7235e4c7fe323cd23d599d"
PROMPT_VERSION = "action_relevant_reasoning_v3_temporal_front256"
LABEL_POLICY_VERSION = "v2"
TEACHER = "openai_compatible"
TEACHER_MODEL = "nvidia/Cosmos3-Nano"
_EXECUTION_ID_RE = re.compile(r"^[a-z0-9]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover ordered raw/Cosmos-label artifacts from a completed "
            "KITScenes data-preparation node."
        )
    )
    parser.add_argument("--metadata-bucket", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--expected-artifact-set-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upload-s3-uri")
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", "autowarefoundation"),
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
    )
    parser.add_argument(
        "--namespace",
        default="auto-e2e-development",
    )
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def _assert_dataset_node_succeeded(
    execution_id: str,
    namespace: str,
) -> None:
    command = [
        "kubectl",
        "-n",
        namespace,
        "get",
        "flyteworkflow",
        execution_id,
        "-o",
        "json",
    ]
    workflow = json.loads(subprocess.check_output(command, text=True))
    status = workflow.get("status", {})
    node = status.get("nodeStatus", {}).get("n0", {})
    if node.get("phase") != 5 or node.get("error"):
        raise RuntimeError(
            f"execution {execution_id} dataset node n0 did not succeed: "
            f"{node}"
        )
    parent_error = status.get("error", {})
    if status.get("phase") != 5 or "OOMKilled" not in parent_error.get(
        "code", ""
    ):
        raise RuntimeError(
            "the source execution is not the expected training-OOM recovery "
            f"case: phase={status.get('phase')} error={parent_error}"
        )


def _literal_map(s3_client, bucket: str, key: str):
    try:
        from flyteidl.core import literals_pb2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "flyteidl is required; install the repository's Flyte tooling "
            "dependencies before running this script"
        ) from error

    payload = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    result = literals_pb2.LiteralMap()
    result.ParseFromString(payload)
    return result


def _string_collection(literal) -> list[str]:
    return [
        item.scalar.primitive.string_value
        for item in literal.collection.literals
    ]


def _nested_string_collection(literal) -> list[list[str]]:
    return [
        _string_collection(item)
        for item in literal.collection.literals
    ]


def _blob_uri_collection(literal) -> list[str]:
    return [
        item.scalar.blob.uri
        for item in literal.collection.literals
    ]


def _parse_s3_directory_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid S3 directory URI: {uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def _list_level(s3_client, bucket: str, prefix: str) -> tuple[set[str], set[str]]:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
    )
    if response.get("IsTruncated"):
        raise RuntimeError(
            f"unexpectedly large one-level S3 listing at s3://{bucket}/{prefix}"
        )
    objects = {
        item["Key"]
        for item in response.get("Contents", [])
        if item["Key"] != prefix
    }
    directories = {
        item["Prefix"] for item in response.get("CommonPrefixes", [])
    }
    return objects, directories


def _validate_raw_artifact(
    s3_client,
    *,
    metadata_bucket: str,
    scene_id: str,
    raw_uri: str,
) -> None:
    bucket, root = _parse_s3_directory_uri(raw_uri)
    if bucket != metadata_bucket:
        raise ValueError(
            f"raw artifact for {scene_id} uses unexpected bucket {bucket}"
        )
    root = root.rstrip("/") + "/"

    root_objects, root_dirs = _list_level(s3_client, bucket, root)
    if root_objects or root_dirs != {f"{root}data/"}:
        raise ValueError(
            f"raw artifact root for {scene_id} has unexpected contents: "
            f"objects={sorted(root_objects)} dirs={sorted(root_dirs)}"
        )

    data_root = f"{root}data/"
    data_objects, data_dirs = _list_level(s3_client, bucket, data_root)
    if data_objects != {f"{data_root}sequence_archives.csv"}:
        raise ValueError(
            f"raw artifact for {scene_id} lacks its pinned archive manifest"
        )
    if data_dirs != {f"{data_root}train/"}:
        raise ValueError(
            f"raw artifact for {scene_id} has unexpected data directories: "
            f"{sorted(data_dirs)}"
        )

    train_root = f"{data_root}train/"
    train_objects, train_dirs = _list_level(s3_client, bucket, train_root)
    expected_scene_prefix = f"{train_root}{scene_id}/"
    if train_objects or train_dirs != {expected_scene_prefix}:
        raise ValueError(
            f"raw artifact scene mismatch for {scene_id}: "
            f"objects={sorted(train_objects)} dirs={sorted(train_dirs)}"
        )


def _validate_label_artifact(
    s3_client,
    *,
    metadata_bucket: str,
    scene_id: str,
    label_uri: str,
) -> tuple[int, list[str]]:
    bucket, root = _parse_s3_directory_uri(label_uri)
    if bucket != metadata_bucket:
        raise ValueError(
            f"label artifact for {scene_id} uses unexpected bucket {bucket}"
        )
    root = root.rstrip("/") + "/"
    label_root = (
        f"dataset={DATASET}/split=train/"
        f"schema_version=reasoning_label_v2/teacher={TEACHER}/"
    )
    relative_keys: list[str] = []
    continuation_token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": root}
        if continuation_token:
            request["ContinuationToken"] = continuation_token
        response = s3_client.list_objects_v2(**request)
        relative_keys.extend(
            item["Key"][len(root):] for item in response.get("Contents", [])
        )
        if not response.get("IsTruncated"):
            break
        continuation_token = response["NextContinuationToken"]

    expected_keys = {
        "meta.json",
        f"{label_root}records.jsonl",
        f"{label_root}reasoning_labels_v2.jsonl",
        f"{label_root}reasoning_labels_v2.parquet",
    }
    if set(relative_keys) != expected_keys:
        raise ValueError(
            f"label artifact files differ for {scene_id}: "
            f"missing={sorted(expected_keys - set(relative_keys))} "
            f"unexpected={sorted(set(relative_keys) - expected_keys)}"
        )

    meta = json.loads(
        s3_client.get_object(
            Bucket=bucket, Key=f"{root}meta.json"
        )["Body"].read()
    )
    expected_meta = {
        "dataset": DATASET,
        "split": "train",
        "teacher": TEACHER,
        "source_revision": SOURCE_REVISION,
        "prompt_version": PROMPT_VERSION,
        "label_policy_version": LABEL_POLICY_VERSION,
        "num_abstained": 0,
    }
    mismatched = {
        key: (meta.get(key), expected)
        for key, expected in expected_meta.items()
        if meta.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            f"label metadata mismatch for {scene_id}: {mismatched}"
        )

    records_payload = s3_client.get_object(
        Bucket=bucket,
        Key=f"{root}{label_root}records.jsonl",
    )["Body"].read()
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        records_payload.splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid records.jsonl for {scene_id} at line {line_number}"
            ) from error

    if int(meta.get("num_records", -1)) != len(records):
        raise ValueError(
            f"label record count mismatch for {scene_id}: "
            f"meta={meta.get('num_records')} actual={len(records)}"
        )
    if int(meta.get("computed", -1)) != len(records):
        raise ValueError(
            f"label computed count mismatch for {scene_id}: "
            f"meta={meta.get('computed')} actual={len(records)}"
        )

    sample_ids: list[str] = []
    expected_uid_prefix = f"kitscenes-v1-{scene_id}-f"
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id.startswith(expected_uid_prefix):
            raise ValueError(
                f"label {sample_id!r} is not aligned to scene {scene_id}"
            )
        if record.get("dataset_name") != DATASET:
            raise ValueError(
                f"label {sample_id} has wrong dataset_name"
            )
        if record.get("teacher_provider") != TEACHER:
            raise ValueError(
                f"label {sample_id} has wrong teacher_provider"
            )
        if record.get("teacher_model") != TEACHER_MODEL:
            raise ValueError(f"label {sample_id} has wrong teacher_model")
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"label {sample_id} has wrong prompt_version")
        if record.get("abstained") is not False:
            raise ValueError(f"label {sample_id} is abstained")
        horizons = [
            float(horizon.get("horizon_sec", -1))
            for horizon in record.get("horizons", [])
        ]
        if horizons != [0.0, 1.0, 2.0, 3.0, 4.0]:
            raise ValueError(
                f"label {sample_id} has invalid horizon order {horizons}"
            )
        sample_ids.append(sample_id)

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            f"label artifact for {scene_id} contains duplicate sample IDs"
        )
    return len(records), sample_ids


def _validate_one_artifact(
    s3_client,
    metadata_bucket: str,
    record: dict[str, Any],
) -> tuple[int, list[str]]:
    _validate_raw_artifact(
        s3_client,
        metadata_bucket=metadata_bucket,
        scene_id=record["scene_id"],
        raw_uri=record["raw_uri"],
    )
    return _validate_label_artifact(
        s3_client,
        metadata_bucket=metadata_bucket,
        scene_id=record["scene_id"],
        label_uri=record["label_uri"],
    )


def _put_immutable_json(s3_client, uri: str, payload: bytes) -> None:
    from botocore.exceptions import ClientError

    bucket, key = _parse_s3_directory_uri(uri)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        status = error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        )
        code = error.response.get("Error", {}).get("Code")
        if status != 412 and code not in {
            "PreconditionFailed",
            "ConditionalRequestConflict",
        }:
            raise
        existing = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if existing != payload:
            raise RuntimeError(
                f"immutable recovery manifest conflict at {uri}"
            ) from error


def main() -> None:
    args = _parse_args()
    if not _EXECUTION_ID_RE.fullmatch(args.execution_id):
        raise SystemExit(f"invalid Flyte execution id {args.execution_id!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_artifact_set_sha256):
        raise SystemExit("--expected-artifact-set-sha256 must be lowercase hex")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    import boto3

    session = boto3.Session(
        profile_name=args.profile or None,
        region_name=args.region,
    )
    s3_client = session.client("s3")
    _assert_dataset_node_succeeded(args.execution_id, args.namespace)

    metadata_root = (
        "metadata/propeller/"
        f"auto-e2e-development-{args.execution_id}/n0/data/0"
    )
    planned = _literal_map(
        s3_client,
        args.metadata_bucket,
        f"{metadata_root}/n0/0/outputs.pb",
    )
    planned_partitions = _nested_string_collection(planned.literals["o0"])
    if any(len(partition) != 1 for partition in planned_partitions):
        raise RuntimeError("source KITScenes plan is not one scene per partition")
    planned_scene_ids = [
        partition[0] for partition in planned_partitions
    ]

    pack_inputs = _literal_map(
        s3_client,
        args.metadata_bucket,
        f"{metadata_root}/n1/0/dn2/inputs.pb",
    )
    literals = pack_inputs.literals
    groups = _nested_string_collection(literals["group_ids"])
    raw_uris = _blob_uri_collection(literals["raw_data"])
    label_uris = _blob_uri_collection(literals["reasoning_labels"])
    if groups != planned_partitions:
        raise RuntimeError(
            "pack inputs no longer match the successful inventory plan"
        )
    if not (
        len(groups) == len(raw_uris) == len(label_uris) == 533
    ):
        raise RuntimeError(
            "expected 533 atomic raw/label tuples, got "
            f"groups={len(groups)} raw={len(raw_uris)} "
            f"labels={len(label_uris)}"
        )
    if KNOWN_MISSING_TRAIN_SCENE in planned_scene_ids:
        raise RuntimeError("known unavailable KITScenes scene is unexpectedly present")

    scalar_expectations = {
        "dataset": DATASET,
        "source_revision": SOURCE_REVISION,
        "dataset_version": "v2.1",
    }
    for key, expected in scalar_expectations.items():
        actual = literals[key].scalar.primitive.string_value
        if actual != expected:
            raise RuntimeError(
                f"source pack input {key} mismatch: {actual!r}"
            )
    if literals["episodes"].scalar.primitive.integer != 0:
        raise RuntimeError("source pack did not target all selected scenes")
    if literals["image_size"].scalar.primitive.integer != 256:
        raise RuntimeError("source pack image_size was not 256")
    if not literals["world_model"].scalar.primitive.boolean:
        raise RuntimeError("source pack did not include world-model windows")

    entries = [
        {
            "index": index,
            "scene_id": group[0],
            "raw_uri": raw_uri,
            "label_uri": label_uri,
        }
        for index, (group, raw_uri, label_uri) in enumerate(
            zip(groups, raw_uris, label_uris)
        )
    ]
    digest = artifact_set_sha256(entries)
    if digest != args.expected_artifact_set_sha256:
        raise RuntimeError(
            "recovered artifact tuples differ from the audited set: "
            f"expected={args.expected_artifact_set_sha256} actual={digest}"
        )

    all_sample_ids: set[str] = set()
    label_counts: list[int] = [0] * len(entries)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(
                _validate_one_artifact,
                s3_client,
                args.metadata_bucket,
                entry,
            ): index
            for index, entry in enumerate(entries)
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            label_count, sample_ids = future.result()
            duplicates = all_sample_ids.intersection(sample_ids)
            if duplicates:
                raise RuntimeError(
                    "duplicate sample IDs across label artifacts: "
                    f"{sorted(duplicates)[:3]}"
                )
            all_sample_ids.update(sample_ids)
            label_counts[index] = label_count
            completed += 1
            if completed % 25 == 0 or completed == len(entries):
                print(
                    f"Validated {completed}/{len(entries)} raw/label pairs",
                    flush=True,
                )

    for entry, count in zip(entries, label_counts):
        entry["expected_label_count"] = count
    total_labels = sum(label_counts)
    empty_label_partitions = sum(count == 0 for count in label_counts)
    if total_labels != 4598 or empty_label_partitions != 129:
        raise RuntimeError(
            "validated label coverage differs from the audited full run: "
            f"labels={total_labels} empty_partitions={empty_label_partitions}"
        )

    manifest = {
        "schema_version": RECOVERY_MANIFEST_SCHEMA,
        "source_execution_id": args.execution_id,
        "source_dataset_node": "n0",
        "dataset": DATASET,
        "source_revision": SOURCE_REVISION,
        "split": "train",
        "missing_scene_ids": [KNOWN_MISSING_TRAIN_SCENE],
        "artifact_set_sha256": digest,
        "summary": {
            "scene_count": len(entries),
            "label_count": total_labels,
            "empty_label_partitions": empty_label_partitions,
            "abstained_label_count": 0,
        },
        "entries": entries,
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        f"Wrote {output}: scenes={len(entries)} labels={total_labels} "
        f"artifact_set_sha256={digest}"
    )
    if args.upload_s3_uri:
        _put_immutable_json(s3_client, args.upload_s3_uri, payload)
        print(f"Uploaded immutable manifest to {args.upload_s3_uri}")


if __name__ == "__main__":
    main()
