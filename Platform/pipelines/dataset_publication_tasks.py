"""Flyte tasks that publish partition outputs into the canonical dataset bucket."""

from __future__ import annotations

import os
from typing import List

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from Platform.pipelines.dataset_publication import (
    PUBLICATION_SCHEMA,
    DatasetPublication,
)


ECR_PREFIX = os.environ.get(
    "ECR_PREFIX", "381491877296.dkr.ecr.us-west-2.amazonaws.com"
)
DATA_PREP_IMAGE = os.environ.get(
    "AUTO_E2E_DATA_PREP_IMAGE",
    f"{ECR_PREFIX}/auto-e2e/data-prep:latest",
)
_MAX_COPY_OBJECT_BYTES = 5 * 1024**3


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be provided")
    return value


def _split_s3_uri(uri: str) -> tuple[str, str]:
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"expected an s3:// FlyteDirectory, got {uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def _remote_directory_uri(directory: FlyteDirectory) -> str:
    for candidate in (
        getattr(directory, "remote_source", None),
        getattr(directory, "path", None),
    ):
        if candidate and str(candidate).startswith("s3://"):
            return str(candidate)
    raise ValueError(
        "dataset publication requires an S3-backed FlyteDirectory; "
        "run the workflow remotely"
    )


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", {})
    return str(response.get("Error", {}).get("Code", ""))


def _is_not_found(exc: Exception) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return _error_code(exc) in {"PreconditionFailed", "412"} or status == 412


def _etag_header(value: object, object_uri: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) > 2:
        opaque = raw[1:-1]
    else:
        opaque = raw
    if (
        not opaque
        or '"' in opaque
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in opaque)
    ):
        raise ValueError(f"object has an invalid ETag: {object_uri}")
    return f'"{opaque}"'


def _list_objects(s3, bucket: str, prefix: str) -> list[dict]:
    normalized = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            relative = key[len(normalized):]
            if not relative or relative.startswith("/") or ".." in relative.split("/"):
                raise ValueError(f"unsafe object under FlyteDirectory: {key!r}")
            etag_header = _etag_header(
                item.get("ETag"), f"s3://{bucket}/{key}"
            )
            etag = etag_header[1:-1]
            size = int(item["Size"])
            identity = _content_identity(etag, size)
            objects.append({
                "bucket": bucket,
                "key": key,
                "relative": relative,
                "etag": etag,
                "etag_header": etag_header,
                "size": size,
                "content_identity": identity,
            })
    objects.sort(key=lambda item: item["relative"])
    return objects


def _content_identity(etag: str, size: int) -> str:
    from Platform.pipelines.dataset_publication import sha256_bytes

    return sha256_bytes(f"{etag}:{size}".encode())


def _get_bytes(s3, bucket: str, key: str, *, max_bytes: int | None = None) -> bytes:
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        size = int(response.get("ContentLength", 0))
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"s3://{bucket}/{key} exceeds {max_bytes} bytes")
        body = response["Body"].read()
    finally:
        response["Body"].close()
    if size and len(body) != size:
        raise IOError(
            f"short S3 read for s3://{bucket}/{key}: {len(body)} != {size}"
        )
    return body


def _content_type(key: str) -> str:
    if key.endswith(".tar"):
        return "application/x-tar"
    if key.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if key.endswith(".json"):
        return "application/json"
    if key.endswith(".parquet"):
        return "application/vnd.apache.parquet"
    return "application/octet-stream"


def _copy_immutable(
    s3,
    source: dict,
    *,
    destination_bucket: str,
    destination_key: str,
) -> str:
    from botocore.exceptions import ClientError

    if source["size"] > _MAX_COPY_OBJECT_BYTES:
        raise ValueError(
            f"{source['relative']} is larger than the 5 GiB CopyObject limit; "
            "reduce samples_per_shard before publication"
        )
    metadata = {
        "source-identity": source["content_identity"],
        "source-etag": source["etag"],
        "publication-schema": PUBLICATION_SCHEMA,
    }
    try:
        response = s3.copy_object(
            Bucket=destination_bucket,
            Key=destination_key,
            CopySource={"Bucket": source["bucket"], "Key": source["key"]},
            CopySourceIfMatch=source["etag_header"],
            IfNoneMatch="*",
            Metadata=metadata,
            MetadataDirective="REPLACE",
            ContentType=_content_type(destination_key),
            CacheControl="private, max-age=31536000, immutable",
        )
        return _etag_header(
            response.get("CopyObjectResult", {}).get("ETag"),
            f"s3://{destination_bucket}/{destination_key}",
        )
    except ClientError as exc:
        if not _is_precondition_failed(exc):
            raise

    try:
        existing = s3.head_object(
            Bucket=destination_bucket, Key=destination_key
        )
    except ClientError as exc:
        if _is_not_found(exc):
            raise RuntimeError(
                f"conditional copy failed but destination disappeared: "
                f"s3://{destination_bucket}/{destination_key}"
            ) from exc
        raise
    if (
        int(existing["ContentLength"]) != source["size"]
        or existing.get("Metadata", {}).get("source-identity")
        != source["content_identity"]
    ):
        raise RuntimeError(
            "immutable dataset object already exists with different content: "
            f"s3://{destination_bucket}/{destination_key}"
        )
    return _etag_header(
        existing.get("ETag"),
        f"s3://{destination_bucket}/{destination_key}",
    )


def _put_immutable(
    s3,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
    content_encoding: str | None = None,
) -> str:
    from botocore.exceptions import ClientError
    from Platform.pipelines.dataset_publication import sha256_bytes

    digest = sha256_bytes(payload)
    request = {
        "Bucket": bucket,
        "Key": key,
        "Body": payload,
        "IfNoneMatch": "*",
        "ContentType": content_type,
        "CacheControl": "private, max-age=31536000, immutable",
        "Metadata": {
            "sha256": digest,
            "publication-schema": PUBLICATION_SCHEMA,
        },
    }
    if content_encoding:
        request["ContentEncoding"] = content_encoding
    try:
        s3.put_object(**request)
        return digest
    except ClientError as exc:
        if not _is_precondition_failed(exc):
            raise

    existing = s3.head_object(Bucket=bucket, Key=key)
    if (
        int(existing["ContentLength"]) != len(payload)
        or existing.get("Metadata", {}).get("sha256") != digest
    ):
        raise RuntimeError(
            "immutable dataset metadata already exists with different content: "
            f"s3://{bucket}/{key}"
        )
    return digest


def _assert_compatible_or_absent(
    s3,
    *,
    bucket: str,
    key: str,
    byte_size: int,
    sha256: str,
) -> None:
    from botocore.exceptions import ClientError

    try:
        existing = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_not_found(exc):
            return
        raise
    if (
        int(existing["ContentLength"]) != byte_size
        or existing.get("Metadata", {}).get("sha256") != sha256
    ):
        raise RuntimeError(
            "immutable dataset manifest already exists with different content: "
            f"s3://{bucket}/{key}"
        )


def _geo_inventory(s3, objects_by_relative: dict[str, dict]) -> dict | None:
    import json
    import math

    import numpy as np

    summary_source = objects_by_relative.get("geo/summary.json")
    if summary_source is None:
        return None
    summary = json.loads(
        _get_bytes(
            s3,
            summary_source["bucket"],
            summary_source["key"],
            max_bytes=1 << 20,
        )
    )
    privacy = summary.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("geo summary has no privacy policy")
    grid_deg = float(privacy["heatmap_grid_degrees"])
    if grid_deg <= 0:
        raise ValueError("heatmap grid must be positive")

    paths = []
    cell_stats: dict[tuple[int, int], dict[str, int]] = {}
    for relative, source in sorted(objects_by_relative.items()):
        if not relative.startswith("geo/episode_paths/"):
            continue
        filename = relative.removeprefix("geo/episode_paths/")
        payload = _get_bytes(s3, source["bucket"], source["key"])
        if not payload or len(payload) % (4 * 8):
            raise ValueError(f"invalid float64 episode path {relative!r}")
        rows = np.frombuffer(payload, dtype="<f8").reshape(-1, 4)
        episode_cells: set[tuple[int, int]] = set()
        point_count = 0
        for row in rows:
            lat, lon = float(row[0]), float(row[1])
            if (
                not math.isfinite(lat)
                or not math.isfinite(lon)
                or not -90 <= lat <= 90
                or not -180 <= lon <= 180
                or (lat == 0 and lon == 0)
            ):
                raise ValueError(f"invalid coordinate in {relative!r}")
            cell = (math.floor(lat / grid_deg), math.floor(lon / grid_deg))
            stats = cell_stats.setdefault(
                cell, {"sample_count": 0, "episode_count": 0}
            )
            stats["sample_count"] += 1
            episode_cells.add(cell)
            point_count += 1
        for cell in episode_cells:
            cell_stats[cell]["episode_count"] += 1
        paths.append({
            "filename": filename,
            "point_count": point_count,
            "key": source["destination_key"],
            "byte_size": source["size"],
            "content_identity": source["content_identity"],
        })

    parquet = objects_by_relative.get("geo/sample_pose.parquet")
    sample_pose_count = int(summary.get("sample_pose_count", 0))
    if sample_pose_count and parquet is None:
        raise ValueError("geo summary declares sample poses but parquet is absent")
    return {
        "privacy": privacy,
        "source_coordinate_dtype": summary["source_coordinate_dtype"],
        "stored_coordinate_dtype": summary["stored_coordinate_dtype"],
        "timestamp_dtype": summary["timestamp_dtype"],
        "gps_accuracy_available": bool(
            summary.get("gps_accuracy_available", False)
        ),
        "sample_pose_count": sample_pose_count,
        "sample_pose_source": (
            {
                key: parquet[key]
                for key in (
                    "bucket",
                    "key",
                    "etag",
                    "size",
                    "content_identity",
                )
            }
            if parquet is not None
            else None
        ),
        "episode_paths": paths,
        "cells": [
            {
                "lat_cell": key[0],
                "lon_cell": key[1],
                **counts,
            }
            for key, counts in sorted(cell_stats.items())
        ],
    }


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="4", mem="8Gi", ephemeral_storage="20Gi"),
    limits=Resources(cpu="4", mem="8Gi", ephemeral_storage="20Gi"),
    retries=2,
)
def publish_dataset_partition(
    shard_dir: FlyteDirectory,
    published_dataset: str,
    dataset_version: str,
    datasets_bucket: str,
    aws_region: str = "us-west-2",
    copy_workers: int = 16,
) -> FlyteFile:
    """Copy one remote partition without downloading its shard or frame pool."""
    import hashlib
    import json
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import boto3

    from Platform.pipelines.dataset_publication import (
        canonical_json_bytes,
        dataset_prefix,
        episode_path_key,
        pool_key,
        sha256_bytes,
        shard_key,
    )

    _required(datasets_bucket, "datasets_bucket")
    if copy_workers < 1 or copy_workers > 64:
        raise ValueError("copy_workers must be between 1 and 64")
    prefix = dataset_prefix(published_dataset, dataset_version)
    source_uri = _remote_directory_uri(shard_dir)
    source_bucket, source_prefix = _split_s3_uri(source_uri)
    s3 = boto3.client("s3", region_name=aws_region)
    objects = _list_objects(s3, source_bucket, source_prefix)
    by_relative = {item["relative"]: item for item in objects}
    if len(by_relative) != len(objects):
        raise ValueError(f"duplicate relative objects under {source_uri}")

    source_manifest = by_relative.get("manifest.json")
    rig_source = by_relative.get("rig/projection.json")
    if source_manifest is None or rig_source is None:
        raise ValueError(f"partition {source_uri} lacks manifest or rig projection")
    manifest_payload = _get_bytes(
        s3,
        source_bucket,
        source_manifest["key"],
        max_bytes=4 << 20,
    )
    manifest = json.loads(manifest_payload)
    if manifest.get("dataset_version") != dataset_version:
        raise ValueError(
            f"partition version {manifest.get('dataset_version')!r} differs "
            f"from publication {dataset_version!r}"
        )
    rig = json.loads(
        _get_bytes(
            s3, source_bucket, rig_source["key"], max_bytes=4 << 20
        )
    )

    copy_sources = []
    shards = []
    pool_sources = []
    for source in objects:
        relative = source["relative"]
        destination_key = None
        if "/" not in relative and relative.endswith(".tar"):
            destination_key = shard_key(
                published_dataset, dataset_version, relative
            )
            shards.append({
                "name": relative,
                "key": destination_key,
                "byte_size": source["size"],
                "content_identity": source["content_identity"],
            })
        elif relative.startswith("pool/"):
            destination_key = pool_key(
                published_dataset, dataset_version, relative
            )
            pool_sources.append(source)
        elif relative.startswith("geo/episode_paths/"):
            destination_key = episode_path_key(
                published_dataset,
                dataset_version,
                relative.removeprefix("geo/episode_paths/"),
            )
        if destination_key is not None:
            source["destination_key"] = destination_key
            copy_sources.append(source)

    expected_names = sorted(manifest.get("shard_names", []))
    actual_names = sorted(shard["name"] for shard in shards)
    if expected_names != actual_names:
        raise ValueError(
            f"partition shard inventory differs: {expected_names} != {actual_names}"
        )
    if (
        manifest.get("total_samples", 0)
        and manifest.get("has_world_model", False)
        and not pool_sources
    ):
        raise ValueError("world-model partition has no sibling frame pool")

    with ThreadPoolExecutor(max_workers=copy_workers) as executor:
        destination_etags = list(executor.map(
            lambda source: _copy_immutable(
                s3,
                source,
                destination_bucket=datasets_bucket,
                destination_key=source["destination_key"],
            ),
            copy_sources,
        ))
    for source, destination_etag in zip(
        copy_sources, destination_etags, strict=True
    ):
        source["destination_etag"] = destination_etag
    for shard in shards:
        shard["etag"] = by_relative[shard["name"]]["destination_etag"]

    pool_digest = hashlib.sha256()
    for source in pool_sources:
        pool_digest.update(source["relative"].encode())
        pool_digest.update(b"\0")
        pool_digest.update(source["content_identity"].encode())
        pool_digest.update(b"\n")
    geo = _geo_inventory(s3, by_relative)
    result = {
        "schema_version": PUBLICATION_SCHEMA,
        "source_uri": source_uri.rstrip("/"),
        "source_manifest_sha256": sha256_bytes(manifest_payload),
        "dataset_version": dataset_version,
        "published_prefix": prefix,
        "manifest": manifest,
        "rig": rig,
        "shards": sorted(shards, key=lambda shard: shard["name"]),
        "pool": {
            "object_count": len(pool_sources),
            "byte_size": sum(source["size"] for source in pool_sources),
            "digest": pool_digest.hexdigest(),
        },
        "geo": geo,
    }
    output_dir = Path(tempfile.mkdtemp(prefix="dataset-publication-"))
    result_path = output_dir / "partition.json"
    result_path.write_bytes(canonical_json_bytes(result, pretty=True))
    return FlyteFile(str(result_path))


def _merge_pose_parquet(s3, results: list[dict]) -> tuple[bytes | None, int]:
    import io

    descriptors = []
    expected_rows = 0
    for result in results:
        geo = result.get("geo")
        if not geo:
            continue
        expected_rows += int(geo["sample_pose_count"])
        source = geo.get("sample_pose_source")
        if source is not None:
            descriptors.append((
                str(result["manifest"].get("partition_id") or ""),
                source,
            ))
    if not descriptors:
        if expected_rows:
            raise ValueError("sample pose rows have no parquet sources")
        return None, 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    sample_uids: set[str] = set()
    for _, source in sorted(descriptors):
        payload = _get_bytes(s3, source["bucket"], source["key"])
        table = pq.read_table(pa.BufferReader(payload))
        if "sample_uid" not in table.column_names:
            raise ValueError("sample pose parquet has no sample_uid")
        for value in table.column("sample_uid").to_pylist():
            uid = str(value)
            if uid in sample_uids:
                raise ValueError(f"duplicate sample pose uid {uid}")
            sample_uids.add(uid)
        tables.append(table)
    merged = pa.concat_tables(tables)
    if merged.num_rows != expected_rows:
        raise ValueError(
            f"sample pose row count {merged.num_rows} != {expected_rows}"
        )
    output = io.BytesIO()
    pq.write_table(merged, output, compression="zstd")
    return output.getvalue(), merged.num_rows


@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", ephemeral_storage="20Gi"),
    limits=Resources(cpu="4", mem="16Gi", ephemeral_storage="20Gi"),
    retries=2,
)
def finalize_dataset_publication(
    partition_results: List[FlyteFile],
    published_dataset: str,
    dataset_version: str,
    datasets_bucket: str,
    dynamo_table: str,
    aws_region: str = "us-west-2",
) -> DatasetPublication:
    """Publish merged metadata last, then expose the GEO pointer."""
    import json
    from pathlib import Path

    import boto3

    from Platform.pipelines.dataset_publication import (
        canonical_json_bytes,
        dataset_prefix,
        geo_pointer_item,
        gzip_json_bytes,
        merge_partition_results,
        rig_key,
        sha256_bytes,
    )

    _required(datasets_bucket, "datasets_bucket")
    results = [
        json.loads(Path(result.download()).read_text())
        for result in partition_results
    ]
    manifest, rigs, heatmap = merge_partition_results(
        results,
        dataset=published_dataset,
        version=dataset_version,
    )
    prefix = dataset_prefix(published_dataset, dataset_version)
    s3 = boto3.client("s3", region_name=aws_region)

    for rig_digest, rig in rigs.items():
        rig_payload = canonical_json_bytes(rig, pretty=True)
        if sha256_bytes(rig_payload) != rig_digest:
            raise RuntimeError("rig digest changed during publication")
        _put_immutable(
            s3,
            bucket=datasets_bucket,
            key=rig_key(published_dataset, dataset_version, rig_digest),
            payload=rig_payload,
            content_type="application/json",
        )

    geo_summary = manifest.get("geo")
    if geo_summary is not None:
        parquet_payload, parquet_rows = _merge_pose_parquet(s3, results)
        if parquet_rows != int(geo_summary["sample_pose_count"]):
            raise ValueError("merged geo summary and parquet row counts differ")
        if parquet_payload is not None:
            _put_immutable(
                s3,
                bucket=datasets_bucket,
                key=f"{prefix}/geo/sample_pose.parquet",
                payload=parquet_payload,
                content_type="application/vnd.apache.parquet",
            )
        summary_payload = canonical_json_bytes(geo_summary, pretty=True)
        heatmap_payload = gzip_json_bytes(heatmap)
        _put_immutable(
            s3,
            bucket=datasets_bucket,
            key=f"{prefix}/geo/summary.json",
            payload=summary_payload,
            content_type="application/json",
        )
        _put_immutable(
            s3,
            bucket=datasets_bucket,
            key=f"{prefix}/geo/heatmap.geojson.gz",
            payload=heatmap_payload,
            content_type="application/geo+json",
            content_encoding="gzip",
        )
        manifest["geo_artifacts"] = {
            "summary_key": f"{prefix}/geo/summary.json",
            "heatmap_key": f"{prefix}/geo/heatmap.geojson.gz",
            "sample_pose_key": (
                f"{prefix}/geo/sample_pose.parquet"
                if parquet_payload is not None
                else None
            ),
            "heatmap_sha256": sha256_bytes(heatmap_payload),
        }

    manifest_payload = canonical_json_bytes(manifest, pretty=True)
    manifest_key = f"{prefix}/shards/manifest.json"
    manifest_sha256 = sha256_bytes(manifest_payload)
    _assert_compatible_or_absent(
        s3,
        bucket=datasets_bucket,
        key=manifest_key,
        byte_size=len(manifest_payload),
        sha256=manifest_sha256,
    )
    # A hidden write-once lock serializes concurrent finalizers without making
    # the dataset visible. A retry with the same manifest is idempotent; a
    # different partition set can never replace this immutable version.
    _put_immutable(
        s3,
        bucket=datasets_bucket,
        key=f"{prefix}/shards/.publication-lock.json",
        payload=canonical_json_bytes({
            "schema_version": PUBLICATION_SCHEMA,
            "manifest_sha256": manifest_sha256,
        }, pretty=True),
        content_type="application/json",
    )

    if geo_summary is not None:
        _required(dynamo_table, "dynamo_table")
        item = geo_pointer_item(
            published_dataset,
            dataset_version,
            summary=geo_summary,
            n_samples=int(geo_summary["sample_pose_count"]),
            computed_at=_utc_now(),
            manifest_sha256=manifest_sha256,
        )
        if len(item["summary"].encode()) > 300_000:
            raise ValueError("geo serving summary is too large for DynamoDB")
        table = boto3.resource(
            "dynamodb", region_name=aws_region
        ).Table(dynamo_table)
        table.put_item(Item=item)

    # This is the public gate and MUST remain the final write. Console readers
    # require it before advertising v2.1+; all S3 bodies and Dynamo pointers
    # already exist when the conditional put succeeds.
    written_sha256 = _put_immutable(
        s3,
        bucket=datasets_bucket,
        key=manifest_key,
        payload=manifest_payload,
        content_type="application/json",
    )
    if written_sha256 != manifest_sha256:
        raise RuntimeError("manifest digest changed during publication")

    return DatasetPublication(
        manifest_key=manifest_key,
        manifest_sha256=manifest_sha256,
    )
