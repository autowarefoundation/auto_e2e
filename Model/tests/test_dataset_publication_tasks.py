"""S3 write-once behavior for dataset publication tasks."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from botocore.exceptions import ClientError

pytest.importorskip("flytekit")

from Platform.pipelines.dataset_publication_tasks import (
    _assert_compatible_or_absent,
    _content_identity,
    _copy_immutable,
    _geo_inventory,
    _put_immutable,
)


def _precondition_failed() -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        },
        "PutObject",
    )


class _Body(io.BytesIO):
    pass


class _S3:
    def __init__(self):
        self.copy_calls = []
        self.put_calls = []
        self.head = None
        self.head_error = None
        self.objects = {}
        self.fail_copy = False
        self.fail_put = False

    def copy_object(self, **kwargs):
        self.copy_calls.append(kwargs)
        if self.fail_copy:
            raise _precondition_failed()
        return {"CopyObjectResult": {"ETag": '"destination-etag"'}}

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        if self.fail_put:
            raise _precondition_failed()

    def head_object(self, **kwargs):
        if self.head_error is not None:
            raise self.head_error
        assert self.head is not None
        return self.head

    def get_object(self, *, Bucket, Key):
        payload = self.objects[(Bucket, Key)]
        return {
            "Body": _Body(payload),
            "ContentLength": len(payload),
        }


def _source() -> dict:
    etag = "abc123"
    size = 12
    return {
        "bucket": "flyte-artifacts",
        "key": "raw/output/train.tar",
        "relative": "train.tar",
        "etag": etag,
        "etag_header": f'"{etag}"',
        "size": size,
        "content_identity": _content_identity(etag, size),
    }


def test_copy_uses_destination_and_source_preconditions():
    s3 = _S3()
    source = _source()
    destination_etag = _copy_immutable(
        s3,
        source,
        destination_bucket="datasets",
        destination_key="l2d/v2.1/shards/train.tar",
    )

    request = s3.copy_calls[0]
    assert request["IfNoneMatch"] == "*"
    assert request["CopySourceIfMatch"] == '"abc123"'
    assert request["Metadata"]["source-identity"] == (
        source["content_identity"]
    )
    assert destination_etag == '"destination-etag"'


def test_copy_retry_accepts_only_identical_existing_object():
    s3 = _S3()
    s3.fail_copy = True
    source = _source()
    s3.head = {
        "ContentLength": source["size"],
        "ETag": '"existing-destination-etag"',
        "Metadata": {"source-identity": source["content_identity"]},
    }
    destination_etag = _copy_immutable(
        s3,
        source,
        destination_bucket="datasets",
        destination_key="l2d/v2.1/shards/train.tar",
    )
    assert destination_etag == '"existing-destination-etag"'

    s3.head["Metadata"]["source-identity"] = "different"
    with pytest.raises(RuntimeError, match="different content"):
        _copy_immutable(
            s3,
            source,
            destination_bucket="datasets",
            destination_key="l2d/v2.1/shards/train.tar",
        )


def test_metadata_put_retry_is_content_addressed():
    s3 = _S3()
    payload = b'{"status":"ready"}'
    digest = _put_immutable(
        s3,
        bucket="datasets",
        key="l2d/v2.1/shards/manifest.json",
        payload=payload,
        content_type="application/json",
    )
    assert s3.put_calls[0]["IfNoneMatch"] == "*"
    assert s3.put_calls[0]["Metadata"]["sha256"] == digest

    s3.fail_put = True
    s3.head = {
        "ContentLength": len(payload),
        "Metadata": {"sha256": digest},
    }
    assert _put_immutable(
        s3,
        bucket="datasets",
        key="l2d/v2.1/shards/manifest.json",
        payload=payload,
        content_type="application/json",
    ) == digest


def test_manifest_preflight_rejects_conflict_before_pointer_write():
    s3 = _S3()
    s3.head = {
        "ContentLength": 10,
        "Metadata": {"sha256": "expected"},
    }
    _assert_compatible_or_absent(
        s3,
        bucket="datasets",
        key="l2d/v2.1/shards/manifest.json",
        byte_size=10,
        sha256="expected",
    )

    with pytest.raises(RuntimeError, match="manifest"):
        _assert_compatible_or_absent(
            s3,
            bucket="datasets",
            key="l2d/v2.1/shards/manifest.json",
            byte_size=10,
            sha256="different",
        )

    s3.head_error = ClientError(
        {"Error": {"Code": "404"}},
        "HeadObject",
    )
    _assert_compatible_or_absent(
        s3,
        bucket="datasets",
        key="l2d/v2.1/shards/manifest.json",
        byte_size=10,
        sha256="new",
    )


def test_geo_inventory_keeps_unsuppressed_cells_for_global_reducer():
    s3 = _S3()
    summary = {
        "privacy": {
            "k_anonymity": 5,
            "endpoint_exclusion_frames": 10,
            "heatmap_grid_degrees": 0.01,
        },
        "source_coordinate_dtype": "float32",
        "stored_coordinate_dtype": "float64",
        "timestamp_dtype": "int64_ns",
        "gps_accuracy_available": False,
        "sample_pose_count": 1,
    }
    path = np.array([
        [49.0, 11.0, 90.0, 1.0],
        [49.001, 11.001, 90.0, 2.0],
    ], dtype="<f8").tobytes()
    s3.objects = {
        ("flyte", "out/geo/summary.json"): json.dumps(summary).encode(),
        ("flyte", "out/geo/episode_paths/000001.f64"): path,
    }
    identity = _content_identity("etag", len(path))
    objects = {
        "geo/summary.json": {
            "bucket": "flyte",
            "key": "out/geo/summary.json",
        },
        "geo/sample_pose.parquet": {
            "bucket": "flyte",
            "key": "out/geo/sample_pose.parquet",
            "etag": "parquet",
            "size": 100,
            "content_identity": "parquet-id",
        },
        "geo/episode_paths/000001.f64": {
            "bucket": "flyte",
            "key": "out/geo/episode_paths/000001.f64",
            "etag": "etag",
            "size": len(path),
            "content_identity": identity,
            "destination_key": (
                "l2d/v2.1/geo/episode_paths/000001.f64"
            ),
        },
    }

    geo = _geo_inventory(s3, objects)

    assert geo is not None
    assert geo["episode_paths"][0]["point_count"] == 2
    assert geo["cells"] == [{
        "lat_cell": 4900,
        "lon_cell": 1100,
        "sample_count": 2,
        "episode_count": 1,
    }]
