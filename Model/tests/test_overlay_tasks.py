"""Concurrency and write-once tests for overlay publication tasks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

pytest.importorskip("flytekit")

from Platform.pipelines.overlay_tasks import (
    OVERLAY_TASK_ENV,
    _gate_token,
    _parse_gate,
    _publish_overlay_set_ready,
    _put_dynamo_immutable,
    _put_s3_immutable,
    _resolve_model_version_for_execution,
)


def test_overlay_tasks_configure_deterministic_cublas_workspace():
    assert OVERLAY_TASK_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def _client_error(code: str, operation: str, status: int = 400) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


class _S3:
    def __init__(self):
        self.put_calls = []
        self.head = None
        self.fail_put = False

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        if self.fail_put:
            raise _client_error("PreconditionFailed", "PutObject", 412)

    def head_object(self, **kwargs):
        assert self.head is not None
        return self.head


class _Table:
    def __init__(self):
        self.put_calls = []
        self.item = None
        self.fail_put = False

    def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        if self.fail_put:
            raise _client_error(
                "ConditionalCheckFailedException", "PutItem"
            )
        self.item = dict(kwargs["Item"])

    def get_item(self, **kwargs):
        return {"Item": self.item} if self.item is not None else {}


def test_s3_put_is_conditional_and_identical_retry_is_accepted():
    payload = b"overlay"
    metadata = {"sha256": "a" * 64, "cache-identity": "b" * 64}
    s3 = _S3()
    _put_s3_immutable(
        s3,
        bucket="artifacts",
        key="overlay.bin.gz",
        payload=payload,
        metadata=metadata,
        content_type="application/octet-stream",
        content_encoding="gzip",
    )
    assert s3.put_calls[0]["IfNoneMatch"] == "*"

    s3.fail_put = True
    s3.head = {"ContentLength": len(payload), "Metadata": metadata}
    _put_s3_immutable(
        s3,
        bucket="artifacts",
        key="overlay.bin.gz",
        payload=payload,
        metadata=metadata,
        content_type="application/octet-stream",
    )


def test_s3_put_rejects_concurrent_different_content():
    s3 = _S3()
    s3.fail_put = True
    s3.head = {
        "ContentLength": 7,
        "Metadata": {"sha256": "f" * 64, "cache-identity": "b" * 64},
    }
    with pytest.raises(RuntimeError, match="different identity"):
        _put_s3_immutable(
            s3,
            bucket="artifacts",
            key="overlay.bin.gz",
            payload=b"overlay",
            metadata={
                "sha256": "a" * 64,
                "cache-identity": "b" * 64,
            },
            content_type="application/octet-stream",
        )


def test_dynamo_retry_preserves_existing_ready_item():
    item = {
        "pk": "OVLSET#model#l2d#v2.1",
        "sk": "META",
        "status": "building",
        "request_identity": "a" * 64,
    }
    table = _Table()
    written = _put_dynamo_immutable(
        table, item, identity_fields=("pk", "sk", "request_identity")
    )
    assert written["status"] == "building"
    assert "attribute_not_exists" in table.put_calls[0]["ConditionExpression"]

    table.fail_put = True
    table.item = {**item, "status": "ready"}
    existing = _put_dynamo_immutable(
        table, item, identity_fields=("pk", "sk", "request_identity")
    )
    assert existing["status"] == "ready"

    table.item["request_identity"] = "b" * 64
    with pytest.raises(RuntimeError, match="different identity"):
        _put_dynamo_immutable(
            table, item, identity_fields=("pk", "sk", "request_identity")
        )


def _ready_item() -> dict:
    return {
        "pk": "OVLSET#model#l2d#v2.1",
        "sk": "META",
        "status": "ready",
        "request_identity": "a" * 64,
        "cache_identity": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "artifacts_bucket": "artifacts",
        "overlay_schema": "v1",
        "seeds": [0],
        "n_shards": 2,
        "n_samples": 20,
        "manifest_key": "manifest.json",
        "created_at": "2026-07-15T00:00:00Z",
    }


def test_ready_publication_only_transitions_from_compatible_building():
    item = _ready_item()
    table = _Table()
    _publish_overlay_set_ready(table, item)
    request = table.put_calls[0]
    assert "#status = :building" in request["ConditionExpression"]

    table.fail_put = True
    table.item = dict(item)
    _publish_overlay_set_ready(table, item)

    table.item["n_samples"] = 21
    with pytest.raises(RuntimeError, match="immutable"):
        _publish_overlay_set_ready(table, item)


def test_gate_token_keeps_the_winning_creation_time_and_ready_state():
    item = {
        **_ready_item(),
        "dataset_manifest_sha256": "c" * 64,
    }
    token = _gate_token(item, "d" * 64)
    gate = _parse_gate(token)
    assert gate["status"] == "ready"
    assert gate["created_at"] == "2026-07-15T00:00:00Z"
    assert gate["request_identity"] == "a" * 64


class _MLflowClient:
    def __init__(self, versions, runs):
        self.versions = versions
        self.runs = runs

    def search_model_versions(self, query):
        assert query == "name='auto-e2e-driving-policy'"
        return self.versions

    def get_run(self, run_id):
        return self.runs[run_id]


def _model_version(version, run_id, digest):
    return SimpleNamespace(
        version=str(version),
        run_id=run_id,
        tags={"checkpoint_sha256": digest} if digest else {},
    )


def _run(execution_id, *, dataset_version="v2.1"):
    return SimpleNamespace(
        data=SimpleNamespace(
            params={
                "ctx/train_execution_id": execution_id,
                "data/dataset": "KIT-MRT/KITScenes-Multimodal",
                "data/dataset_version": dataset_version,
            },
            tags={},
        )
    )


def _resolve(client):
    return _resolve_model_version_for_execution(
        client,
        registered_model_name="auto-e2e-driving-policy",
        train_execution_id="a1234567890123456789",
        expected_dataset="KIT-MRT/KITScenes-Multimodal",
        expected_dataset_version="v2.1",
    )


def test_full_run_model_resolution_uses_exact_execution_lineage():
    client = _MLflowClient(
        [
            _model_version(40, "other", "b" * 64),
            _model_version(41, "target", "a" * 64),
        ],
        {
            "other": _run("a0000000000000000000"),
            "target": _run("a1234567890123456789"),
        },
    )

    assert _resolve(client) == "41"


def test_full_run_model_resolution_dedupes_identical_re_evaluation():
    client = _MLflowClient(
        [
            _model_version(41, "first", "a" * 64),
            _model_version(43, "retry", "a" * 64),
        ],
        {
            "first": _run("a1234567890123456789"),
            "retry": _run("a1234567890123456789"),
        },
    )

    assert _resolve(client) == "43"


def test_full_run_model_resolution_rejects_ambiguous_checkpoints():
    client = _MLflowClient(
        [
            _model_version(41, "first", "a" * 64),
            _model_version(43, "retry", "b" * 64),
        ],
        {
            "first": _run("a1234567890123456789"),
            "retry": _run("a1234567890123456789"),
        },
    )

    with pytest.raises(ValueError, match="checkpoint identity is ambiguous"):
        _resolve(client)


def test_full_run_model_resolution_checks_dataset_version():
    client = _MLflowClient(
        [_model_version(41, "target", "a" * 64)],
        {
            "target": _run(
                "a1234567890123456789",
                dataset_version="v2.0",
            )
        },
    )

    with pytest.raises(ValueError, match="different dataset coordinate"):
        _resolve(client)
