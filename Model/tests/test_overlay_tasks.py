"""Concurrency and write-once tests for overlay publication tasks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

pytest.importorskip("flytekit")

from Platform.pipelines.overlay_tasks import (
    OVERLAY_TASK_ENV,
    _gate_token,
    _metric_at_epoch,
    _parse_gate,
    _publish_overlay_set_ready,
    _put_dynamo_immutable,
    _put_s3_immutable,
    _register_selected_checkpoint_version,
    _resolve_model_version_for_execution,
    _validate_empty_overlay_partition,
    _validate_selected_checkpoint_payload,
)


def test_overlay_tasks_configure_deterministic_cublas_workspace():
    assert OVERLAY_TASK_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_empty_overlay_partition_requires_an_explicit_empty_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "total_samples": 0,
        "shards": 0,
        "shard_names": [],
    }))

    _validate_empty_overlay_partition(str(tmp_path))


@pytest.mark.parametrize(
    "manifest",
    [
        {"total_samples": 1, "shards": 0, "shard_names": []},
        {"total_samples": 0, "shards": 1, "shard_names": []},
        {
            "total_samples": 0,
            "shards": 0,
            "shard_names": ["missing.tar"],
        },
    ],
)
def test_empty_overlay_partition_rejects_missing_advertised_shards(
    tmp_path,
    manifest,
):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(FileNotFoundError, match="advertises samples"):
        _validate_empty_overlay_partition(str(tmp_path))


def test_empty_overlay_partition_requires_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="no tar shards or manifest"):
        _validate_empty_overlay_partition(str(tmp_path))


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


def _model_version(version, run_id, digest, **tags):
    return SimpleNamespace(
        version=str(version),
        run_id=run_id,
        source=tags.pop("source", ""),
        tags={
            **({"checkpoint_sha256": digest} if digest else {}),
            **tags,
        },
    )


def _run(execution_id="", *, dataset_version="v2.1"):
    params = {
        "data/dataset": "KIT-MRT/KITScenes-Multimodal",
        "data/dataset_version": dataset_version,
    }
    if execution_id:
        params["ctx/train_execution_id"] = execution_id
    return SimpleNamespace(
        data=SimpleNamespace(
            params=params,
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


def test_full_run_model_resolution_prefers_version_lineage_tags():
    client = _MLflowClient(
        [
            _model_version(
                44,
                "target",
                "a" * 64,
                train_execution_id="a1234567890123456789",
                dataset="KIT-MRT/KITScenes-Multimodal",
                dataset_version="v2.1",
            ),
        ],
        {"target": _run()},
    )

    assert _resolve(client) == "44"


def test_full_run_model_resolution_ignores_legacy_when_tags_match():
    client = _MLflowClient(
        [
            _model_version(41, "legacy", "b" * 64),
            _model_version(
                44,
                "selected",
                "a" * 64,
                train_execution_id="a1234567890123456789",
                dataset="KIT-MRT/KITScenes-Multimodal",
                dataset_version="v2.1",
                checkpoint_role="selected-overlay",
            ),
        ],
        {
            "legacy": _run("a1234567890123456789"),
            "selected": _run(),
        },
    )

    assert _resolve(client) == "44"


def test_full_run_model_resolution_prefers_selected_tagged_checkpoint():
    tags = {
        "train_execution_id": "a1234567890123456789",
        "dataset": "KIT-MRT/KITScenes-Multimodal",
        "dataset_version": "v2.1",
    }
    client = _MLflowClient(
        [
            _model_version(
                44,
                "best",
                "b" * 64,
                checkpoint_role="best",
                **tags,
            ),
            _model_version(
                43,
                "selected",
                "a" * 64,
                checkpoint_role="selected-overlay",
                **tags,
            ),
        ],
        {
            "best": _run(),
            "selected": _run(),
        },
    )

    assert _resolve(client) == "43"


def test_full_run_model_resolution_rejects_partial_lineage_tags():
    client = _MLflowClient(
        [
            _model_version(
                44,
                "partial",
                "a" * 64,
                train_execution_id="a1234567890123456789",
            ),
        ],
        {"partial": _run()},
    )

    with pytest.raises(ValueError, match="incomplete lineage tags"):
        _resolve(client)


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


class _SelectedCheckpointClient:
    def __init__(self, versions=()):
        self.versions = list(versions)
        self.created_versions = []
        self.tags = {}

    def get_registered_model(self, name):
        assert name == "auto-e2e-driving-policy"
        return SimpleNamespace(name=name)

    def search_model_versions(self, query):
        assert query == "name='auto-e2e-driving-policy'"
        return self.versions

    def create_model_version(self, *, name, source, run_id):
        version = SimpleNamespace(
            version="44",
            source=source,
            run_id=run_id,
            tags={},
        )
        self.created_versions.append(version)
        self.versions.append(version)
        return version

    def set_model_version_tag(self, name, version, key, value):
        assert name == "auto-e2e-driving-policy"
        assert version == "44"
        self.tags[key] = value


def _register_selected(client):
    return _register_selected_checkpoint_version(
        client,
        registered_model_name="auto-e2e-driving-policy",
        run_id="1" * 32,
        checkpoint_uri=(
            "s3://checkpoints/imitation-learning/"
            f"{'1' * 32}/epoch-0004.pt"
        ),
        checkpoint_sha256="a" * 64,
        checkpoint_epoch=4,
        train_execution_id="a1234567890123456789",
        dataset="KIT-MRT/KITScenes-Multimodal",
        dataset_version="v2.2",
        checkpoint_schema="il_checkpoint_v2",
        data_fingerprint="f" * 64,
        validation_ade=9.689568680297887,
        validation_fde=29.656355911506907,
    )


def test_selected_checkpoint_registration_records_exact_provenance():
    client = _SelectedCheckpointClient()

    assert _register_selected(client) == "44"
    assert len(client.created_versions) == 1
    assert client.tags == {
        "checkpoint_epoch": "4",
        "checkpoint_s3_uri": (
            "s3://checkpoints/imitation-learning/"
            f"{'1' * 32}/epoch-0004.pt"
        ),
        "checkpoint_sha256": "a" * 64,
        "train_execution_id": "a1234567890123456789",
        "dataset": "KIT-MRT/KITScenes-Multimodal",
        "dataset_version": "v2.2",
        "checkpoint_schema": "il_checkpoint_v2",
        "data_fingerprint": "f" * 64,
        "checkpoint_role": "selected-overlay",
        "validation_ade": "9.689568680297887",
        "validation_fde": "29.656355911506907",
    }


def test_selected_checkpoint_registration_reuses_and_preserves_roles():
    source = (
        "s3://checkpoints/imitation-learning/"
        f"{'1' * 32}/epoch-0004.pt"
    )
    version = _model_version(
        44,
        "1" * 32,
        "a" * 64,
        source=source,
        checkpoint_epoch="4",
        checkpoint_s3_uri=source,
        train_execution_id="a1234567890123456789",
        dataset="KIT-MRT/KITScenes-Multimodal",
        dataset_version="v2.2",
        checkpoint_role="best",
    )
    client = _SelectedCheckpointClient([version])

    assert _register_selected(client) == "44"
    assert client.created_versions == []
    assert client.tags["checkpoint_role"] == "best,selected-overlay"


def test_selected_checkpoint_registration_rejects_source_reassignment():
    source = (
        "s3://checkpoints/imitation-learning/"
        f"{'1' * 32}/epoch-0004.pt"
    )
    client = _SelectedCheckpointClient([
        _model_version(
            44,
            "2" * 32,
            "a" * 64,
            source=source,
        )
    ])

    with pytest.raises(RuntimeError, match="different MLflow run"):
        _register_selected(client)


def test_selected_checkpoint_registration_rejects_a_second_selected_epoch():
    source = (
        "s3://checkpoints/imitation-learning/"
        f"{'1' * 32}/epoch-0003.pt"
    )
    client = _SelectedCheckpointClient([
        _model_version(
            43,
            "1" * 32,
            "b" * 64,
            source=source,
            checkpoint_role="selected-overlay",
            train_execution_id="a1234567890123456789",
        )
    ])

    with pytest.raises(RuntimeError, match="different selected-overlay"):
        _register_selected(client)


def test_metric_at_epoch_uses_latest_finite_retry():
    client = SimpleNamespace(
        get_metric_history=lambda run_id, key: [
            SimpleNamespace(step=3, value=1.0, timestamp=1),
            SimpleNamespace(step=4, value=9.7, timestamp=2),
            SimpleNamespace(step=4, value=9.6, timestamp=3),
        ]
    )

    assert _metric_at_epoch(
        client,
        run_id="1" * 32,
        metric_key="val/ade",
        epoch=4,
    ) == 9.6


def test_metric_at_epoch_rejects_missing_epoch():
    client = SimpleNamespace(get_metric_history=lambda run_id, key: [])

    with pytest.raises(ValueError, match="has no 'val/ade' at epoch 4"):
        _metric_at_epoch(
            client,
            run_id="1" * 32,
            metric_key="val/ade",
            epoch=4,
        )


def _selected_checkpoint_payload():
    return {
        "schema_version": "il_checkpoint_v2",
        "epoch": 4,
        "data_fingerprint": "f" * 64,
        "training_state": {
            "run_id": "1" * 32,
            "current_checkpoint_uri": (
                "s3://checkpoints/imitation-learning/"
                f"{'1' * 32}/epoch-0004.pt"
            ),
            "metric_history": [
                {
                    "epoch": epoch,
                    "val_ade": 9.689568680297887
                    if epoch == 4 else 20.0 - epoch,
                    "val_fde": 29.656355911506907
                    if epoch == 4 else 40.0 - epoch,
                }
                for epoch in range(1, 5)
            ],
        },
    }


def _validate_payload(payload):
    return _validate_selected_checkpoint_payload(
        payload,
        checkpoint_schema="il_checkpoint_v2",
        checkpoint_epoch=4,
        checkpoint_uri=(
            "s3://checkpoints/imitation-learning/"
            f"{'1' * 32}/epoch-0004.pt"
        ),
        run_id="1" * 32,
        data_fingerprint="f" * 64,
        validation_ade=9.689568680297887,
        validation_fde=29.656355911506907,
    )


def test_selected_checkpoint_payload_accepts_exact_epoch_provenance():
    _validate_payload(_selected_checkpoint_payload())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(schema_version="legacy"),
            "unsupported schema",
        ),
        (
            lambda payload: payload.update(epoch=3),
            "epoch differs",
        ),
        (
            lambda payload: payload.update(data_fingerprint="e" * 64),
            "fingerprint differs",
        ),
        (
            lambda payload: payload["training_state"].update(
                run_id="2" * 32
            ),
            "different MLflow run",
        ),
        (
            lambda payload: payload["training_state"].update(
                current_checkpoint_uri="s3://checkpoints/other.pt"
            ),
            "self URI differs",
        ),
        (
            lambda payload: payload["training_state"][
                "metric_history"
            ].pop(1),
            "not contiguous",
        ),
        (
            lambda payload: payload["training_state"][
                "metric_history"
            ][-1].update(val_ade=10.0),
            "val_ade differs",
        ),
        (
            lambda payload: payload["training_state"][
                "metric_history"
            ][-1].update(val_fde=30.0),
            "val_fde differs",
        ),
    ],
)
def test_selected_checkpoint_payload_rejects_provenance_drift(
    mutate,
    message,
):
    payload = _selected_checkpoint_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        _validate_payload(payload)
