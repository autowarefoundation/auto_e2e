"""DynamoDB overlay key/item contract tests."""

from decimal import Decimal

import pytest

from Platform.pipelines.overlay_store import (
    model_profile_item,
    model_version_item,
    overlay_pointer_item,
    overlay_set_item,
    shard_model_pk,
)


def _metadata():
    return {
        "registered_model_name": "auto-e2e-driving-policy",
        "model_version": "30",
        "run_id": "run-123",
        "artifact_uri": "s3://artifacts/mlflow/model",
        "model_name": "swin_v2_tiny",
        "eval_ade": 1.25,
        "eval_fde": 2.5,
        "eval_gate_pass": 1,
        "dataset": "l2d",
        "dataset_version": "v2.1",
        "train_execution_id": "flyte-123",
        "val_fraction": 0.1,
    }


def test_pointer_item_contains_no_overlay_body_or_split_dimension():
    item = overlay_pointer_item(
        dataset="l2d",
        version="v2.1",
        shard="train-000001.tar",
        model_artifact_id="a" * 64,
        s3_key="overlays/body",
        sha256="b" * 64,
        byte_size=1234,
        sample_count=1000,
        overlay_schema="v1",
        created_at="2026-07-14T00:00:00Z",
        model_metadata=_metadata(),
    )
    assert item["pk"] == "SHARD#l2d#v2.1#train-000001.tar"
    assert item["sk"] == "MODEL#" + "a" * 64
    assert item["eval_ade"] == Decimal("1.25")
    assert "payload" not in item and "body" not in item
    assert not any("split" in key for key in item)


def test_model_registry_coordinate_resolves_content_identity():
    profile = model_profile_item(
        "a" * 64, _metadata(), created_at="2026-07-14T00:00:00Z"
    )
    version = model_version_item("a" * 64, _metadata())
    assert profile["pk"] == "MODEL#" + "a" * 64
    assert version["pk"] == "MODELVER#auto-e2e-driving-policy#30"
    assert version["checkpoint_sha256"] == "a" * 64


def test_overlay_set_status_is_explicit():
    item = overlay_set_item(
        "a" * 64,
        "l2d",
        "v2.1",
        status="ready",
        seeds=[0, 1],
        overlay_schema="v1",
        created_at="2026-07-14T00:00:00Z",
        n_shards=2,
        n_samples=2000,
        manifest_key="manifest.json",
    )
    assert item["pk"].startswith("OVLSET#")
    assert item["status"] == "ready"
    with pytest.raises(ValueError, match="status"):
        overlay_set_item(
            "a" * 64,
            "l2d",
            "v2.1",
            status="partial",
            seeds=[0],
            overlay_schema="v1",
            created_at="now",
        )


def test_shard_key_has_single_version_coordinate():
    assert shard_model_pk("l2d", "v2.1", "train.tar") == (
        "SHARD#l2d#v2.1#train.tar"
    )
