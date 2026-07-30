"""Resumable imitation-learning checkpoint contract tests."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from botocore.exceptions import ClientError

from evaluation.checkpoint_selection import SELECTOR_POLICY_VERSION
from Platform.pipelines.training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    best_pointer_key,
    capture_rng_state,
    checkpoint_key,
    metric_pair_is_better,
    restore_rng_state,
    stable_digest,
    update_best_pointer,
    upload_immutable_checkpoint,
    validate_immutable_checkpoint_record,
    validate_resume_envelope,
    validate_resume_payload,
)


CANONICAL_METRIC_CONTRACT = {
    "version": "rollout_validation_v2",
    "horizon_seconds": 3.0,
    "horizon_steps": 30,
    "target_source": "logged_xy",
    "aggregation": "scene_balanced",
}


class _FakeS3:
    def __init__(self, conditional_conflicts=0):
        self.objects = {}
        self.versions = []
        self.conditional_conflicts = conditional_conflicts

    def put_object(self, **kwargs):
        identity = (kwargs["Bucket"], kwargs["Key"])
        if (
            kwargs.get("IfNoneMatch") == "*"
            and self.conditional_conflicts > 0
        ):
            self.conditional_conflicts -= 1
            raise ClientError(
                {
                    "Error": {
                        "Code": "ConditionalRequestConflict",
                        "Message": "conditional write is still in flight",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 409},
                },
                "PutObject",
            )
        if kwargs.get("IfNoneMatch") == "*" and identity in self.objects:
            raise ClientError(
                {
                    "Error": {
                        "Code": "PreconditionFailed",
                        "Message": "object already exists",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )
        body = kwargs["Body"]
        payload = body.read() if hasattr(body, "read") else bytes(body)
        record = {
            "Body": payload,
            "ContentLength": len(payload),
            "ContentType": kwargs.get("ContentType"),
            "Metadata": kwargs.get("Metadata", {}),
        }
        self.objects[identity] = record
        self.versions.append((identity, record))
        return {}

    def head_object(self, *, Bucket, Key):
        record = self.objects[(Bucket, Key)]
        return {
            "ContentLength": record["ContentLength"],
            "Metadata": record["Metadata"],
        }


def _resume_payload():
    config = {"backbone": "swin_v2_tiny", "enable_world_model": True}
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "scaler_state_dict": {},
        "rng_state": {
            "python": (),
            "numpy": (),
            "torch_cpu": b"",
            "torch_cuda": [],
        },
        "epoch": 3,
        "config": config,
        "training_state": {"run_id": "run-1"},
        "data_fingerprint": "data-1",
    }, config


def test_checkpoint_key_is_epoch_immutable():
    assert checkpoint_key("run-1", 1) == (
        "imitation-learning/run-1/epoch-0001.pt"
    )
    assert checkpoint_key("run-1", 42).endswith("epoch-0042.pt")

    with pytest.raises(ValueError, match="positive"):
        checkpoint_key("run-1", 0)
    with pytest.raises(ValueError, match="run id"):
        checkpoint_key("bad/run", 1)


def test_best_pointer_keys_are_independent():
    assert best_pointer_key("run-1") == (
        "imitation-learning/run-1/best.json"
    )
    assert best_pointer_key("run-1", role="best_trajectory") == (
        "imitation-learning/run-1/best-trajectory.json"
    )
    with pytest.raises(ValueError, match="role"):
        best_pointer_key("run-1", role="final")


def test_metric_ranking_uses_ade_then_fde():
    assert metric_pair_is_better(1.9, 8.0, 2.0, 1.0)
    assert metric_pair_is_better(2.0, 0.9, 2.0, 1.0)
    assert not metric_pair_is_better(2.0, 1.1, 2.0, 1.0)
    assert not metric_pair_is_better(2.1, 0.1, 2.0, 1.0)


def test_stable_digest_is_order_independent():
    assert stable_digest({"b": 2, "a": [1]}) == stable_digest(
        {"a": [1], "b": 2}
    )


def test_resume_validation_rejects_schema_config_and_data_drift():
    payload, config = _resume_payload()
    validate_resume_payload(
        payload,
        expected_config=config,
        expected_data_fingerprint="data-1",
    )

    bad_schema = dict(payload, schema_version="legacy")
    with pytest.raises(ValueError, match="schema"):
        validate_resume_payload(
            bad_schema,
            expected_config=config,
            expected_data_fingerprint="data-1",
        )

    with pytest.raises(ValueError, match="config"):
        validate_resume_payload(
            payload,
            expected_config={"backbone": "resnet_50"},
            expected_data_fingerprint="data-1",
        )

    with pytest.raises(ValueError, match="fingerprint"):
        validate_resume_payload(
            payload,
            expected_config=config,
            expected_data_fingerprint="different",
        )


def test_resume_envelope_rejects_malformed_transition_input():
    payload, _ = _resume_payload()

    validate_resume_envelope(payload)

    with pytest.raises(ValueError, match="schema"):
        validate_resume_envelope(
            dict(payload, schema_version="legacy")
        )
    malformed = dict(payload)
    malformed.pop("config")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_resume_envelope(malformed)


def test_resume_validation_allows_explicit_policy_transition_only():
    payload, config = _resume_payload()
    payload["config"] = {
        **config,
        "junction_sampling": {"enabled": False, "policy": None},
        "early_stopping_patience": 3,
    }
    requested = {
        **config,
        "junction_sampling": {
            "enabled": True,
            "policy": {"version": "navigation_repeat_v1"},
        },
        "early_stopping_patience": 8,
    }

    validate_resume_payload(
        payload,
        expected_config=requested,
        expected_data_fingerprint="data-2",
        allowed_config_changes=frozenset({
            "junction_sampling",
            "early_stopping_patience",
        }),
        compatible_data_fingerprints=frozenset({"data-1"}),
    )

    with pytest.raises(ValueError, match="config"):
        validate_resume_payload(
            payload,
            expected_config={**requested, "backbone": "resnet_50"},
            expected_data_fingerprint="data-2",
            allowed_config_changes=frozenset({
                "junction_sampling",
                "early_stopping_patience",
            }),
            compatible_data_fingerprints=frozenset({"data-1"}),
        )


def test_resume_rejects_selector_v2_history_under_v3_policy():
    payload, config = _resume_payload()
    payload["config"] = {
        **config,
        "checkpoint_selection": {
            "policy_version": "rollout_composite_selector_v2",
        },
    }
    requested = {
        **config,
        "checkpoint_selection": {
            "policy_version": SELECTOR_POLICY_VERSION,
        },
    }

    with pytest.raises(ValueError, match="config"):
        validate_resume_payload(
            payload,
            expected_config=requested,
            expected_data_fingerprint="data-1",
        )


def test_rng_state_round_trip():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    random.random()
    np.random.random()
    torch.rand(3)

    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_immutable_upload_accepts_identical_retry(tmp_path):
    s3 = _FakeS3()
    checkpoint = tmp_path / "epoch-0001.pt"
    checkpoint.write_bytes(b"checkpoint-v1")

    first = upload_immutable_checkpoint(
        s3,
        bucket="checkpoints",
        key="run/epoch-0001.pt",
        path=checkpoint,
    )
    retry = upload_immutable_checkpoint(
        s3,
        bucket="checkpoints",
        key="run/epoch-0001.pt",
        path=checkpoint,
    )

    assert first["created"] is True
    assert retry["created"] is False
    assert retry["sha256"] == first["sha256"]
    assert len(s3.objects) == 1


def test_immutable_upload_rejects_different_retry(tmp_path):
    s3 = _FakeS3()
    checkpoint = tmp_path / "epoch-0001.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    upload_immutable_checkpoint(
        s3,
        bucket="checkpoints",
        key="run/epoch-0001.pt",
        path=checkpoint,
    )
    checkpoint.write_bytes(b"checkpoint-v2")

    with pytest.raises(RuntimeError, match="immutable checkpoint conflict"):
        upload_immutable_checkpoint(
            s3,
            bucket="checkpoints",
            key="run/epoch-0001.pt",
            path=checkpoint,
        )


def test_immutable_upload_retries_conditional_conflict(tmp_path):
    s3 = _FakeS3(conditional_conflicts=2)
    checkpoint = tmp_path / "epoch-0001.pt"
    checkpoint.write_bytes(b"checkpoint-v1")

    uploaded = upload_immutable_checkpoint(
        s3,
        bucket="checkpoints",
        key="run/epoch-0001.pt",
        path=checkpoint,
    )

    assert uploaded["created"] is True
    assert s3.conditional_conflicts == 0
    assert len(s3.objects) == 1


def test_best_pointer_is_versioned_json():
    s3 = _FakeS3()
    uri = update_best_pointer(
        s3,
        bucket="checkpoints",
        run_id="run-1",
        epoch=2,
        checkpoint_uri="s3://checkpoints/run/epoch-0002.pt",
        checkpoint_sha256="a" * 64,
        ade=1.25,
        fde=2.5,
    )
    update_best_pointer(
        s3,
        bucket="checkpoints",
        run_id="run-1",
        epoch=3,
        checkpoint_uri="s3://checkpoints/run/epoch-0003.pt",
        checkpoint_sha256="b" * 64,
        ade=1.0,
        fde=2.0,
    )

    assert uri == "s3://checkpoints/imitation-learning/run-1/best.json"
    assert len(s3.versions) == 2
    latest = json.loads(s3.versions[-1][1]["Body"])
    assert latest["checkpoint_role"] == "best"
    assert latest["epoch"] == 3
    assert latest["checkpoint_sha256"] == "b" * 64


def test_trajectory_best_pointer_is_separate():
    s3 = _FakeS3()

    uri = update_best_pointer(
        s3,
        bucket="checkpoints",
        run_id="run-1",
        role="best_trajectory",
        epoch=2,
        checkpoint_uri="s3://checkpoints/run/epoch-0002.pt",
        checkpoint_sha256="a" * 64,
        ade=1.25,
        fde=2.5,
    )

    assert uri == (
        "s3://checkpoints/imitation-learning/run-1/"
        "best-trajectory.json"
    )
    payload = json.loads(s3.versions[-1][1]["Body"])
    assert payload["checkpoint_role"] == "best_trajectory"
    assert (
        "checkpoints",
        "imitation-learning/run-1/best.json",
    ) not in s3.objects


def test_best_pointer_rejects_invalid_checkpoint_digest():
    with pytest.raises(ValueError, match="SHA-256"):
        update_best_pointer(
            _FakeS3(),
            bucket="checkpoints",
            run_id="run-1",
            epoch=2,
            checkpoint_uri="s3://checkpoints/run/epoch-0002.pt",
            checkpoint_sha256="None",
            ade=1.25,
            fde=2.5,
        )


def test_checkpoint_record_is_verified_against_s3_metadata(tmp_path):
    s3 = _FakeS3()
    checkpoint = tmp_path / "epoch-0002.pt"
    checkpoint.write_bytes(b"checkpoint-v2")
    uploaded = upload_immutable_checkpoint(
        s3,
        bucket="checkpoints",
        key="run/epoch-0002.pt",
        path=checkpoint,
    )
    record = {
        "uri": uploaded["uri"],
        "sha256": uploaded["sha256"],
        "size": uploaded["size"],
    }

    validate_immutable_checkpoint_record(s3, record)

    with pytest.raises(ValueError, match="metadata"):
        validate_immutable_checkpoint_record(
            s3,
            {**record, "sha256": "a" * 64},
        )
    with pytest.raises(ValueError, match="size"):
        validate_immutable_checkpoint_record(
            s3,
            {**record, "size": uploaded["size"] + 1},
        )


def test_best_pointer_records_composite_selection_policy():
    s3 = _FakeS3()
    selection = {
        "policy_version": SELECTOR_POLICY_VERSION,
        "score": 0.75,
        "components": {
            "trajectory": 0.5,
            "comfort": 1.0,
        },
    }

    update_best_pointer(
        s3,
        bucket="checkpoints",
        run_id="run-1",
        epoch=4,
        checkpoint_uri="s3://checkpoints/run/epoch-0004.pt",
        checkpoint_sha256="c" * 64,
        ade=1.0,
        fde=2.0,
        selection=selection,
        metric_contract=CANONICAL_METRIC_CONTRACT,
    )

    latest = json.loads(s3.versions[-1][1]["Body"])
    assert latest["schema_version"] == "best_checkpoint_pointer_v3"
    assert latest["selection"] == selection
    assert latest["metric_contract"] == CANONICAL_METRIC_CONTRACT
