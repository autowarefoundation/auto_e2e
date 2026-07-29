from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from Platform.pipelines.odd_metric_projection_tasks import (
    _decode_shard_index,
    _deterministic_jsonl_gzip,
    _event_values,
    _metric_samples_from_shard,
    _projection_root,
    _validation_contract,
)
from Platform.pipelines.overlay import (
    BEV_HEATMAP_NAMES,
    BEV_HEATMAP_SIZE,
    encode_overlay,
)
from Platform.pipelines.training_checkpoint import stable_digest


def _overlay(sample_uids: list[str], speeds: list[float]) -> bytes:
    controls = np.zeros(
        (len(sample_uids), 1, 64, 2),
        dtype=np.float32,
    )
    heatmaps = np.zeros(
        (
            len(sample_uids),
            len(BEV_HEATMAP_NAMES),
            BEV_HEATMAP_SIZE,
            BEV_HEATMAP_SIZE,
        ),
        dtype=np.float32,
    )
    return encode_overlay(
        sample_uids,
        controls,
        np.asarray(speeds, dtype=np.float32),
        bev_heatmaps=heatmaps,
    )


def _index_sample(
    sample_uid: str,
    group_uid: str,
    *,
    speed: float,
    timestamp_ns: int,
) -> dict:
    return {
        "sample_uid": sample_uid,
        "split_group_uid": group_uid,
        "ego_now": [speed, 0.0, 0.0, 0.0],
        "ego_future": [0.0] * (64 * 2),
        "pose_current": {"timestamp_ns": str(timestamp_ns)},
    }


def _training_metadata() -> dict:
    groups = ["group-a", "group-b"]
    group_digest = stable_digest(groups)
    return {
        "training": {
            "validation_split": {
                "strategy": "exact_group_fraction",
                "split_id": "split-v1",
                "validation_group_uids": groups,
                "validation_group_uid_digest": group_digest,
            },
        },
        "validation": {
            "sample_count": 3,
            "sample_uid_digest": "a" * 64,
        },
    }


def test_metric_samples_require_exact_overlay_index_alignment():
    payload = _overlay(["sample-a", "sample-b"], [5.0, 7.0])
    index = {
        "samples": [
            _index_sample(
                "sample-a",
                "validation",
                speed=5.0,
                timestamp_ns=100,
            ),
            _index_sample(
                "sample-b",
                "training",
                speed=7.0,
                timestamp_ns=200,
            ),
        ],
    }

    samples = _metric_samples_from_shard(
        overlay_payload=payload,
        index=index,
        scene_uid="scene-1",
        validation_group_uids=frozenset({"validation"}),
    )

    assert len(samples) == 1
    assert samples[0].sample_uid == "sample-a"
    assert samples[0].scene_uid == "scene-1"
    assert samples[0].sample_anchor_timestamp_ns == 100
    assert samples[0].predicted_controls.shape == (1, 64, 2)
    assert samples[0].ground_truth_controls.shape == (64, 2)

    index["samples"][0]["ego_now"][0] = 5.1
    with pytest.raises(ValueError, match="initial speed differs"):
        _metric_samples_from_shard(
            overlay_payload=payload,
            index=index,
            scene_uid="scene-1",
            validation_group_uids=frozenset({"validation"}),
        )


def test_metric_samples_reject_index_uid_set_drift():
    payload = _overlay(["sample-a"], [5.0])
    index = {
        "samples": [
            _index_sample(
                "sample-other",
                "validation",
                speed=5.0,
                timestamp_ns=100,
            ),
        ],
    }

    with pytest.raises(ValueError, match="has no row"):
        _metric_samples_from_shard(
            overlay_payload=payload,
            index=index,
            scene_uid="scene-1",
            validation_group_uids=frozenset({"validation"}),
        )


def test_validation_contract_requires_frozen_group_identity():
    groups, count, digest, strategy, split_id = _validation_contract(
        _training_metadata()
    )

    assert groups == frozenset({"group-a", "group-b"})
    assert count == 3
    assert digest == "a" * 64
    assert strategy == "exact_group_fraction"
    assert split_id == "split-v1"

    metadata = _training_metadata()
    metadata["training"]["validation_split"][
        "validation_group_uid_digest"
    ] = "b" * 64
    with pytest.raises(ValueError, match="group digest differs"):
        _validation_contract(metadata)


def test_event_values_are_recovered_from_primary_observations():
    observations = [
        {
            "observation_uid": "obs-primary",
            "label_key": "event.ego.maneuver",
            "values": ["turn_left"],
        },
        {
            "observation_uid": "obs-secondary",
            "label_key": "event.outcome",
            "values": ["normal_completion"],
        },
    ]
    events = [
        {
            "event_uid": "event-1",
            "primary_event_key": "event.ego.maneuver",
            "observation_uids": ["obs-secondary", "obs-primary"],
        },
    ]

    enriched = _event_values(events, observations)

    assert enriched[0]["primary_values"] == ["turn_left"]
    assert "primary_values" not in events[0]


def test_cached_index_decode_and_projection_artifacts_are_deterministic():
    index = {"shard": "shard.tar", "samples": []}
    item = {
        "payload": {
            "B": gzip.compress(
                json.dumps(index, sort_keys=True).encode("ascii"),
                mtime=0,
            ),
        },
    }

    assert _decode_shard_index(item, "shard.tar") == index
    rows = [{"sample_uid": "sample-a", "metrics": {"ade": 1.0}}]
    first = _deterministic_jsonl_gzip(rows)
    second = _deterministic_jsonl_gzip(rows)
    assert first == second
    assert gzip.decompress(first).endswith(b"\n")

    root = _projection_root(
        odd_dataset="kitscenes",
        odd_version="v3.0",
        labelset_id="oddls-1",
        model_artifact_id="1" * 64,
        projection_identity="2" * 64,
    )
    assert root == (
        "odd_metric_projections/schema=v1/"
        "dataset=kitscenes/version=v3.0/labelset=oddls-1/"
        f"model={'1' * 64}/projection={'2' * 64}"
    )
