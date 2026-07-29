from __future__ import annotations

import numpy as np
import pytest

from Platform.pipelines.odd_metric_projection import (
    METRIC_POLICY_VERSION,
    PROJECTION_POLICY_VERSION,
    MetricSample,
    compute_sample_metrics,
    project_metric_samples,
    projection_cache_identity,
    sample_uid_digest,
)


def _controls(
    *,
    acceleration: float = 0.0,
    curvature: float = 0.0,
) -> np.ndarray:
    controls = np.zeros((64, 2), dtype=np.float32)
    controls[:, 0] = acceleration
    controls[:, 1] = curvature
    return controls


def _sample(
    sample_uid: str = "sample-1",
    *,
    anchor: int = 10_000_000_000,
    predicted: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
) -> MetricSample:
    return MetricSample(
        sample_uid=sample_uid,
        scene_uid="scene-1",
        split_group_uid="group-1",
        sample_anchor_timestamp_ns=anchor,
        predicted_controls=(
            predicted
            if predicted is not None
            else _controls()[None, :, :]
        ),
        ground_truth_controls=(
            ground_truth if ground_truth is not None else _controls()
        ),
        initial_speed_mps=5.0,
    )


def test_compute_sample_metrics_is_zero_for_identical_controls():
    controls = _controls(acceleration=0.2, curvature=0.01)
    metrics = compute_sample_metrics(
        controls[None, :, :],
        controls,
        initial_speed_mps=8.0,
    )

    assert set(metrics) == {
        "ade_1s_m",
        "ade_2s_m",
        "ade_3s_m",
        "ade_horizon_m",
        "fde_horizon_m",
        "acceleration_mae",
        "curvature_mae",
    }
    assert all(value == pytest.approx(0.0) for value in metrics.values())


def test_compute_sample_metrics_averages_model_seeds():
    ground_truth = _controls()
    exact = ground_truth.copy()
    accelerating = _controls(acceleration=1.0)

    exact_metrics = compute_sample_metrics(
        exact[None, :, :],
        ground_truth,
        initial_speed_mps=2.0,
    )
    accelerating_metrics = compute_sample_metrics(
        accelerating[None, :, :],
        ground_truth,
        initial_speed_mps=2.0,
    )
    combined_metrics = compute_sample_metrics(
        np.stack([exact, accelerating]),
        ground_truth,
        initial_speed_mps=2.0,
    )

    for name in exact_metrics:
        assert combined_metrics[name] == pytest.approx(
            (exact_metrics[name] + accelerating_metrics[name]) / 2.0
        )


def test_projection_uses_anchor_containment_and_horizon_event_overlap():
    anchor = 10_000_000_000
    observations = [
        {
            "observation_uid": "obs-contained-a",
            "scene_uid": "scene-1",
            "key": "odd.road.context",
            "status": "valid",
            "values": ["urban"],
            "start_timestamp_ns": anchor - 1,
            "end_timestamp_ns": anchor + 1,
        },
        {
            "observation_uid": "obs-contained-b",
            "scene_uid": "scene-1",
            "key": "odd.road.context",
            "status": "valid",
            "values": ["urban"],
            "start_timestamp_ns": anchor,
            "end_timestamp_ns": anchor + 2,
        },
        {
            "observation_uid": "obs-ended-at-anchor",
            "scene_uid": "scene-1",
            "key": "odd.environment.sky",
            "status": "valid",
            "values": ["clear"],
            "start_timestamp_ns": anchor - 2,
            "end_timestamp_ns": anchor,
        },
        {
            "observation_uid": "obs-unavailable",
            "scene_uid": "scene-1",
            "key": "odd.road.surface_type",
            "status": "unavailable",
            "values": [],
            "start_timestamp_ns": anchor,
            "end_timestamp_ns": anchor + 2,
        },
    ]
    events = [
        {
            "event_uid": "event-overlap",
            "scene_uid": "scene-1",
            "primary_event_key": "event.ego.maneuver",
            "primary_values": ["turn_left"],
            "status": "valid",
            "start_timestamp_ns": anchor + 6_000_000_000,
            "end_timestamp_ns": anchor + 7_000_000_000,
        },
        {
            "event_uid": "event-after-horizon",
            "scene_uid": "scene-1",
            "primary_event_key": "event.ego.maneuver",
            "primary_values": ["turn_right"],
            "status": "valid",
            "start_timestamp_ns": anchor + 6_400_000_000,
            "end_timestamp_ns": anchor + 7_000_000_000,
        },
    ]

    records, summary = project_metric_samples(
        [_sample(anchor=anchor)],
        observations,
        events,
    )

    assert records[0]["label_observation_uids"] == [
        "obs-contained-a",
        "obs-contained-b",
        "obs-unavailable",
    ]
    assert records[0]["overlapping_event_uids"] == ["event-overlap"]
    assert summary["projection_policy_version"] == PROJECTION_POLICY_VERSION
    assert summary["metric_policy_version"] == METRIC_POLICY_VERSION
    assert summary["samples_with_observations"] == 1
    assert summary["samples_with_events"] == 1

    slices = {
        (
            item["kind"],
            item["key"],
            item["value"],
            item["status"],
        ): item
        for item in summary["slices"]
    }
    urban = slices[
        ("observation", "odd.road.context", "urban", "valid")
    ]
    assert urban["sample_count"] == 1
    assert urban["scene_count"] == 1
    assert (
        "observation",
        "odd.environment.sky",
        "clear",
        "valid",
    ) not in slices
    assert (
        "observation",
        "odd.road.surface_type",
        None,
        "unavailable",
    ) in slices
    assert (
        "event",
        "event.ego.maneuver",
        "turn_left",
        "valid",
    ) in slices
    assert (
        "event",
        "event.ego.maneuver",
        "turn_right",
        "valid",
    ) not in slices


def test_projection_sorts_records_and_rejects_duplicate_samples():
    samples = [_sample("sample-b"), _sample("sample-a")]

    records, summary = project_metric_samples(samples, [], [])

    assert [row["sample_uid"] for row in records] == [
        "sample-a",
        "sample-b",
    ]
    assert summary["sample_uid_digest"] == sample_uid_digest(
        ["sample-b", "sample-a"]
    )
    with pytest.raises(ValueError, match="duplicate sample_uid"):
        project_metric_samples([_sample(), _sample()], [], [])


def test_projection_identity_binds_both_dataset_manifests_and_policies():
    values = {
        "model_artifact_sha256": "1" * 64,
        "overlay_manifest_sha256": "2" * 64,
        "evaluation_dataset_manifest_sha256": "3" * 64,
        "labelset_manifest_sha256": "4" * 64,
        "labelset_dataset_manifest_sha256": "5" * 64,
        "validation_sample_uid_digest": "6" * 64,
    }

    identity = projection_cache_identity(**values)
    assert len(identity) == 64
    assert identity == projection_cache_identity(**values)

    for name in (
        "overlay_manifest_sha256",
        "evaluation_dataset_manifest_sha256",
        "labelset_manifest_sha256",
        "labelset_dataset_manifest_sha256",
        "validation_sample_uid_digest",
    ):
        changed = {**values, name: "a" * 64}
        assert projection_cache_identity(**changed) != identity


def test_metric_input_rejects_non_finite_controls():
    prediction = _controls()[None, :, :]
    prediction[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinity"):
        compute_sample_metrics(
            prediction,
            _controls(),
            initial_speed_mps=0.0,
        )
