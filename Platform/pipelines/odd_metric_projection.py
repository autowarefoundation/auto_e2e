"""Derived model-metric projection onto scene-native ODD intervals.

The projection is an analysis artifact. It is not part of an ODD LabelSet and
must never be consumed as a model input or training target.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROJECTION_SCHEMA_VERSION = "odd_model_metric_projection_v1"
PROJECTION_POLICY_VERSION = "odd_interval_projection_v1"
METRIC_POLICY_VERSION = "control_displacement_seed_mean_v1"

METRIC_NAMES = (
    "ade_1s_m",
    "ade_2s_m",
    "ade_3s_m",
    "ade_horizon_m",
    "fde_horizon_m",
    "acceleration_mae",
    "curvature_mae",
)


@dataclass(frozen=True)
class MetricSample:
    """One model evaluation row before its ODD interval join."""

    sample_uid: str
    scene_uid: str
    split_group_uid: str
    sample_anchor_timestamp_ns: int
    predicted_controls: np.ndarray
    ground_truth_controls: np.ndarray
    initial_speed_mps: float


@dataclass
class _MetricAccumulator:
    count: int
    sums: dict[str, float]
    scenes: set[str]

    @classmethod
    def empty(cls) -> "_MetricAccumulator":
        return cls(
            count=0,
            sums={name: 0.0 for name in METRIC_NAMES},
            scenes=set(),
        )

    def add(
        self,
        metrics: Mapping[str, float],
        *,
        scene_uid: str,
    ) -> None:
        self.count += 1
        self.scenes.add(scene_uid)
        for name in METRIC_NAMES:
            self.sums[name] += float(metrics[name])

    def result(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("metric accumulator must not be empty")
        return {
            "sample_count": self.count,
            "scene_count": len(self.scenes),
            "metrics": {
                name: self.sums[name] / self.count
                for name in METRIC_NAMES
            },
        }


def _sha256_hex(value: str, name: str) -> str:
    normalized = str(value).removeprefix("sha256:")
    if (
        len(normalized) != 64
        or normalized.lower() != normalized
        or any(char not in "0123456789abcdef" for char in normalized)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def projection_cache_identity(
    *,
    model_artifact_sha256: str,
    overlay_manifest_sha256: str,
    evaluation_dataset_manifest_sha256: str,
    labelset_manifest_sha256: str,
    labelset_dataset_manifest_sha256: str,
    validation_sample_uid_digest: str,
) -> str:
    """Hash every immutable input and policy that determines a projection."""
    payload = {
        "evaluation_dataset_manifest_sha256": _sha256_hex(
            evaluation_dataset_manifest_sha256,
            "evaluation_dataset_manifest_sha256",
        ),
        "labelset_dataset_manifest_sha256": _sha256_hex(
            labelset_dataset_manifest_sha256,
            "labelset_dataset_manifest_sha256",
        ),
        "labelset_manifest_sha256": _sha256_hex(
            labelset_manifest_sha256,
            "labelset_manifest_sha256",
        ),
        "metric_policy_version": METRIC_POLICY_VERSION,
        "model_artifact_sha256": _sha256_hex(
            model_artifact_sha256,
            "model_artifact_sha256",
        ),
        "overlay_manifest_sha256": _sha256_hex(
            overlay_manifest_sha256,
            "overlay_manifest_sha256",
        ),
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "validation_sample_uid_digest": _sha256_hex(
            validation_sample_uid_digest,
            "validation_sample_uid_digest",
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sample_uid_digest(sample_uids: Sequence[str]) -> str:
    """Return the canonical digest for one unique evaluation population."""
    normalized = sorted(str(uid) for uid in sample_uids)
    if not normalized or any(not uid for uid in normalized):
        raise ValueError("sample_uids must contain non-empty values")
    if len(normalized) != len(set(normalized)):
        raise ValueError("sample_uids must be unique")
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _control_array(
    value: np.ndarray,
    *,
    name: str,
    seed_axis: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not seed_axis and array.ndim == 2:
        expected_tail = (array.shape[0], 2)
    elif seed_axis and array.ndim == 3:
        expected_tail = (array.shape[0], array.shape[1], 2)
    else:
        shape = "[seed,horizon,2]" if seed_axis else "[horizon,2]"
        raise ValueError(f"{name} must have shape {shape}")
    if array.shape != expected_tail or array.shape[-1] != 2:
        raise ValueError(f"{name} must have acceleration/curvature pairs")
    if array.shape[-2] < 30:
        raise ValueError(f"{name} must cover at least three seconds at 10 Hz")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _integrate_controls(
    controls: np.ndarray,
    initial_speed_mps: float,
    dt_seconds: float,
) -> np.ndarray:
    """Integrate ``[seed,horizon,(acceleration,curvature)]`` controls."""
    seed_count, horizon, _ = controls.shape
    positions = np.zeros((seed_count, horizon, 2), dtype=np.float64)
    speed = np.full(seed_count, initial_speed_mps, dtype=np.float64)
    heading = np.zeros(seed_count, dtype=np.float64)
    x = np.zeros(seed_count, dtype=np.float64)
    y = np.zeros(seed_count, dtype=np.float64)
    for step in range(horizon):
        speed = np.maximum(
            0.0,
            speed + controls[:, step, 0] * dt_seconds,
        )
        heading = (
            heading + controls[:, step, 1] * speed * dt_seconds
        )
        x = x + speed * np.cos(heading) * dt_seconds
        y = y + speed * np.sin(heading) * dt_seconds
        positions[:, step, 0] = x
        positions[:, step, 1] = y
    return positions


def compute_sample_metrics(
    predicted_controls: np.ndarray,
    ground_truth_controls: np.ndarray,
    *,
    initial_speed_mps: float,
    frequency_hz: int = 10,
) -> dict[str, float]:
    """Compute one row's displacement metrics, averaged across model seeds."""
    predicted = _control_array(
        predicted_controls,
        name="predicted_controls",
        seed_axis=True,
    )
    ground_truth = _control_array(
        ground_truth_controls,
        name="ground_truth_controls",
        seed_axis=False,
    )
    if predicted.shape[1] != ground_truth.shape[0]:
        raise ValueError("prediction and ground truth horizons differ")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if not math.isfinite(initial_speed_mps) or initial_speed_mps < 0.0:
        raise ValueError("initial_speed_mps must be finite and non-negative")

    dt_seconds = 1.0 / frequency_hz
    predicted_xy = _integrate_controls(
        predicted,
        initial_speed_mps,
        dt_seconds,
    )
    ground_truth_xy = _integrate_controls(
        ground_truth[None, :, :],
        initial_speed_mps,
        dt_seconds,
    )[0]
    displacement = np.linalg.norm(
        predicted_xy - ground_truth_xy[None, :, :],
        axis=2,
    )

    def horizon_steps(seconds: int) -> int:
        return min(displacement.shape[1], seconds * frequency_hz)

    metrics = {
        "ade_1s_m": float(
            displacement[:, :horizon_steps(1)].mean()
        ),
        "ade_2s_m": float(
            displacement[:, :horizon_steps(2)].mean()
        ),
        "ade_3s_m": float(
            displacement[:, :horizon_steps(3)].mean()
        ),
        "ade_horizon_m": float(displacement.mean()),
        "fde_horizon_m": float(displacement[:, -1].mean()),
        "acceleration_mae": float(
            np.abs(
                predicted[:, :, 0] - ground_truth[None, :, 0]
            ).mean()
        ),
        "curvature_mae": float(
            np.abs(
                predicted[:, :, 1] - ground_truth[None, :, 1]
            ).mean()
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("computed metrics contain NaN or infinity")
    return metrics


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, ""))
    if not value:
        raise ValueError(f"{key} must be provided")
    return value


def _interval(row: Mapping[str, Any]) -> tuple[int, int]:
    start = int(row["start_timestamp_ns"])
    end = int(row["end_timestamp_ns"])
    if start < 0 or end <= start:
        raise ValueError("interval must be non-empty and non-negative")
    return start, end


def _values(row: Mapping[str, Any], key: str) -> tuple[str | None, ...]:
    raw = row.get(key, [])
    if raw is None:
        raw = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    values = tuple(sorted({str(value) for value in raw if str(value)}))
    return values or (None,)


def _slice_result(
    identity: tuple[str, str, str | None, str],
    accumulator: _MetricAccumulator,
) -> dict[str, Any]:
    kind, label_key, label_value, status = identity
    return {
        "kind": kind,
        "key": label_key,
        "value": label_value,
        "status": status,
        **accumulator.result(),
    }


def project_metric_samples(
    samples: Sequence[MetricSample],
    observations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    frequency_hz: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join evaluation rows to ODD intervals and aggregate metric slices.

    Observation intervals use half-open anchor containment:
    ``start <= anchor < end``. Events use positive overlap with the model
    horizon: ``event_start < horizon_end and event_end > anchor``.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    observations_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observation_uids: set[str] = set()
    for observation in observations:
        uid = _required_string(observation, "observation_uid")
        scene_uid = _required_string(observation, "scene_uid")
        _required_string(observation, "key")
        _required_string(observation, "status")
        _interval(observation)
        _values(observation, "values")
        if uid in observation_uids:
            raise ValueError(f"duplicate observation_uid: {uid}")
        observation_uids.add(uid)
        observations_by_scene[scene_uid].append(observation)

    events_by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    event_uids: set[str] = set()
    for event in events:
        uid = _required_string(event, "event_uid")
        scene_uid = _required_string(event, "scene_uid")
        _required_string(event, "primary_event_key")
        _interval(event)
        _values(event, "primary_values")
        if uid in event_uids:
            raise ValueError(f"duplicate event_uid: {uid}")
        event_uids.add(uid)
        events_by_scene[scene_uid].append(event)

    for rows in observations_by_scene.values():
        rows.sort(
            key=lambda row: (
                int(row["start_timestamp_ns"]),
                int(row["end_timestamp_ns"]),
                str(row["observation_uid"]),
            )
        )
    for rows in events_by_scene.values():
        rows.sort(
            key=lambda row: (
                int(row["start_timestamp_ns"]),
                int(row["end_timestamp_ns"]),
                str(row["event_uid"]),
            )
        )

    sample_uids: set[str] = set()
    overall = _MetricAccumulator.empty()
    slice_metrics: dict[
        tuple[str, str, str | None, str],
        _MetricAccumulator,
    ] = defaultdict(_MetricAccumulator.empty)
    records: list[dict[str, Any]] = []
    samples_with_observations = 0
    samples_with_events = 0

    for sample in sorted(samples, key=lambda item: item.sample_uid):
        if (
            not sample.sample_uid
            or not sample.scene_uid
            or not sample.split_group_uid
        ):
            raise ValueError("sample identities must be non-empty")
        if sample.sample_uid in sample_uids:
            raise ValueError(f"duplicate sample_uid: {sample.sample_uid}")
        sample_uids.add(sample.sample_uid)
        anchor = int(sample.sample_anchor_timestamp_ns)
        if anchor < 0:
            raise ValueError("sample anchor timestamp must be non-negative")

        metrics = compute_sample_metrics(
            sample.predicted_controls,
            sample.ground_truth_controls,
            initial_speed_mps=float(sample.initial_speed_mps),
            frequency_hz=frequency_hz,
        )
        horizon_steps = int(
            np.asarray(sample.ground_truth_controls).shape[0]
        )
        horizon_end = anchor + round(
            horizon_steps * 1_000_000_000 / frequency_hz
        )

        matched_observations = []
        matched_events = []
        identities: set[tuple[str, str, str | None, str]] = set()
        for observation in observations_by_scene.get(sample.scene_uid, []):
            start, end = _interval(observation)
            if start <= anchor < end:
                matched_observations.append(
                    str(observation["observation_uid"])
                )
                for value in _values(observation, "values"):
                    identities.add((
                        "observation",
                        str(observation["key"]),
                        value,
                        str(observation["status"]),
                    ))
        for event in events_by_scene.get(sample.scene_uid, []):
            start, end = _interval(event)
            if start < horizon_end and end > anchor:
                matched_events.append(str(event["event_uid"]))
                for value in _values(event, "primary_values"):
                    identities.add((
                        "event",
                        str(event["primary_event_key"]),
                        value,
                        str(event.get("status", "valid")),
                    ))

        if matched_observations:
            samples_with_observations += 1
        if matched_events:
            samples_with_events += 1
        overall.add(metrics, scene_uid=sample.scene_uid)
        for identity in identities:
            slice_metrics[identity].add(
                metrics,
                scene_uid=sample.scene_uid,
            )

        records.append({
            "sample_uid": sample.sample_uid,
            "scene_uid": sample.scene_uid,
            "split_group_uid": sample.split_group_uid,
            "sample_anchor_timestamp_ns": anchor,
            "label_observation_uids": sorted(set(matched_observations)),
            "overlapping_event_uids": sorted(set(matched_events)),
            "projection_policy_version": PROJECTION_POLICY_VERSION,
            "metrics": metrics,
        })

    slices = [
        _slice_result(identity, accumulator)
        for identity, accumulator in sorted(
            slice_metrics.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2] or "",
                item[0][3],
            ),
        )
    ]
    summary = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "frequency_hz": frequency_hz,
        "horizon_steps": int(
            np.asarray(samples[0].ground_truth_controls).shape[0]
        ),
        "horizon_seconds": (
            np.asarray(samples[0].ground_truth_controls).shape[0]
            / frequency_hz
        ),
        "observation_join": "start <= anchor < end",
        "event_join": (
            "event_start < anchor + model_horizon and event_end > anchor"
        ),
        "seed_aggregation": "arithmetic_mean",
        "sample_uid_digest": sample_uid_digest(
            [sample.sample_uid for sample in samples]
        ),
        "sample_count": len(samples),
        "scene_count": len({sample.scene_uid for sample in samples}),
        "samples_with_observations": samples_with_observations,
        "samples_with_events": samples_with_events,
        "overall": overall.result(),
        "slices": slices,
    }
    return records, summary
