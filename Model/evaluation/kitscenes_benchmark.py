"""Reproducible KITScenes Multimodal E2E benchmark contracts.

KITScenes currently publishes the paper protocol but not the exact 200-sample
manifest or community evaluator. This module therefore makes provenance an
input: a checkpoint can be evaluated later against an immutable manifest, while
MLflow can distinguish development, paper-protocol approximation, and official
release results.

Only displacement metrics are computed here. Drivable-surface survival,
collision-free rate, centerline distance, and MMS require benchmark-authority
map/occupancy/maneuver assets and must not be approximated from the model's
raster map input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluation.metrics import integrate_trajectory


MANIFEST_SCHEMA_VERSION = "kitscenes_e2e_benchmark_manifest_v1"
PROTOCOL_ID = "kitscenes_multimodal_e2e_v1"
EVALUATOR_VERSION = "kitscenes_control_displacement_v1"
PAPER_PROTOCOL_SOURCE = "https://arxiv.org/html/2606.02956#A8.SS4"

_PROTOCOL_STATUSES = {
    "development",
    "paper_protocol_approximation",
    "official",
}
_KNOWN_SPLITS = {
    "train",
    "val",
    "overlap-train-val",
    "test",
    "test-e2e",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_SAMPLE_UID = re.compile(
    r"^kitscenes-v1-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-"
    r"[0-9a-f]{12}-f[0-9]{6}$"
)
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class KITScenesBenchmarkManifest:
    benchmark_id: str
    protocol_status: str
    protocol_source: str
    authority: str
    release_id: str
    dataset_revision: str
    sdk_revision: str
    source_splits: tuple[str, ...]
    sample_uids: tuple[str, ...]
    frequency_hz: int
    past_seconds: int
    horizons_seconds: tuple[int, ...]
    input_track: str
    history_adapter: str

    @property
    def observation_steps(self) -> int:
        return self.frequency_hz * self.past_seconds

    @property
    def horizon_steps(self) -> tuple[int, ...]:
        return tuple(
            self.frequency_hz * seconds
            for seconds in self.horizons_seconds
        )


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"benchmark manifest is missing {key!r}")
    return payload[key]


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def parse_benchmark_manifest(
    payload: Mapping[str, Any],
) -> KITScenesBenchmarkManifest:
    """Validate and normalize one immutable benchmark sample manifest."""
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark manifest schema "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"benchmark protocol_id must be {PROTOCOL_ID!r}"
        )

    benchmark_id = str(_required(payload, "benchmark_id"))
    if not _SAFE_ID.fullmatch(benchmark_id):
        raise ValueError(f"unsafe benchmark_id {benchmark_id!r}")

    protocol_status = str(_required(payload, "protocol_status"))
    if protocol_status not in _PROTOCOL_STATUSES:
        raise ValueError(
            f"unsupported protocol_status {protocol_status!r}"
        )
    protocol_source = str(_required(payload, "protocol_source"))
    if not protocol_source.startswith("https://"):
        raise ValueError("protocol_source must be an HTTPS URL")

    authority = str(_required(payload, "authority"))
    release_id = str(_required(payload, "release_id"))
    if not _SAFE_ID.fullmatch(authority):
        raise ValueError(f"unsafe authority {authority!r}")
    if not _SAFE_ID.fullmatch(release_id):
        raise ValueError(f"unsafe release_id {release_id!r}")

    dataset_revision = str(_required(payload, "dataset_revision"))
    sdk_revision = str(_required(payload, "sdk_revision"))
    if not _COMMIT_SHA.fullmatch(dataset_revision):
        raise ValueError("dataset_revision must be a 40-character commit SHA")
    if not _COMMIT_SHA.fullmatch(sdk_revision):
        raise ValueError("sdk_revision must be a 40-character commit SHA")

    raw_splits = _required(payload, "source_splits")
    if (
        not isinstance(raw_splits, list)
        or not raw_splits
        or any(not isinstance(split, str) for split in raw_splits)
    ):
        raise ValueError("source_splits must be a non-empty string list")
    source_splits = tuple(raw_splits)
    if len(set(source_splits)) != len(source_splits):
        raise ValueError("source_splits contains duplicates")
    unknown_splits = set(source_splits) - _KNOWN_SPLITS
    if unknown_splits:
        raise ValueError(
            f"benchmark manifest has unknown source splits {sorted(unknown_splits)}"
        )

    raw_uids = _required(payload, "sample_uids")
    if (
        not isinstance(raw_uids, list)
        or not raw_uids
        or any(not isinstance(uid, str) for uid in raw_uids)
    ):
        raise ValueError("sample_uids must be a non-empty string list")
    sample_uids = tuple(raw_uids)
    if len(set(sample_uids)) != len(sample_uids):
        raise ValueError("sample_uids contains duplicates")
    invalid_uids = [
        uid for uid in sample_uids if not _SAFE_SAMPLE_UID.fullmatch(uid)
    ]
    if invalid_uids:
        raise ValueError(
            f"benchmark manifest has unsafe sample UIDs {invalid_uids[:3]}"
        )
    sample_count = _positive_int(
        _required(payload, "sample_count"), "sample_count"
    )
    if sample_count != len(sample_uids):
        raise ValueError(
            f"sample_count={sample_count} does not match "
            f"{len(sample_uids)} sample_uids"
        )

    frequency_hz = _positive_int(
        _required(payload, "frequency_hz"), "frequency_hz"
    )
    past_seconds = _positive_int(
        _required(payload, "past_seconds"), "past_seconds"
    )
    raw_horizons = _required(payload, "horizons_seconds")
    if (
        not isinstance(raw_horizons, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_horizons
        )
    ):
        raise ValueError("horizons_seconds must be an integer list")
    horizons_seconds = tuple(raw_horizons)
    if (
        frequency_hz != 10
        or past_seconds != 4
        or horizons_seconds != (3, 5)
    ):
        raise ValueError(
            "KITScenes E2E protocol requires 10 Hz, 4 seconds of past "
            "observation, and [3, 5] second horizons"
        )

    input_track = str(_required(payload, "input_track"))
    if not _SAFE_ID.fullmatch(input_track):
        raise ValueError(f"unsafe input_track {input_track!r}")
    history_adapter = str(_required(payload, "history_adapter"))
    if history_adapter != "left_zero_pad_to_64":
        raise ValueError(
            "history_adapter must be 'left_zero_pad_to_64'"
        )

    if protocol_status == "official":
        if authority != "KIT-MRT":
            raise ValueError(
                "official benchmark manifests must be issued by KIT-MRT"
            )
        if sample_count != 200:
            raise ValueError(
                "official KITScenes E2E manifests must contain 200 samples"
            )
    elif protocol_status == "paper_protocol_approximation":
        if protocol_source != PAPER_PROTOCOL_SOURCE:
            raise ValueError(
                "paper protocol approximation must cite the KITScenes paper"
            )
        if set(source_splits) != {"val", "overlap-train-val"}:
            raise ValueError(
                "paper protocol approximation must use val and "
                "overlap-train-val"
            )
        if sample_count != 200:
            raise ValueError(
                "paper protocol approximation must contain 200 samples"
            )

    return KITScenesBenchmarkManifest(
        benchmark_id=benchmark_id,
        protocol_status=protocol_status,
        protocol_source=protocol_source,
        authority=authority,
        release_id=release_id,
        dataset_revision=dataset_revision,
        sdk_revision=sdk_revision,
        source_splits=source_splits,
        sample_uids=sample_uids,
        frequency_hz=frequency_hz,
        past_seconds=past_seconds,
        horizons_seconds=horizons_seconds,
        input_track=input_track,
        history_adapter=history_adapter,
    )


def load_benchmark_manifest(
    path: str | Path,
) -> tuple[KITScenesBenchmarkManifest, str]:
    manifest_path = Path(path)
    payload_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("benchmark manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest root must be an object")
    return (
        parse_benchmark_manifest(payload),
        hashlib.sha256(payload_bytes).hexdigest(),
    )


def sample_uid_digest(sample_uids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sample_uids)).encode("utf-8")
    ).hexdigest()


def limit_egomotion_history(
    history: torch.Tensor,
    *,
    observation_steps: int,
) -> torch.Tensor:
    """Mask context older than the protocol window without changing model ABI."""
    if history.ndim != 2 or history.shape[1] != 64 * 4:
        raise ValueError(
            "egomotion history must have shape [batch, 256]"
        )
    if not 0 < observation_steps <= 64:
        raise ValueError("observation_steps must be between 1 and 64")
    limited = history.reshape(history.shape[0], 64, 4).clone()
    limited[:, : 64 - observation_steps, :] = 0
    return limited.reshape(history.shape[0], 64 * 4)


def compute_displacement_metrics(
    predicted_controls: np.ndarray,
    target_controls: np.ndarray,
    initial_speeds: np.ndarray,
    *,
    frequency_hz: int = 10,
    horizons_seconds: Sequence[int] = (3, 5),
) -> tuple[dict[str, float], np.ndarray]:
    """Compute KITScenes ADE/FDE horizons and return predicted XY trajectories."""
    predicted = np.asarray(predicted_controls, dtype=np.float64)
    target = np.asarray(target_controls, dtype=np.float64)
    speeds = np.asarray(initial_speeds, dtype=np.float64)
    if (
        predicted.ndim != 3
        or predicted.shape[2] != 2
        or target.shape != predicted.shape
    ):
        raise ValueError(
            "predicted_controls and target_controls must share shape [B,T,2]"
        )
    if speeds.shape != (predicted.shape[0],):
        raise ValueError("initial_speeds must have shape [B]")
    if predicted.shape[0] == 0:
        raise ValueError("benchmark batch must not be empty")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("benchmark controls contain non-finite values")
    if not np.isfinite(speeds).all():
        raise ValueError("benchmark initial speeds contain non-finite values")
    if np.any(speeds < 0):
        raise ValueError("benchmark initial speeds must be non-negative")

    horizon_steps = tuple(
        _positive_int(seconds, "horizon")
        * _positive_int(frequency_hz, "frequency_hz")
        for seconds in horizons_seconds
    )
    if not horizon_steps or max(horizon_steps) > predicted.shape[1]:
        raise ValueError("benchmark horizon exceeds predicted trajectory")

    max_steps = max(horizon_steps)
    predicted_xy = np.empty(
        (predicted.shape[0], max_steps, 2), dtype=np.float64
    )
    target_xy = np.empty_like(predicted_xy)
    for index in range(predicted.shape[0]):
        predicted_xy[index] = integrate_trajectory(
            predicted[index, :max_steps, 0],
            predicted[index, :max_steps, 1],
            float(speeds[index]),
            dt=1.0 / frequency_hz,
        )
        target_xy[index] = integrate_trajectory(
            target[index, :max_steps, 0],
            target[index, :max_steps, 1],
            float(speeds[index]),
            dt=1.0 / frequency_hz,
        )

    errors = np.linalg.norm(predicted_xy - target_xy, axis=2)
    metrics: dict[str, float] = {}
    for seconds, steps in zip(horizons_seconds, horizon_steps):
        metrics[f"ade_{seconds}s"] = float(errors[:, :steps].mean())
        metrics[f"fde_{seconds}s"] = float(errors[:, steps - 1].mean())
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite benchmark metrics: {metrics}")
    return metrics, predicted_xy
