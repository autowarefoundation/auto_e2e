"""Pose-grounded audit for target control reconstruction."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import ArrayLike

from data_processing.geospatial import decode_gps_future, decode_pose
from training.losses.control_rollout import integrate_controls_torch


AUDIT_SCHEMA_VERSION = "target_rollout_reconstruction_v2"
P95_FDE_3S_LIMIT_M = 1.0
P95_FDE_FULL_LIMIT_M = 2.0
_HISTORY_STEPS = 64
_HISTORY_SIGNALS = 4
_FUTURE_STEPS = 64
_CONTROL_SIGNALS = 2
_EGO_FLOAT_COUNT = (
    _HISTORY_STEPS * _HISTORY_SIGNALS
    + _FUTURE_STEPS * _CONTROL_SIGNALS
)
_AUDIT_MEMBER_SUFFIXES = {
    ".ego.npy": "ego",
    ".gps.npy": "gps",
    ".meta.json": "metadata",
    ".pose.npy": "pose",
}


def _as_float(value: object) -> float:
    return float(cast(Any, value))


@dataclass(frozen=True)
class PackedReconstructionInputs:
    """Minimal pose-grounded inputs read from packed WebDataset shards."""

    target_controls: np.ndarray
    logged_gps: np.ndarray
    current_poses: np.ndarray
    initial_speeds_mps: np.ndarray
    sample_uids: tuple[str, ...]
    split_group_uids: tuple[str, ...]


def load_packed_reconstruction_inputs(
    shard_dirs: Sequence[str | Path],
    *,
    validation_group_uids: Sequence[str] | None = None,
) -> PackedReconstructionInputs:
    """Read audit inputs from tar members without decoding camera or map data."""
    roots = [Path(shard_dir) for shard_dir in shard_dirs]
    if not roots:
        raise ValueError("at least one shard directory is required")
    selected_groups = (
        frozenset(str(uid) for uid in validation_group_uids)
        if validation_group_uids is not None
        else None
    )
    selected_group_count = (
        len(validation_group_uids)
        if validation_group_uids is not None
        else 0
    )
    if selected_groups is not None and (
        not selected_groups
        or len(selected_groups) != selected_group_count
        or any(not uid for uid in selected_groups)
    ):
        raise ValueError(
            "validation_group_uids must contain unique non-empty values"
        )

    records: dict[str, dict[str, bytes]] = {}
    for root in roots:
        tar_paths = sorted(root.glob("*.tar"))
        if not tar_paths:
            raise FileNotFoundError(f"No .tar shards found in {root}")
        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r:*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    matched = next(
                        (
                            (suffix, field)
                            for suffix, field in _AUDIT_MEMBER_SUFFIXES.items()
                            if member.name.endswith(suffix)
                        ),
                        None,
                    )
                    if matched is None:
                        continue
                    suffix, field = matched
                    sample_uid = member.name.removesuffix(suffix)
                    if not sample_uid:
                        raise ValueError(
                            f"empty sample UID in {member.name} from {tar_path}"
                        )
                    record = records.setdefault(sample_uid, {})
                    if field in record:
                        raise ValueError(
                            f"duplicate {field} member for {sample_uid!r}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"could not read {member.name} from {tar_path}"
                        )
                    record[field] = extracted.read()

    parsed = []
    observed_groups: set[str] = set()
    required_fields = frozenset(_AUDIT_MEMBER_SUFFIXES.values())
    for sample_uid, record in sorted(records.items()):
        metadata_bytes = record.get("metadata")
        if metadata_bytes is None:
            raise ValueError(
                f"sample {sample_uid!r} is missing audit metadata"
            )
        try:
            metadata = json.loads(metadata_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid metadata for sample {sample_uid!r}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError(
                f"metadata for sample {sample_uid!r} must be an object"
            )
        if metadata.get("sample_uid") != sample_uid:
            raise ValueError(
                f"metadata UID differs for sample {sample_uid!r}"
            )
        group_uid = metadata.get("split_group_uid")
        if not isinstance(group_uid, str) or not group_uid:
            raise ValueError(
                f"sample {sample_uid!r} has no split_group_uid"
            )
        observed_groups.add(group_uid)
        if selected_groups is not None and group_uid not in selected_groups:
            continue
        if set(record) != required_fields:
            raise ValueError(
                f"sample {sample_uid!r} has audit members "
                f"{sorted(record)}, expected {sorted(required_fields)}"
            )

        ego = np.frombuffer(record["ego"], dtype="<f4")
        if ego.shape != (_EGO_FLOAT_COUNT,):
            raise ValueError(
                f"sample {sample_uid!r} ego payload has {ego.size} floats, "
                f"expected {_EGO_FLOAT_COUNT}"
            )
        history = ego[: _HISTORY_STEPS * _HISTORY_SIGNALS].reshape(
            _HISTORY_STEPS,
            _HISTORY_SIGNALS,
        )
        controls = ego[_HISTORY_STEPS * _HISTORY_SIGNALS :].reshape(
            _FUTURE_STEPS,
            _CONTROL_SIGNALS,
        )
        pose = decode_pose(record["pose"])
        current_pose = np.asarray(
            [
                pose["latitude_deg"],
                pose["longitude_deg"],
                pose["heading_deg_cw_from_north"],
            ],
            dtype=np.float64,
        )
        gps = decode_gps_future(record["gps"])
        initial_speed = float(history[-1, 0])
        if (
            not np.isfinite(history).all()
            or not np.isfinite(controls).all()
            or not np.isfinite(current_pose).all()
            or not np.isfinite(gps).all()
            or initial_speed < 0.0
        ):
            raise ValueError(
                f"sample {sample_uid!r} has invalid audit inputs"
            )
        parsed.append(
            (
                sample_uid,
                group_uid,
                controls.copy(),
                gps,
                current_pose,
                initial_speed,
            )
        )

    if selected_groups is not None:
        missing_groups = selected_groups - observed_groups
        if missing_groups:
            raise ValueError(
                "validation groups are absent from packed shards: "
                f"{sorted(missing_groups)}"
            )
    if not parsed:
        raise ValueError("packed shards contain no selected audit samples")

    return PackedReconstructionInputs(
        target_controls=np.stack([item[2] for item in parsed]).astype(
            np.float32,
            copy=False,
        ),
        logged_gps=np.stack([item[3] for item in parsed]).astype(
            np.float64,
            copy=False,
        ),
        current_poses=np.stack([item[4] for item in parsed]).astype(
            np.float64,
            copy=False,
        ),
        initial_speeds_mps=np.asarray(
            [item[5] for item in parsed],
            dtype=np.float32,
        ),
        sample_uids=tuple(item[0] for item in parsed),
        split_group_uids=tuple(item[1] for item in parsed),
    )


def _identity_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")
    ).hexdigest()


def _distribution(values: ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("audit distribution must be finite and non-empty")
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "max": float(array.max()),
    }


def _optional_distribution(
    values: np.ndarray,
) -> dict[str, float] | None:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return _distribution(flattened) if flattened.size else None


def _heading_alignment(
    logged_xy_m: np.ndarray,
    target_headings_rad: np.ndarray,
    *,
    three_second_steps: int,
    full_horizon_steps: int,
    minimum_baseline_m: float,
) -> dict[str, object]:
    """Compare control headings with centered logged-path tangents."""
    if not np.isfinite(minimum_baseline_m) or minimum_baseline_m <= 0.0:
        raise ValueError("minimum heading baseline must be finite and positive")
    logged = np.asarray(logged_xy_m, dtype=np.float64)
    target = np.asarray(target_headings_rad, dtype=np.float64)
    if target.shape != logged.shape[:2]:
        raise ValueError("target headings must match logged trajectory steps")
    origin = np.zeros_like(logged[:, :1])
    previous = np.concatenate([origin, logged[:, :-1]], axis=1)
    following = np.concatenate(
        [logged[:, 1:], logged[:, -1:]],
        axis=1,
    )
    baseline = following - previous
    baseline_length = np.linalg.norm(baseline, axis=2)
    valid = baseline_length >= minimum_baseline_m
    logged_headings = np.arctan2(baseline[:, :, 1], baseline[:, :, 0])
    difference = target - logged_headings
    error_deg = np.rad2deg(np.abs(np.arctan2(
        np.sin(difference),
        np.cos(difference),
    )))

    def distribution_for(
        errors: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float] | None:
        return _optional_distribution(errors[mask])

    return {
        "method": "centered_logged_xy_tangent_v1",
        "minimum_baseline_m": minimum_baseline_m,
        "valid_step_count": int(valid[:, :full_horizon_steps].sum()),
        "valid_sample_count": int(
            np.any(valid[:, :full_horizon_steps], axis=1).sum()
        ),
        "three_seconds": distribution_for(
            error_deg[:, :three_second_steps],
            valid[:, :three_second_steps],
        ),
        "full_horizon": distribution_for(
            error_deg[:, :full_horizon_steps],
            valid[:, :full_horizon_steps],
        ),
        "at_3s": distribution_for(
            error_deg[:, three_second_steps - 1],
            valid[:, three_second_steps - 1],
        ),
        "at_full_horizon": distribution_for(
            error_deg[:, full_horizon_steps - 1],
            valid[:, full_horizon_steps - 1],
        ),
    }


def audit_target_rollout_reconstruction(
    target_controls: np.ndarray,
    logged_xy_m: np.ndarray,
    initial_speeds_mps: np.ndarray,
    sample_uids: Sequence[str],
    split_group_uids: Sequence[str],
    *,
    dt: float = 0.1,
    three_second_steps: int = 30,
    full_horizon_steps: int = 64,
    p95_fde_3s_limit_m: float = P95_FDE_3S_LIMIT_M,
    p95_fde_full_limit_m: float = P95_FDE_FULL_LIMIT_M,
    minimum_heading_baseline_m: float = 0.1,
) -> dict[str, object]:
    """Compare integrated target controls with logged ego-frame future XY."""
    controls = np.ascontiguousarray(target_controls, dtype=np.float32)
    logged = np.ascontiguousarray(logged_xy_m, dtype=np.float64)
    speeds = np.ascontiguousarray(initial_speeds_mps, dtype=np.float32)
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("target_controls must have shape [B,T,2]")
    if logged.shape != controls.shape:
        raise ValueError("logged_xy_m must match target_controls shape")
    batch_size, timestep_count, _ = controls.shape
    if batch_size == 0:
        raise ValueError("audit inputs must contain at least one sample")
    if speeds.shape != (batch_size,):
        raise ValueError("initial_speeds_mps must have shape [B]")
    if len(sample_uids) != batch_size or len(split_group_uids) != batch_size:
        raise ValueError("audit identities must match the batch size")
    if len(set(sample_uids)) != batch_size or any(not uid for uid in sample_uids):
        raise ValueError("sample_uids must be non-empty and unique")
    if any(not uid for uid in split_group_uids):
        raise ValueError("split_group_uids must be non-empty")
    if not np.isfinite(controls).all() or not np.isfinite(logged).all():
        raise ValueError("audit trajectories contain non-finite values")
    if not np.isfinite(speeds).all() or np.any(speeds < 0.0):
        raise ValueError("audit initial speeds must be finite and non-negative")
    if not 0 < three_second_steps <= full_horizon_steps <= timestep_count:
        raise ValueError("audit horizons do not fit the trajectory")
    if not np.isclose(three_second_steps * dt, 3.0):
        raise ValueError("three_second_steps and dt must describe 3 seconds")
    if not np.isclose(full_horizon_steps * dt, 6.4):
        raise ValueError(
            "full_horizon_steps and dt must describe 6.4 seconds"
        )

    with torch.no_grad():
        predicted_xy, target_headings, _ = integrate_controls_torch(
            torch.from_numpy(controls),
            torch.from_numpy(speeds),
            dt=dt,
        )
    errors = np.linalg.norm(
        predicted_xy.numpy().astype(np.float64) - logged,
        axis=2,
    )
    metric_arrays: dict[str, np.ndarray] = {
        "ade_3s_m": errors[:, :three_second_steps].mean(axis=1),
        "fde_3s_m": errors[:, three_second_steps - 1],
        "ade_full_m": errors[:, :full_horizon_steps].mean(axis=1),
        "fde_full_m": errors[:, full_horizon_steps - 1],
    }

    records: list[dict[str, object]] = []
    for index, (sample_uid, group_uid) in enumerate(
        zip(sample_uids, split_group_uids, strict=True)
    ):
        sample_metrics = {
            name: float(values[index])
            for name, values in metric_arrays.items()
        }
        records.append({
            "sample_uid": sample_uid,
            "split_group_uid": group_uid,
            **sample_metrics,
        })
    records.sort(key=lambda record: str(record["sample_uid"]))

    scene_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for name in metric_arrays:
            scene_values[str(record["split_group_uid"])][name].append(
                _as_float(record[name])
            )

    scenes: list[dict[str, object]] = []
    for group_uid in sorted(scene_values):
        first_metric = next(iter(metric_arrays))
        scenes.append({
            "split_group_uid": group_uid,
            "sample_count": len(scene_values[group_uid][first_metric]),
            **{
                name: float(np.mean(values))
                for name, values in scene_values[group_uid].items()
            },
        })
    metric_summaries: dict[str, dict[str, dict[str, float]]] = {
        name: {
            "natural": _distribution([
                _as_float(record[name]) for record in records
            ]),
            "scene_mean_distribution": _distribution([
                _as_float(scene[name]) for scene in scenes
            ]),
        }
        for name in metric_arrays
    }
    thresholds_pass = (
        metric_summaries["fde_3s_m"]["natural"]["p95"]
        <= p95_fde_3s_limit_m
        and metric_summaries["fde_full_m"]["natural"]["p95"]
        <= p95_fde_full_limit_m
    )
    worst_scenes = {
        name: [
            {
                "split_group_uid": scene["split_group_uid"],
                "value": _as_float(scene[name]),
            }
            for scene in sorted(
                scenes,
                key=lambda item: (
                    -_as_float(item[name]),
                    str(item["split_group_uid"]),
                ),
            )[:10]
        ]
        for name in ("fde_3s_m", "fde_full_m")
    }
    error_by_step = [
        {
            "step": step + 1,
            "horizon_seconds": float((step + 1) * dt),
            "natural": _distribution(errors[:, step]),
        }
        for step in range(full_horizon_steps)
    ]
    heading_alignment = _heading_alignment(
        logged,
        target_headings.numpy(),
        three_second_steps=three_second_steps,
        full_horizon_steps=full_horizon_steps,
        minimum_baseline_m=minimum_heading_baseline_m,
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sample_count": batch_size,
        "scene_count": len(scenes),
        "sample_uid_digest": _identity_digest(sample_uids),
        "split_group_uid_digest": _identity_digest(
            sorted(set(split_group_uids))
        ),
        "horizons": {
            "three_second_steps": three_second_steps,
            "full_horizon_steps": full_horizon_steps,
            "dt": dt,
        },
        "thresholds": {
            "p95_fde_3s_limit_m": p95_fde_3s_limit_m,
            "p95_fde_full_limit_m": p95_fde_full_limit_m,
        },
        "thresholds_pass": thresholds_pass,
        "decision": {
            "status": "pending_review",
            "automatic_recommendation": (
                "go" if thresholds_pass else "review_required"
            ),
            "rationale": None,
        },
        "input_quality": {
            "missing_sample_count": 0,
            "non_finite_sample_count": 0,
        },
        "metrics": metric_summaries,
        "heading_alignment": heading_alignment,
        "error_by_step": error_by_step,
        "worst_scenes": worst_scenes,
        "scenes": scenes,
        "records": records,
    }


def audit_packed_target_rollout_reconstruction(
    inputs: PackedReconstructionInputs,
    **audit_kwargs: Any,
) -> dict[str, object]:
    """Audit packed controls against GPS point 1..64 in current ego FLU."""
    from evaluation.kitscenes_benchmark import (
        wgs84_trajectory_to_ego_xy,
    )

    logged_xy = wgs84_trajectory_to_ego_xy(
        inputs.logged_gps,
        inputs.current_poses,
    )
    if logged_xy.shape != inputs.target_controls.shape:
        raise ValueError(
            "pose-grounded trajectory does not match the control horizon"
        )
    return audit_target_rollout_reconstruction(
        inputs.target_controls,
        logged_xy,
        inputs.initial_speeds_mps,
        inputs.sample_uids,
        inputs.split_group_uids,
        **audit_kwargs,
    )
