"""Pose-grounded audit for target control reconstruction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from training.losses.control_rollout import integrate_controls_torch


AUDIT_SCHEMA_VERSION = "target_rollout_reconstruction_v1"


def _identity_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")
    ).hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float]:
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
    p95_fde_3s_limit_m: float = 1.0,
    p95_fde_full_limit_m: float = 2.0,
) -> dict[str, object]:
    """Compare integrated target controls with logged ego-frame future XY."""
    controls = np.asarray(target_controls, dtype=np.float32)
    logged = np.asarray(logged_xy_m, dtype=np.float64)
    speeds = np.asarray(initial_speeds_mps, dtype=np.float32)
    if controls.ndim != 3 or controls.shape[2] != 2:
        raise ValueError("target_controls must have shape [B,T,2]")
    if logged.shape != controls.shape:
        raise ValueError("logged_xy_m must match target_controls shape")
    batch_size, timestep_count, _ = controls.shape
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

    with torch.no_grad():
        predicted_xy, _, _ = integrate_controls_torch(
            torch.from_numpy(controls),
            torch.from_numpy(speeds),
            dt=dt,
        )
    errors = np.linalg.norm(
        predicted_xy.numpy().astype(np.float64) - logged,
        axis=2,
    )
    metric_arrays = {
        "ade_3s_m": errors[:, :three_second_steps].mean(axis=1),
        "fde_3s_m": errors[:, three_second_steps - 1],
        "ade_full_m": errors[:, :full_horizon_steps].mean(axis=1),
        "fde_full_m": errors[:, full_horizon_steps - 1],
    }

    records = []
    scene_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (sample_uid, group_uid) in enumerate(
        zip(sample_uids, split_group_uids, strict=True)
    ):
        metrics = {
            name: float(values[index])
            for name, values in metric_arrays.items()
        }
        records.append({
            "sample_uid": sample_uid,
            "split_group_uid": group_uid,
            **metrics,
        })
        for name, value in metrics.items():
            scene_values[group_uid][name].append(value)

    scenes = []
    for group_uid in sorted(scene_values):
        scenes.append({
            "split_group_uid": group_uid,
            "sample_count": len(scene_values[group_uid]["fde_full_m"]),
            **{
                name: float(np.mean(values))
                for name, values in scene_values[group_uid].items()
            },
        })
    metrics = {
        name: {
            "natural": _distribution(values),
            "scene_mean_distribution": _distribution([
                float(scene[name]) for scene in scenes
            ]),
        }
        for name, values in metric_arrays.items()
    }
    go = (
        metrics["fde_3s_m"]["natural"]["p95"] <= p95_fde_3s_limit_m
        and metrics["fde_full_m"]["natural"]["p95"]
        <= p95_fde_full_limit_m
    )
    worst_scenes = {
        name: [
            {
                "split_group_uid": scene["split_group_uid"],
                "value": float(scene[name]),
            }
            for scene in sorted(
                scenes,
                key=lambda item: (-float(item[name]), item["split_group_uid"]),
            )[:10]
        ]
        for name in ("fde_3s_m", "fde_full_m")
    }
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
        "go": go,
        "metrics": metrics,
        "worst_scenes": worst_scenes,
        "scenes": scenes,
        "records": records,
    }
