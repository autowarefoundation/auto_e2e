"""Flyte-free batched inference for one canonical shard overlay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from Platform.pipelines.inference import predict_control


def planner_is_deterministic(model: torch.nn.Module) -> bool:
    """Return whether the active trajectory planner ignores its noise prior."""
    try:
        planner = model.Reactive_E2E.TrajectoryPlanner
    except AttributeError as exc:
        raise ValueError(
            "model does not expose Reactive_E2E.TrajectoryPlanner"
        ) from exc
    return planner.__class__.__name__.lower().startswith("bezier")


def batch_to_device(
    batch: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """Move tensor fields to the inference device without touching identities."""
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def infer_loader_controls(
    model: torch.nn.Module,
    loader: Any,
    *,
    model_artifact_id: str,
    dataset_manifest_digest: str,
    base_seeds: Sequence[int] = (0,),
    device: str | torch.device,
    training_policy: Any = None,
) -> tuple[list[str], np.ndarray, np.ndarray, tuple[int, ...]]:
    """Infer every sample from one loader and return ``uids, controls, v0``.

    Controls have shape ``[N,S,64,2]``. A deterministic Bezier planner is run
    once even if callers supply a seed fan because every draw would be identical.
    """
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("base_seeds must not be empty")
    if planner_is_deterministic(model):
        seeds = seeds[:1]

    projection = getattr(loader, "projection", None)
    if projection is not None:
        projection = projection.to(device)
    geometry_type = getattr(loader, "geometry_type", "pseudo")

    all_uids: list[str] = []
    control_batches: list[np.ndarray] = []
    speed_batches: list[np.ndarray] = []
    for raw_batch in loader:
        sample_uids = [str(uid) for uid in raw_batch["sample_uid"]]
        batch = batch_to_device(raw_batch, device)
        if training_policy is not None:
            from training.dataset_policy import adapt_egomotion_history

            batch["egomotion_history"] = adapt_egomotion_history(
                batch["egomotion_history"],
                training_policy,
            )
        per_seed = [
            predict_control(
                model,
                batch,
                sample_uids=sample_uids,
                model_artifact_id=model_artifact_id,
                dataset_manifest_digest=dataset_manifest_digest,
                base_seed=seed,
                projection=projection,
                geometry_type=geometry_type,
            )
            for seed in seeds
        ]
        controls = np.stack(per_seed, axis=1)
        history = raw_batch["egomotion_history"].reshape(
            len(sample_uids), 64, 4
        )
        speeds = history[:, -1, 0].detach().cpu().numpy().astype(np.float32)

        all_uids.extend(sample_uids)
        control_batches.append(controls)
        speed_batches.append(speeds)

    if not all_uids:
        raise ValueError("overlay loader yielded no samples")
    if len(set(all_uids)) != len(all_uids):
        raise ValueError("overlay loader yielded duplicate sample_uids")
    return (
        all_uids,
        np.concatenate(control_batches, axis=0),
        np.concatenate(speed_batches, axis=0),
        seeds,
    )
