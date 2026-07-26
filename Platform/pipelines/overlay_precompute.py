"""Flyte-free batched inference for one canonical shard overlay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from Platform.pipelines.inference import predict_control
from Platform.pipelines.overlay import BEV_HEATMAP_NAMES, BEV_HEATMAP_SIZE


class _BEVActivationRecorder:
    def __init__(self, model: torch.nn.Module):
        try:
            reactive = model.Reactive_E2E
            modules = (
                reactive.FeatureFusion,
                reactive.NavigationEncoder,
                reactive.MapBEVFusion,
            )
        except AttributeError as exc:
            raise ValueError(
                "model does not expose the BEV diagnostic modules"
            ) from exc
        self._latest: dict[str, np.ndarray] = {}
        self._enabled = False
        self._handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in zip(BEV_HEATMAP_NAMES, modules)
        ]

    def _hook(self, name: str):
        def record(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self._enabled:
                return
            if not torch.is_tensor(output) or output.ndim != 4:
                raise ValueError(
                    f"{name} BEV output must be a [B,C,H,W] tensor"
                )
            channel_rms = torch.linalg.vector_norm(
                output.detach(), ord=2, dim=1
            ) / float(output.shape[1]) ** 0.5
            downsampled = F.adaptive_avg_pool2d(
                channel_rms[:, None].float(),
                (BEV_HEATMAP_SIZE, BEV_HEATMAP_SIZE),
            )[:, 0]
            self._latest[name] = downsampled.cpu().numpy()

        return record

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if enabled:
            self._latest.clear()

    def take(self, batch_size: int) -> np.ndarray:
        missing = set(BEV_HEATMAP_NAMES) - set(self._latest)
        if missing:
            raise RuntimeError(
                f"BEV diagnostic hooks did not run: {sorted(missing)}"
            )
        heatmaps = np.stack(
            [self._latest[name] for name in BEV_HEATMAP_NAMES],
            axis=1,
        )
        expected = (
            batch_size,
            len(BEV_HEATMAP_NAMES),
            BEV_HEATMAP_SIZE,
            BEV_HEATMAP_SIZE,
        )
        if heatmaps.shape != expected:
            raise RuntimeError(
                f"BEV diagnostic shape changed: {heatmaps.shape} != {expected}"
            )
        self._latest.clear()
        return np.ascontiguousarray(heatmaps, dtype=np.float32)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


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
    uids, controls, v0, seeds, _ = _infer_loader(
        model,
        loader,
        model_artifact_id=model_artifact_id,
        dataset_manifest_digest=dataset_manifest_digest,
        base_seeds=base_seeds,
        device=device,
        training_policy=training_policy,
        capture_bev_heatmaps=False,
    )
    return uids, controls, v0, seeds


def infer_loader_overlay(
    model: torch.nn.Module,
    loader: Any,
    *,
    model_artifact_id: str,
    dataset_manifest_digest: str,
    base_seeds: Sequence[int] = (0,),
    device: str | torch.device,
    training_policy: Any = None,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    np.ndarray,
]:
    """Infer controls and compact BEV activation maps for one shard."""
    uids, controls, v0, seeds, heatmaps = _infer_loader(
        model,
        loader,
        model_artifact_id=model_artifact_id,
        dataset_manifest_digest=dataset_manifest_digest,
        base_seeds=base_seeds,
        device=device,
        training_policy=training_policy,
        capture_bev_heatmaps=True,
    )
    assert heatmaps is not None
    return uids, controls, v0, seeds, heatmaps


def _infer_loader(
    model: torch.nn.Module,
    loader: Any,
    *,
    model_artifact_id: str,
    dataset_manifest_digest: str,
    base_seeds: Sequence[int],
    device: str | torch.device,
    training_policy: Any,
    capture_bev_heatmaps: bool,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    np.ndarray | None,
]:
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
    heatmap_batches: list[np.ndarray] = []
    recorder = _BEVActivationRecorder(model) if capture_bev_heatmaps else None
    try:
        for raw_batch in loader:
            sample_uids = [str(uid) for uid in raw_batch["sample_uid"]]
            batch = batch_to_device(raw_batch, device)
            if training_policy is not None:
                from training.dataset_policy import adapt_egomotion_history

                batch["egomotion_history"] = adapt_egomotion_history(
                    batch["egomotion_history"],
                    training_policy,
                )
            per_seed = []
            for seed_index, seed in enumerate(seeds):
                if recorder is not None:
                    recorder.set_enabled(seed_index == 0)
                per_seed.append(
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
                )
            controls = np.stack(per_seed, axis=1)
            history = raw_batch["egomotion_history"].reshape(
                len(sample_uids), 64, 4
            )
            speeds = history[:, -1, 0].detach().cpu().numpy().astype(np.float32)

            all_uids.extend(sample_uids)
            control_batches.append(controls)
            speed_batches.append(speeds)
            if recorder is not None:
                heatmap_batches.append(recorder.take(len(sample_uids)))
    finally:
        if recorder is not None:
            recorder.close()

    if not all_uids:
        raise ValueError("overlay loader yielded no samples")
    if len(set(all_uids)) != len(all_uids):
        raise ValueError("overlay loader yielded duplicate sample_uids")
    heatmaps = (
        np.concatenate(heatmap_batches, axis=0)
        if heatmap_batches
        else None
    )
    return (
        all_uids,
        np.concatenate(control_batches, axis=0),
        np.concatenate(speed_batches, axis=0),
        seeds,
        heatmaps,
    )
