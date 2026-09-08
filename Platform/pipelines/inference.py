"""Flyte-free policy inference helpers for deterministic trajectory overlays."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


NOISE_POLICY_VERSION = "v1"
INFERENCE_CONTRACT_VERSION = "v4"


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    """Return a checkpoint's SHA-256 without buffering the whole file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Filter checkpoint config to arguments accepted by the current AutoE2E."""
    from model_components.auto_e2e import AutoE2E

    valid = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
    return {key: value for key, value in config.items() if key in valid}


def load_policy(
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Load an AutoE2E checkpoint and return its content-addressed identity."""
    from model_components.auto_e2e import AutoE2E

    checkpoint_path = Path(checkpoint_path)
    artifact_id = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint["config"])
    model = AutoE2E(**model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, artifact_id


def stable_seed64(
    model_artifact_id: str,
    dataset_manifest_digest: str,
    sample_uid: str,
    base_seed: int,
) -> int:
    """Derive a stable unsigned 64-bit seed from overlay identity fields."""
    digest = hashlib.blake2b(
        digest_size=8,
        person=b"ae2e-noise-v1",
    )
    for value in (
        model_artifact_id,
        dataset_manifest_digest,
        sample_uid,
        str(base_seed),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little", signed=False)


def noise_from(
    model_artifact_id: str,
    dataset_manifest_digest: str,
    sample_uid: str,
    base_seed: int,
    shape: Sequence[int],
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate one sample's batch-independent initial noise tensor."""
    generator = torch.Generator(device=device)
    generator.manual_seed(
        stable_seed64(
            model_artifact_id,
            dataset_manifest_digest,
            sample_uid,
            base_seed,
        )
    )
    return torch.randn(
        tuple(shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _planner(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.Reactive_E2E.TrajectoryPlanner
    except AttributeError as exc:
        raise ValueError(
            "model does not expose Reactive_E2E.TrajectoryPlanner"
        ) from exc


def create_stateful_camera_fpn_cache(model: torch.nn.Module) -> Any:
    """Create one explicit per-stream BEVFormer T8 inference cache."""
    factory = getattr(model, "create_stateful_camera_fpn_cache", None)
    if not callable(factory):
        raise ValueError(
            "model does not support a stateful camera FPN cache"
        )
    return factory()


def predict_control(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    sample_uids: Sequence[str],
    model_artifact_id: str,
    dataset_manifest_digest: str,
    base_seed: int = 0,
    projection: Any = None,
    geometry_type: str = "pseudo",
    camera_fpn_cache: Any = None,
    camera_fpn_stream_ids: Sequence[str] | None = None,
    camera_fpn_timestamps_us: Any = None,
) -> np.ndarray:
    """Return raw ``[B,T,S]`` acceleration/curvature controls for one batch.

    Stateful callers must provide stable stream IDs across consecutive calls.
    Changing an ID marks a scene transition and resets that cache lane.
    Source timestamps must also be explicit and exactly 500 ms apart. Do not
    synthesize timestamps across dropped source frames.
    """
    model.eval()
    planner = _planner(model)
    num_timesteps = int(planner.num_timesteps)
    num_signals = int(planner.num_signals)
    trajectory_dim = num_timesteps * num_signals

    visual = batch["visual_tiles"]
    batch_size = int(visual.shape[0])
    if len(sample_uids) != batch_size:
        raise ValueError(
            f"sample_uids length must equal batch size {batch_size}, "
            f"got {len(sample_uids)}"
        )

    device = visual.device
    from training.reactive_stage_runner import (
        resolve_reactive_batch_projection,
        resolve_reactive_camera_history,
        resolve_reactive_front_projection,
    )

    projection, geometry_type = resolve_reactive_batch_projection(
        batch,
        projection,
        geometry_type,
        device=device,
    )
    front_projection = resolve_reactive_front_projection(
        batch,
        geometry_type,
        device=device,
    )
    camera_history_tiles, history_projections = (
        resolve_reactive_camera_history(
            batch,
            geometry_type,
            device=device,
        )
    )
    initial_noise = torch.stack(
        [
            noise_from(
                model_artifact_id,
                dataset_manifest_digest,
                uid,
                base_seed,
                (trajectory_dim,),
                device,
                visual.dtype,
            )
            for uid in sample_uids
        ]
    )

    if hasattr(model, "reset_visual_history"):
        model.reset_visual_history()

    front_camera_fpn_tile = None
    front_camera_fpn_available = None
    if camera_fpn_cache is not None:
        if camera_fpn_stream_ids is None:
            raise ValueError(
                "stateful camera FPN cache requires explicit stream IDs"
            )
        if isinstance(camera_fpn_stream_ids, str):
            normalized_stream_ids = (camera_fpn_stream_ids,)
        else:
            normalized_stream_ids = tuple(camera_fpn_stream_ids)
        if (
            len(normalized_stream_ids) != batch_size
            or any(
                not isinstance(value, str) or not value
                for value in normalized_stream_ids
            )
        ):
            raise ValueError(
                "stateful camera FPN cache requires one explicit stream ID "
                "per sample"
            )
        camera_fpn_stream_ids = normalized_stream_ids
        if camera_fpn_timestamps_us is None:
            raise ValueError(
                "stateful camera FPN cache requires explicit source timestamps"
            )
        front_camera_fpn_tile = batch.get("front_camera_fpn_tile")
        front_camera_fpn_available = batch.get(
            "front_camera_fpn_available"
        )
        front_camera_tile = batch.get("front_camera_tile")
        if (front_camera_tile is None) != (
            front_camera_fpn_tile is None
        ):
            raise ValueError(
                "stateful camera FPN cache requires native and exact "
                "base-resolution Front tiles together"
            )
        if front_camera_tile is not None:
            if front_camera_fpn_available is None:
                raise ValueError(
                    "stateful camera FPN cache requires Front companion "
                    "availability"
                )
            available = torch.as_tensor(
                front_camera_fpn_available,
                dtype=torch.bool,
            ).reshape(-1)
            if available.numel() != batch_size or not bool(
                available.all().item()
            ):
                raise ValueError(
                    "stateful camera FPN cache requires an exact Front "
                    "companion for every sample"
                )

    kwargs: dict[str, Any] = {
        "projection": projection,
        "geometry_type": geometry_type,
        "camera_history_tiles": camera_history_tiles,
        "history_projections": history_projections,
        "camera_fpn_cache": camera_fpn_cache,
        "camera_fpn_stream_ids": camera_fpn_stream_ids,
        "camera_fpn_timestamps_us": camera_fpn_timestamps_us,
        "front_camera_tile": batch.get("front_camera_tile"),
        "front_camera_fpn_tile": front_camera_fpn_tile,
        "front_camera_fpn_available": front_camera_fpn_available,
        "front_projection": front_projection,
        "mode": "infer",
        "initial_noise": initial_noise,
    }
    if batch.get("history_frames") is not None:
        kwargs["history_frames"] = batch["history_frames"]
    if batch.get("future_frames") is not None:
        kwargs["future_frames"] = batch["future_frames"]

    with torch.no_grad():
        prediction = model(
            visual,
            batch["map_context"],
            batch["visual_history"],
            batch["egomotion_history"],
            route_mask=batch["route_mask"],
            map_valid=batch["map_valid"],
            route_valid=batch["route_valid"],
            **kwargs,
        )
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    expected = (batch_size, trajectory_dim)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            f"policy returned shape {tuple(prediction.shape)}, expected {expected}"
        )
    return (
        prediction.reshape(batch_size, num_timesteps, num_signals)
        .detach()
        .cpu()
        .numpy()
    )
