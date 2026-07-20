"""Reasoning-branch faithfulness check via intervention (#98).

Recent VLA benchmarks show a model's stated reasoning can be *decorative* rather
than causal — high observational alignment while interventions on the reasoning
leave the trajectory unchanged (VLADriveBench, arXiv:2606.12706). This module
measures the causal notion directly: run the same batch with and without the
reasoning branch's planner coupling and report how much the trajectory moves.

Because the coupling is zero-initialised, the delta is exactly 0.0 at
initialisation and only grows once training opens the gate — so this doubles as
a regression check that enabling reasoning does not perturb the reactive
baseline before training.

Uses the current AutoE2E forward ABI (projection / geometry_type /
image_transform), and compares against the EFFECTIVE context the planner uses:
the intervention bypasses the head inside ``ReactiveE2E`` (where reasoning runs,
after TemporalMemory), and the World Model's rolling buffer is snapshot/restored
so both runs see the same history.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _traj(out: Any) -> torch.Tensor:
    """Extract the trajectory tensor from a forward return (tuple or tensor)."""
    return out[0] if isinstance(out, tuple) else out


def _snapshot_buffer(model: torch.nn.Module):
    buffer = getattr(model, "visual_history_buffer", None)
    saved = list(buffer._buf) if buffer is not None else None

    def restore() -> None:
        if buffer is not None and saved is not None:
            buffer._buf = list(saved)

    return restore


def reasoning_intervention_delta(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_input: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    projection: Optional[Any] = None,
    geometry_type: Optional[str] = None,
    image_transform: Optional[Any] = None,
) -> dict[str, float]:
    """Mean trajectory L2 between the reasoning-coupled and bypassed runs.

    Args:
        model: an ``AutoE2E`` built with ``enable_reasoning=True``.
        camera_tiles / map_input / visual_history / egomotion_history: one batch.
        projection / geometry_type / image_transform: current geometry ABI.

    Returns:
        ``{"trajectory_l2": float}`` — 0.0 while the coupling gate is untrained.

    Raises:
        ValueError: if the model has no reasoning head to intervene on.
    """
    reactive = getattr(model, "Reactive_E2E", None)
    head = getattr(reactive, "ReasoningHead", None) if reactive is not None else None
    if head is None:
        raise ValueError(
            "reasoning_intervention_delta needs a model built with enable_reasoning=True."
        )
    assert reactive is not None  # implied by head is not None; narrows for mypy

    was_training = model.training
    model.eval()
    restore_buffer = _snapshot_buffer(model)
    fwd = dict(projection=projection, geometry_type=geometry_type,
               image_transform=image_transform, mode="infer")

    try:
        with torch.no_grad():
            coupled = _traj(model(camera_tiles, map_input, visual_history,
                                  egomotion_history, **fwd))
            restore_buffer()
            reactive.ReasoningHead = None  # bypass the branch (intervention)
            try:
                bypassed = _traj(model(camera_tiles, map_input, visual_history,
                                       egomotion_history, **fwd))
            finally:
                reactive.ReasoningHead = head
                restore_buffer()
    finally:
        if was_training:
            model.train()

    trajectory_l2 = torch.linalg.vector_norm(coupled - bypassed, dim=-1).mean()
    return {"trajectory_l2": float(trajectory_l2)}


def horizon_intervention_delta(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_input: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    intervention: str = "zero_all",
    projection: Optional[Any] = None,
    geometry_type: Optional[str] = None,
    image_transform: Optional[Any] = None,
) -> dict[str, float]:
    """Trajectory delta under a targeted intervention on the horizon tokens.

    Wraps the model's reasoning head so its ``horizon_tokens`` / ``reasoning_latent``
    are perturbed before the planner consumes them, then compares the trajectory
    against the un-perturbed run. Preserves *when* a hazard matters as a testable
    property (#98 §9.3).

    Args:
        intervention: one of ``"zero_all"`` (zero every horizon token),
            ``"zero_1s"`` / ``"zero_2s"`` (zero one horizon token),
            ``"shuffle"`` (reverse horizon order).
        others: as in :func:`reasoning_intervention_delta`.

    Returns:
        ``{"trajectory_l2": float}`` — the trajectory movement caused by the
        intervention (0.0 at init, since the coupling gate is zero).

    Raises:
        ValueError: if the model has no reasoning head, or the intervention is
            unknown.
    """
    reactive = getattr(model, "Reactive_E2E", None)
    head = getattr(reactive, "ReasoningHead", None) if reactive is not None else None
    if head is None:
        raise ValueError("horizon_intervention_delta needs enable_reasoning=True.")

    horizon_idx = {"zero_1s": 1, "zero_2s": 2, "zero_3s": 3, "zero_4s": 4}
    if intervention not in ({"zero_all", "shuffle"} | set(horizon_idx)):
        raise ValueError(f"unknown intervention {intervention!r}.")

    def _perturb(pred):
        tokens = pred.horizon_tokens
        if intervention == "zero_all":
            tokens = torch.zeros_like(tokens)
        elif intervention == "shuffle":
            tokens = tokens.flip(dims=(1,))
        else:
            tokens = tokens.clone()
            tokens[:, horizon_idx[intervention]] = 0.0
        pred.horizon_tokens = tokens
        return pred

    was_training = model.training
    model.eval()
    restore_buffer = _snapshot_buffer(model)
    fwd = dict(projection=projection, geometry_type=geometry_type,
               image_transform=image_transform, mode="infer")

    original_forward = head.forward

    def _wrapped(*args, **kwargs):
        return _perturb(original_forward(*args, **kwargs))

    try:
        with torch.no_grad():
            baseline = _traj(model(camera_tiles, map_input, visual_history,
                                   egomotion_history, **fwd))
            restore_buffer()
            head.forward = _wrapped  # type: ignore[method-assign]
            try:
                perturbed = _traj(model(camera_tiles, map_input, visual_history,
                                        egomotion_history, **fwd))
            finally:
                head.forward = original_forward  # type: ignore[method-assign]
                restore_buffer()
    finally:
        if was_training:
            model.train()

    trajectory_l2 = torch.linalg.vector_norm(baseline - perturbed, dim=-1).mean()
    return {"trajectory_l2": float(trajectory_l2)}
