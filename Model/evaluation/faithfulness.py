"""Reasoning-band faithfulness check via intervention (#98/#103).

Recent VLA benchmarks show that a model's stated reasoning can be *decorative*
rather than causal — high observational alignment while interventions on the
reasoning leave the trajectory unchanged (VLADriveBench, arXiv:2606.12706).
This module measures the opposite, causal notion directly in our stack: run
the same batch **with and without the reasoning band's planner coupling** and
report how much the trajectory actually moves.

Because the band's gate is zero-initialised (no-op), the delta is exactly 0.0
at initialisation and only becomes positive once training pushes the gate away
from zero — so this doubles as a regression check that enabling the band does
not perturb the reactive baseline before training.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


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
    """Measure how much the reasoning band's coupling moves the trajectory.

    Runs ``model`` twice in ``mode="infer"`` on the same inputs: once as-is
    (reasoning band active) and once with the band bypassed (intervention),
    then compares the predicted trajectories.

    Args:
        model: an ``AutoE2E`` instance with ``enable_reasoning_band=True``.
        camera_tiles / map_input / visual_history / egomotion_history: one
            evaluation batch, as in ``AutoE2E.forward``.
        projection / geometry_type / image_transform: the current geometry ABI
            forwarded to ``AutoE2E.forward`` (replaces the old ``camera_params``
            argument).

    Returns:
        dict with:
        * ``trajectory_l2``: mean L2 distance between the coupled and
          intervened trajectories (0.0 while the gate is untrained).
        * ``history_shift``: mean L2 between the modulated and the **effective**
          visual history the band actually receives inside the model (with the
          World Model on, that is the WAM-aggregated history, not the raw
          caller input) — how hard the gate is steering the planner input.

    Raises:
        ValueError: if the model has no reasoning band to intervene on.
    """
    band = getattr(model, "Reasoning_Band", None)
    if band is None:
        raise ValueError(
            "reasoning_intervention_delta needs a model built with "
            "enable_reasoning_band=True."
        )

    was_training = model.training
    model.eval()

    # The World Model's rolling buffer is per-sequence state that every
    # forward PUSHES to — without snapshot/restore the coupled and intervened
    # runs would see different histories (non-zero delta even with an
    # untrained gate) and the caller's rollout state would be advanced.
    buffer = getattr(model, "visual_history_buffer", None)
    saved_frames = list(buffer._buf) if buffer is not None else None

    def _restore_buffer() -> None:
        if buffer is not None and saved_frames is not None:
            buffer._buf = list(saved_frames)

    fwd_kwargs = dict(
        projection=projection,
        geometry_type=geometry_type,
        image_transform=image_transform,
        mode="infer",
    )

    # Capture the EFFECTIVE visual history the band receives inside the model.
    # With the World Model enabled, AutoE2E replaces the caller's
    # ``visual_history`` with the WAM-aggregated history before the band runs;
    # measuring history_shift against the raw caller input would be wrong.
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: torch.nn.Module, inputs: Any, output: Any) -> None:
        captured["effective"] = inputs[0].detach()
        captured["modulated"] = output.modulated_visual_history.detach()

    handle = band.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            coupled = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                **fwd_kwargs,
            )
            _restore_buffer()
    finally:
        handle.remove()

    try:
        with torch.no_grad():
            # Intervention: bypass the band entirely (planner sees the effective
            # visual history unmodulated), then restore it.  setattr keeps mypy
            # happy about temporarily nulling an nn.Module attribute.
            setattr(model, "Reasoning_Band", None)
            intervened = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                **fwd_kwargs,
            )
    finally:
        model.Reasoning_Band = band
        _restore_buffer()
        if was_training:
            model.train()

    if "modulated" not in captured:
        raise RuntimeError(
            "the reasoning band did not run during the coupled forward; "
            "cannot compute history_shift."
        )

    coupled_traj = coupled[0] if isinstance(coupled, tuple) else coupled
    intervened_traj = intervened[0] if isinstance(intervened, tuple) else intervened

    trajectory_l2 = torch.linalg.vector_norm(
        coupled_traj - intervened_traj, dim=-1
    ).mean()
    history_shift = torch.linalg.vector_norm(
        captured["modulated"] - captured["effective"], dim=-1
    ).mean()

    return {
        "trajectory_l2": float(trajectory_l2),
        "history_shift": float(history_shift),
    }
