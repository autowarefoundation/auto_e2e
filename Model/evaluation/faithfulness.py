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

from typing import Optional

import torch


def reasoning_intervention_delta(
    model: torch.nn.Module,
    camera_tiles: torch.Tensor,
    map_input: torch.Tensor,
    visual_history: torch.Tensor,
    egomotion_history: torch.Tensor,
    camera_params: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Measure how much the reasoning band's coupling moves the trajectory.

    Runs ``model`` twice in ``mode="infer"`` on the same inputs: once as-is
    (reasoning band active) and once with the band bypassed (intervention),
    then compares the predicted trajectories.

    Args:
        model: an ``AutoE2E`` instance with ``enable_reasoning_band=True``.
        camera_tiles / map_input / visual_history / egomotion_history /
            camera_params: one evaluation batch, as in ``AutoE2E.forward``.

    Returns:
        dict with:
        * ``trajectory_l2``: mean L2 distance between the coupled and
          intervened trajectories (0.0 while the gate is untrained).
        * ``history_shift``: mean L2 between the modulated and raw visual
          history (how hard the gate is actually steering the planner input).

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

    try:
        with torch.no_grad():
            coupled = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                camera_params=camera_params, mode="infer",
            )
            _restore_buffer()
            pred = band(visual_history, mode="infer")
            # Intervention: bypass the band entirely (planner sees the raw
            # visual history), then restore it.  setattr keeps mypy happy
            # about temporarily nulling an nn.Module attribute.
            setattr(model, "Reasoning_Band", None)
            intervened = model(
                camera_tiles, map_input, visual_history, egomotion_history,
                camera_params=camera_params, mode="infer",
            )
    finally:
        model.Reasoning_Band = band
        _restore_buffer()
        if was_training:
            model.train()

    coupled_traj = coupled[0] if isinstance(coupled, tuple) else coupled
    intervened_traj = intervened[0] if isinstance(intervened, tuple) else intervened

    trajectory_l2 = torch.linalg.vector_norm(
        coupled_traj - intervened_traj, dim=-1
    ).mean()
    history_shift = torch.linalg.vector_norm(
        pred.modulated_visual_history - visual_history, dim=-1
    ).mean()

    return {
        "trajectory_l2": float(trajectory_l2),
        "history_shift": float(history_shift),
    }
