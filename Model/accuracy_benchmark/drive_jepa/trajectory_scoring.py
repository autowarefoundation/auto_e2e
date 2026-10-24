"""AutoE2E Reactive displacement aggregation policy for external evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


DISPLACEMENT_HORIZON_STEPS = {
    "1s": 10,
    "2s": 20,
    "3s": 30,
    "4s": 40,
    "5s": 50,
    "6p4s": 64,
}


def displacement_horizon_totals(
    errors: torch.Tensor,
    valid: torch.Tensor,
    horizon_steps: Mapping[str, int],
) -> dict[str, tuple[float, int, float, int]]:
    """Return ADE sum/count and FDE sum/count using the Reactive mask policy.

    ADE gives each sample equal weight over its valid points. FDE requires
    the endpoint to be valid. Prediction support must be checked by the caller;
    a shorter prediction is never permission to label a longer-horizon ADE.
    """
    totals = {}
    for name, step_count in horizon_steps.items():
        usable = min(step_count, errors.shape[1])
        horizon_valid = valid[:, :usable]
        per_sample_count = horizon_valid.sum(dim=1)
        eligible_ade = per_sample_count > 0
        per_sample_error = (errors[:, :usable] * horizon_valid).sum(
            dim=1
        ) / per_sample_count.clamp_min(1)
        ade_sum = float(per_sample_error[eligible_ade].sum().item())
        ade_count = int(eligible_ade.sum().item())
        fde_sum, fde_count = 0.0, 0
        if step_count <= errors.shape[1]:
            eligible_fde = valid[:, step_count - 1]
            fde_sum = float(errors[eligible_fde, step_count - 1].sum().item())
            fde_count = int(eligible_fde.sum().item())
        totals[name] = (ade_sum, ade_count, fde_sum, fde_count)
    return totals
