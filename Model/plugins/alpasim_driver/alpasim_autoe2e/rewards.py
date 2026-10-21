"""Phase-1 closed-loop reward: deviation from the recorded drive.

The agreed starting point for closed-loop RL (#123, #177) is the imitation
objective turned into a reward: the further AutoE2E's plan is from what the
recorded driver did, the lower the reward. On top of it sits the renderer
validity bound. A reconstructed world is only reliable near the recorded path,
so once the ego itself is more than ``max_deviation_m`` from the recording,
the step is penalised and the episode should end. The bound is a property of
the renderer, not a driving preference, so it does not share the tracking
weight.

Per AutoE2E decision at time t0:

* ``plan_xy``: the 64 planned waypoints, rig frame at t0, 0.1 s apart;
* ``recorded_xy``: the recorded ego positions at the same timestamps, rig
  frame at t0 (shorter near the end of a clip);
* ``ego_deviation_m``: distance between the ego and the recorded ego at t0.

``tools/score_rollout.py`` builds these from an AlpaSim ``rollout.asl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    """One decision's reward and its parts."""

    reward: float
    tracking: float
    """``-(ade_weight * ADE + fde_weight * FDE)``, already scaled by ``tracking_weight``."""
    boundary: float
    """``-terminal_penalty`` when the ego left the valid region, else 0."""
    truncated: bool
    """True when the episode should end because the renderer is no longer reliable."""
    ade_m: float
    fde_m: float
    ego_deviation_m: float


def _as_xy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"expected [T, 2] positions, got shape {array.shape}")
    return array[:, :2]


class GroundTruthDeviationReward:
    """Negative displacement between the plan and the recorded drive, with the renderer bound."""

    def __init__(
        self,
        ade_weight: float = 1.0,
        fde_weight: float = 0.5,
        tracking_weight: float = 1.0,
        max_deviation_m: float = 3.0,
        terminal_penalty: float = 10.0,
    ) -> None:
        if max_deviation_m <= 0 or terminal_penalty < 0:
            raise ValueError("max_deviation_m must be positive and terminal_penalty non-negative")
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.tracking_weight = tracking_weight
        self.max_deviation_m = max_deviation_m
        self.terminal_penalty = terminal_penalty

    def compute(
        self,
        plan_xy: Any,
        recorded_xy: Any,
        ego_deviation_m: float,
    ) -> RewardResult:
        """Score one decision. Steps without a recorded position are left out."""
        plan = _as_xy(plan_xy)
        recorded = _as_xy(recorded_xy)
        steps = min(len(plan), len(recorded))
        if steps == 0:
            raise ValueError("the plan and the recorded drive must overlap in time")
        distances = np.linalg.norm(plan[:steps] - recorded[:steps], axis=1)
        ade, fde = float(distances.mean()), float(distances[-1])
        tracking = -self.tracking_weight * (self.ade_weight * ade + self.fde_weight * fde)

        truncated = bool(ego_deviation_m > self.max_deviation_m)
        boundary = -self.terminal_penalty if truncated else 0.0
        return RewardResult(
            reward=tracking + boundary,
            tracking=tracking,
            boundary=boundary,
            truncated=truncated,
            ade_m=ade,
            fde_m=fde,
            ego_deviation_m=float(ego_deviation_m),
        )


__all__ = ["GroundTruthDeviationReward", "RewardResult"]
