"""Training loss modules for AutoE2E (kept outside the model per Zain's criterion)."""

from .horizon_reasoning_loss import HorizonReasoningLoss
from .control_rollout import (
    ROLLOUT_POLICY_VERSION,
    integrate_controls_torch,
)
from .rollout_aligned_loss import (
    ROLLOUT_ALIGNED_LOSS_VERSION,
    RolloutAlignedLoss,
)
from .route_consistency_loss import (
    RouteConsistencyLoss,
    RouteConsistencyWeights,
    ego_points_to_grid,
)

__all__ = [
    "HorizonReasoningLoss",
    "ROLLOUT_ALIGNED_LOSS_VERSION",
    "ROLLOUT_POLICY_VERSION",
    "RouteConsistencyLoss",
    "RouteConsistencyWeights",
    "RolloutAlignedLoss",
    "ego_points_to_grid",
    "integrate_controls_torch",
]
