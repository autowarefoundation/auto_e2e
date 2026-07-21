from .metrics import (
    COMFORT_THRESHOLDS,
    compute_comfort_metrics,
    compute_open_loop_metrics,
    gate_check,
    integrate_trajectory,
    offroad_rate,
)
from .baselines import constant_velocity_baseline, hold_last_action_baseline
from .splits import episode_range_split, geographic_holdout_split, long_tail_split
from .faithfulness import horizon_intervention_delta, reasoning_intervention_delta

__all__ = [
    # existing (open-loop displacement metrics + gate)
    "compute_open_loop_metrics",
    "gate_check",
    "integrate_trajectory",
    # complementary: comfort + off-road (#66 §2-3)
    "compute_comfort_metrics",
    "COMFORT_THRESHOLDS",
    "offroad_rate",
    # training-free baselines (#66 §5)
    "constant_velocity_baseline",
    "hold_last_action_baseline",
    # validation splits (#66 §4)
    "episode_range_split",
    "geographic_holdout_split",
    # reasoning branch evaluation (#98)
    "long_tail_split",
    "reasoning_intervention_delta",
    "horizon_intervention_delta",
]
