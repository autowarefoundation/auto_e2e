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
from .world_model_quality import (
    jepa_reconstruction_metrics,
    null_predictor_metrics,
    open_loop_pair_metrics,
    relative_jepa_improvement,
    summarize_world_model_quality,
    trajectory_impact_metrics,
    world_model_trajectory_impact,
)

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
    # world-model quality (JEPA recon + Reactive vs Combined)
    "jepa_reconstruction_metrics",
    "null_predictor_metrics",
    "relative_jepa_improvement",
    "trajectory_impact_metrics",
    "open_loop_pair_metrics",
    "world_model_trajectory_impact",
    "summarize_world_model_quality",
]
