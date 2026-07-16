from .base import BasePlanner
from .flow_matching_planner import FlowMatchingPlanner
from .bezier_planner import BezierPlanner

PLANNER_REGISTRY = {
    "flow_matching": FlowMatchingPlanner,
    "bezier": BezierPlanner,
}

def build_planner(planner_mode, **kwargs):
    """Construct a planner by name.

    Forwards ``**kwargs`` directly to the planner __init__; planners ignore
    args they do not understand by accepting only what they need (the
    registry does not pre-filter, so callers must pass kwargs the chosen
    planner accepts).
    """
    if planner_mode not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner_mode {planner_mode!r}. "
            f"Available: {sorted(PLANNER_REGISTRY)}."
        )
    if planner_mode == "flow_matching":
        # The training loop (train_il) regresses forward()'s output with SmoothL1,
        # but forward() Euler-integrates from a fresh noise sample every step — so
        # that is NOT the flow-matching objective (velocity MSE against x1-x0) and
        # drives the model to the conditional mean. The proper objective lives in
        # compute_planner_loss, which is not wired into the train loop yet. Warn
        # loudly so nobody trains this expecting real flow matching.
        import warnings
        warnings.warn(
            "planner_mode='flow_matching' is NOT correctly trainable via the "
            "current train_il loop (it L1-regresses an Euler-from-noise rollout, "
            "not the velocity-MSE flow objective). Use 'bezier' unless/until "
            "compute_planner_loss is wired in.",
            RuntimeWarning, stacklevel=2,
        )
    return PLANNER_REGISTRY[planner_mode](**kwargs)

__all__ = [
    "BasePlanner",
    "FlowMatchingPlanner",
    "BezierPlanner",
    "PLANNER_REGISTRY",
    "build_planner",
]
