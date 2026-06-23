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
    return PLANNER_REGISTRY[planner_mode](**kwargs)

__all__ = [
    "BasePlanner",
    "FlowMatchingPlanner",
    "BezierPlanner",
    "PLANNER_REGISTRY",
    "build_planner",
]
