from .base import BasePlanner
from .gru_planner import GRUPlanner
from .flow_matching_planner import FlowMatchingPlanner


PLANNER_REGISTRY = {
    "gru": GRUPlanner,
    "flow_matching": FlowMatchingPlanner,
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
    "GRUPlanner",
    "FlowMatchingPlanner",
    "PLANNER_REGISTRY",
    "build_planner",
]
