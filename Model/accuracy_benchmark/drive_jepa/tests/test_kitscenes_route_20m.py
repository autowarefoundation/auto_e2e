from types import SimpleNamespace
import numpy as np
import pytest
from accuracy_benchmark.drive_jepa.kitscenes_route_20m import (
    command_from_route,
    UNKNOWN,
)


def inputs(points):
    route = SimpleNamespace(
        valid=True,
        lane_sequence=[
            SimpleNamespace(
                lane_id="lane", centerline_enu_m=np.array(points, dtype=float)
            )
        ],
    )
    metadata = {
        "route_valid": True,
        "map_valid": True,
        "current_route_lane_id": "lane",
        "sample_pose_enu": {"x_enu_m": 0, "y_enu_m": 0, "yaw_rad": 0},
    }
    return route, metadata


@pytest.mark.parametrize(
    "points,slot",
    [
        ([[0, 0], [40, 0]], 1),
        ([[0, 0], [10, 0], [10, 30]], 0),
        ([[0, 0], [10, 0], [10, -30]], 2),
        ([[0, 0], [30, 0], [30, 40]], 1),
    ],
)
def test_direction_and_distant_turn(points, slot):
    command, detail = command_from_route(*inputs(points))
    assert command[slot] == 1 and sum(command) == 1
    assert detail["reason"] == "available"


def test_arc_length_not_forward_coordinate():
    command, detail = command_from_route(*inputs([[0, 0], [10, 0], [10, 30]]))
    assert detail["goal_enu_m"] == [10, 10]


def test_current_projection_and_ego_rotation():
    route, metadata = inputs([[0, 0], [0, 60]])
    metadata["sample_pose_enu"] = {"x_enu_m": 0, "y_enu_m": 15, "yaw_rad": np.pi / 2}
    command, detail = command_from_route(route, metadata)
    assert command == [0, 1, 0, 0]
    assert detail["goal_enu_m"] == [0, 35]


def test_short_invalid_and_missing_lane_are_unknown():
    route, metadata = inputs([[0, 0], [10, 0]])
    assert command_from_route(route, metadata)[0] == UNKNOWN
    metadata["current_route_lane_id"] = "absent"
    assert command_from_route(route, metadata)[0] == UNKNOWN
    metadata["route_valid"] = False
    assert command_from_route(route, metadata)[0] == UNKNOWN


def test_future_targets_do_not_affect_command():
    route, metadata = inputs([[0, 0], [40, 0]])
    before = command_from_route(route, metadata)
    metadata["future_trajectory"] = [[0, 100], [-100, 0]]
    assert command_from_route(route, metadata) == before


def test_discontinuous_route_is_unknown():
    route, metadata = inputs([[0, 0], [10, 0]])
    route.lane_sequence.append(
        SimpleNamespace(lane_id="next", centerline_enu_m=np.array([[15, 0], [40, 0]]))
    )
    assert command_from_route(route, metadata)[1]["reason"] == "discontinuous_route"


def test_gap_beyond_lookahead_does_not_change_command():
    route, metadata = inputs([[0, 0], [40, 0]])
    route.lane_sequence.append(
        SimpleNamespace(lane_id="next", centerline_enu_m=np.array([[50, 0], [70, 0]]))
    )
    assert command_from_route(route, metadata)[0] == [0, 1, 0, 0]
