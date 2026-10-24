import pytest

from accuracy_benchmark.drive_jepa.kitscenes_route_commands import route_command


@pytest.mark.parametrize(
    "maneuver,expected",
    [
        ("left", [1.0, 0.0, 0.0, 0.0]),
        ("straight", [0.0, 1.0, 0.0, 0.0]),
        ("right", [0.0, 0.0, 1.0, 0.0]),
        ("unknown", [0.0, 0.0, 0.0, 1.0]),
        ("merge", [0.0, 0.0, 0.0, 1.0]),
        ("exit", [0.0, 0.0, 0.0, 1.0]),
        ("u_turn", [0.0, 0.0, 0.0, 1.0]),
        ("destination", [0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_published_maneuver_to_command(maneuver, expected):
    vector, _ = route_command(
        {"route_valid": True, "map_valid": True, "route_maneuver": maneuver}
    )
    assert vector == expected
    assert sum(vector) == 1


@pytest.mark.parametrize("field", ["route_valid", "map_valid"])
def test_invalid_route_or_map_is_unknown(field):
    metadata = {"route_valid": True, "map_valid": True, "route_maneuver": "left"}
    metadata[field] = False
    assert route_command(metadata) == ([0.0, 0.0, 0.0, 1.0], "unknown")


def test_commands_ignore_future_and_ego_motion_fields():
    metadata = {"route_valid": True, "map_valid": True, "route_maneuver": "right"}
    assert route_command(metadata) == route_command(
        {**metadata, "future_xy": [[-100, 100]], "velocity": -99}
    )


def test_unrecognized_labels_and_invalid_flags_fail():
    with pytest.raises(ValueError):
        route_command(
            {"route_valid": "false", "map_valid": True, "route_maneuver": "left"}
        )
    with pytest.raises(ValueError):
        route_command(
            {"route_valid": True, "map_valid": True, "route_maneuver": "invented"}
        )
