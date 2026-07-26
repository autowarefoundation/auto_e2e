"""Tests for logged-XY rollout validation records."""

from __future__ import annotations

import numpy as np
import torch

from evaluation.rollout_validation import (
    build_rollout_validation_records,
)
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from training.losses.control_rollout import integrate_controls_torch


GEOMETRY = DEFAULT_NAVIGATION_GEOMETRY


def _controls(curvature: float = 0.0) -> torch.Tensor:
    controls = torch.zeros(1, 64, 2, dtype=torch.float32)
    controls[:, :, 1] = curvature
    return controls


def _band_field(half_width_m: float = 3.0) -> torch.Tensor:
    _, y_left = GEOMETRY.pixel_center_grids()
    return torch.from_numpy(
        np.maximum(np.abs(y_left) - half_width_m, 0.0).astype(
            np.float32
        )
    ).unsqueeze(0)


def _supervision(
    field: torch.Tensor,
    *,
    destination_visible: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        "distance_to_corridor_m": field,
        "distance_to_drivable_m": field,
        "destination_xy_m": torch.tensor([[32.0, 0.0]]),
        "destination_visible": torch.tensor([destination_visible]),
        "available": torch.tensor([True]),
    }


def _logged_straight() -> torch.Tensor:
    positions, _, _ = integrate_controls_torch(
        _controls(),
        torch.tensor([5.0]),
    )
    return positions


def test_logged_xy_equal_prediction_has_zero_selector_metrics():
    controls = _controls()

    records = build_rollout_validation_records(
        controls,
        controls,
        torch.tensor([5.0]),
        _logged_straight(),
        _supervision(torch.zeros(1, 256, 256)),
        torch.tensor([True]),
        torch.tensor([True]),
        ["sample-a"],
        ["scene-a"],
    )

    assert records == [{
        "sample_uid": "sample-a",
        "split_group_uid": "scene-a",
        "ade_3s_m": 0.0,
        "fde_6_4s_m": 0.0,
        "comfort_excess": 0.0,
        "offroad_excess": 0.0,
        "route_gap": 0.0,
        "wrong_branch_excess": None,
        "destination_error_m": None,
        "diagnostic_predicted_offroad_rate": 0.0,
        "diagnostic_target_offroad_rate": 0.0,
        "diagnostic_predicted_route_compliance": 1.0,
        "diagnostic_target_route_compliance": 1.0,
        "diagnostic_raster_tolerance_m": 0.5,
    }]


def test_logged_xy_metrics_detect_map_route_and_destination_regression():
    field = _band_field()

    records = build_rollout_validation_records(
        _controls(curvature=0.04),
        _controls(),
        torch.tensor([5.0]),
        _logged_straight(),
        _supervision(field, destination_visible=True),
        torch.tensor([True]),
        torch.tensor([True]),
        ["sample-a"],
        ["scene-a"],
        route_intersections=[True],
    )
    record = records[0]

    assert record["ade_3s_m"] > 0.0
    assert record["fde_6_4s_m"] > record["ade_3s_m"]
    assert record["offroad_excess"] > 0.0
    assert record["route_gap"] > 0.0
    assert record["wrong_branch_excess"] == 1.0
    assert record["destination_error_m"] > 0.0


def test_invalid_map_and_route_are_unavailable_not_perfect():
    controls = _controls(curvature=0.04)

    record = build_rollout_validation_records(
        controls,
        _controls(),
        torch.tensor([5.0]),
        _logged_straight(),
        _supervision(_band_field()),
        torch.tensor([False]),
        torch.tensor([False]),
        ["sample-a"],
        ["scene-a"],
    )[0]

    assert record["offroad_excess"] is None
    assert record["route_gap"] is None
    assert record["wrong_branch_excess"] is None
    assert record["destination_error_m"] is None


def test_inside_metrics_use_half_pixel_raster_tolerance():
    controls = _controls()
    shape = (1, GEOMETRY.height_px, GEOMETRY.width_px)

    inside = build_rollout_validation_records(
        controls,
        controls,
        torch.tensor([5.0]),
        _logged_straight(),
        _supervision(torch.full(shape, 0.49)),
        torch.tensor([True]),
        torch.tensor([True]),
        ["sample-a"],
        ["scene-a"],
    )[0]
    outside = build_rollout_validation_records(
        controls,
        controls,
        torch.tensor([5.0]),
        _logged_straight(),
        _supervision(torch.full(shape, 0.51)),
        torch.tensor([True]),
        torch.tensor([True]),
        ["sample-a"],
        ["scene-a"],
    )[0]

    assert inside["diagnostic_target_offroad_rate"] == 0.0
    assert inside["diagnostic_target_route_compliance"] == 1.0
    assert outside["diagnostic_target_offroad_rate"] == 1.0
    assert outside["diagnostic_target_route_compliance"] == 0.0
