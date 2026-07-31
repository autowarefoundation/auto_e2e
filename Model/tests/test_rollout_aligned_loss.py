"""Tests for rollout-aligned planner loss terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from training.losses.control_rollout import integrate_controls_torch
from training.losses.rollout_aligned_loss import (
    RolloutAlignedLoss,
    _footprint_outside_distance,
)


TIMESTEPS = 64
GEOMETRY = DEFAULT_NAVIGATION_GEOMETRY


def _controls(
    *,
    acceleration: float = 0.0,
    curvature: float = 0.0,
    requires_grad: bool = False,
) -> torch.Tensor:
    value = torch.zeros(1, TIMESTEPS, 2, dtype=torch.float32)
    value[:, :, 0] = acceleration
    value[:, :, 1] = curvature
    value.requires_grad_(requires_grad)
    return value


def _field_from_lateral_band(half_width_m: float = 3.0) -> torch.Tensor:
    _, y_left = GEOMETRY.pixel_center_grids()
    outside = np.maximum(np.abs(y_left) - half_width_m, 0.0)
    return torch.from_numpy(outside.astype(np.float32)).unsqueeze(0)


def _supervision(
    *,
    route_field: torch.Tensor | None = None,
    drivable_field: torch.Tensor | None = None,
    available: bool = True,
) -> dict[str, torch.Tensor]:
    shape = (1, GEOMETRY.height_px, GEOMETRY.width_px)
    zeros = torch.zeros(shape, dtype=torch.float32)
    return {
        "distance_to_corridor_m": (
            route_field if route_field is not None else zeros.clone()
        ),
        "distance_to_drivable_m": (
            drivable_field
            if drivable_field is not None
            else zeros.clone()
        ),
        "available": torch.tensor([available], dtype=torch.bool),
        "drivable_available": torch.tensor(
            [available],
            dtype=torch.bool,
        ),
    }


def _loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    logged_positions: torch.Tensor | None = None,
    supervision: dict[str, torch.Tensor] | None = None,
    map_valid: bool = True,
    route_valid: bool = True,
) -> dict[str, torch.Tensor]:
    initial_speed = torch.tensor([5.0], dtype=torch.float32)
    if logged_positions is None:
        logged_positions, _, _ = integrate_controls_torch(
            target,
            initial_speed,
        )
    return RolloutAlignedLoss()(
        predicted,
        target,
        initial_speed,
        logged_positions,
        supervision or _supervision(),
        torch.tensor([map_valid], dtype=torch.bool),
        torch.tensor([route_valid], dtype=torch.bool),
    )


def test_prediction_equal_target_has_zero_losses():
    target = _controls(acceleration=0.1, curvature=0.01)

    terms = _loss(target.clone(), target)

    for name in (
        "rollout",
        "path",
        "final",
        "constraint",
        "comfort",
        "map",
        "route",
        "drivable",
    ):
        assert terms[name].item() == 0.0


def test_empty_batch_is_rejected_with_clear_error():
    controls = torch.empty(0, TIMESTEPS, 2)
    field = torch.empty(
        0,
        GEOMETRY.height_px,
        GEOMETRY.width_px,
    )

    with pytest.raises(
        ValueError,
        match="requires a non-empty batch",
    ):
        RolloutAlignedLoss()(
            controls,
            controls,
            torch.empty(0),
            torch.empty(0, TIMESTEPS, 2),
            {
                "distance_to_corridor_m": field,
                "distance_to_drivable_m": field,
                "available": torch.empty(0, dtype=torch.bool),
                "drivable_available": torch.empty(
                    0,
                    dtype=torch.bool,
                ),
            },
            torch.empty(0, dtype=torch.bool),
            torch.empty(0, dtype=torch.bool),
        )


def test_equal_footprints_have_zero_map_loss_on_nontrivial_field():
    target = _controls(acceleration=0.1, curvature=0.01)
    field = _field_from_lateral_band()

    terms = _loss(
        target.clone(),
        target,
        supervision=_supervision(
            route_field=field,
            drivable_field=field,
        ),
    )

    assert terms["map"].item() == 0.0
    assert terms["route"].item() == 0.0
    assert terms["drivable"].item() == 0.0


def test_footprint_corners_detect_violation_missed_by_center():
    field = _field_from_lateral_band(half_width_m=0.5)
    positions = torch.zeros(1, 1, 2)
    headings = torch.zeros(1, 1)

    distance = _footprint_outside_distance(
        field,
        positions,
        headings,
        GEOMETRY,
        length_m=4.8,
        width_m=2.0,
    )

    assert distance.item() > 0.0


def test_out_of_raster_footprint_distance_is_differentiable():
    field = torch.zeros(
        1,
        GEOMETRY.height_px,
        GEOMETRY.width_px,
    )
    positions = torch.tensor(
        [[[GEOMETRY.x_max_m + 1.0, 0.0]]],
        requires_grad=True,
    )

    distance = _footprint_outside_distance(
        field,
        positions,
        torch.zeros(1, 1),
        GEOMETRY,
        length_m=4.8,
        width_m=2.0,
    )
    distance.sum().backward()

    assert distance.item() > 0.0
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert positions.grad[0, 0, 0].item() > 0.0


def test_rollout_position_target_is_logged_xy_not_target_controls():
    target = _controls(acceleration=0.1, curvature=0.01)
    logged_positions, _, _ = integrate_controls_torch(
        target,
        torch.tensor([5.0], dtype=torch.float32),
    )
    logged_positions = logged_positions.clone()
    logged_positions[:, :, 1] += 2.0

    terms = _loss(
        target.clone(),
        target,
        logged_positions=logged_positions,
        map_valid=False,
        route_valid=False,
    )

    assert terms["rollout"].item() > 0.0
    assert terms["comfort"].item() == 0.0


def test_non_finite_logged_xy_is_rejected():
    logged_positions = torch.zeros(1, TIMESTEPS, 2)
    logged_positions[:, -1, 0] = torch.nan

    with pytest.raises(
        ValueError,
        match="logged_positions must be finite",
    ):
        _loss(
            _controls(),
            _controls(),
            logged_positions=logged_positions,
        )


def test_rollout_loss_reaches_acceleration_and_curvature():
    predicted = _controls(requires_grad=True)
    target = _controls(acceleration=0.2, curvature=0.02)

    terms = _loss(
        predicted,
        target,
        map_valid=False,
        route_valid=False,
    )
    terms["rollout"].backward()

    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[:, :, 0].abs().sum().item() > 0.0
    assert predicted.grad[:, :, 1].abs().sum().item() > 0.0


def test_comfort_ignores_peak_timing_shift():
    predicted = _controls()
    target = _controls()
    predicted[:, 20:, 0] = 1.0
    target[:, 10:, 0] = 1.0

    terms = _loss(
        predicted,
        target,
        map_valid=False,
        route_valid=False,
    )

    assert terms["comfort"].item() == 0.0
    assert terms["jerk"].item() == 0.0


def test_comfort_penalizes_larger_prediction_peak():
    predicted = _controls()
    target = _controls()
    predicted[:, 20:, 0] = 2.0
    target[:, 10:, 0] = 1.0

    terms = _loss(
        predicted,
        target,
        map_valid=False,
        route_valid=False,
    )

    assert terms["comfort"].item() > 0.0
    assert terms["jerk"].item() > 0.0


def test_comfort_uses_float32_under_autocast():
    predicted = _controls().to(dtype=torch.bfloat16)
    target = _controls().to(dtype=torch.bfloat16)
    predicted[:, 20:, 0] = 2.0
    target[:, 10:, 0] = 1.0

    with torch.autocast("cpu", dtype=torch.bfloat16):
        terms = _loss(
            predicted,
            target,
            map_valid=False,
            route_valid=False,
        )

    assert terms["comfort"].dtype == torch.float32
    assert terms["jerk"].dtype == torch.float32
    assert terms["comfort"].item() > 0.0


def test_map_loss_increases_when_footprint_leaves_target_band():
    field = _field_from_lateral_band()
    predicted = _controls(curvature=0.04, requires_grad=True)
    target = _controls()

    terms = _loss(
        predicted,
        target,
        supervision=_supervision(
            route_field=field,
            drivable_field=field,
        ),
    )
    terms["map"].backward()

    assert terms["map"].item() > 0.0
    assert terms["route"].item() > 0.0
    assert terms["drivable"].item() > 0.0
    assert predicted.grad is not None
    assert predicted.grad[:, :, 1].abs().sum().item() > 0.0


def test_map_validity_masks_regions_before_reduction():
    field = _field_from_lateral_band()
    predicted = _controls(curvature=0.04)
    target = _controls()

    route_only = _loss(
        predicted,
        target,
        supervision=_supervision(
            route_field=field,
            drivable_field=field,
        ),
        map_valid=False,
        route_valid=True,
    )
    unavailable = _loss(
        predicted,
        target,
        supervision=_supervision(
            route_field=field,
            drivable_field=field,
        ),
        map_valid=False,
        route_valid=False,
    )

    assert route_only["route_sample_count"].item() == 1
    assert route_only["drivable_sample_count"].item() == 0
    assert route_only["map"].item() == route_only["route"].item()
    assert unavailable["map_sample_count"].item() == 0
    assert unavailable["map"].item() == 0.0
    assert torch.isfinite(unavailable["constraint"])


def test_unavailable_nonfinite_map_terms_are_masked_before_reduction():
    field = torch.full(
        (1, GEOMETRY.height_px, GEOMETRY.width_px),
        torch.nan,
    )
    predicted = _controls().requires_grad_(True)

    terms = _loss(
        predicted,
        _controls(),
        supervision=_supervision(
            route_field=field,
            drivable_field=field,
            available=False,
        ),
        map_valid=False,
        route_valid=False,
    )

    assert terms["map"].item() == 0.0
    assert torch.isfinite(terms["constraint"])
    terms["constraint"].backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_active_nonfinite_map_field_is_rejected():
    field = torch.full(
        (1, GEOMETRY.height_px, GEOMETRY.width_px),
        torch.nan,
    )

    with pytest.raises(
        ValueError,
        match="active distance_to_corridor_m must be finite",
    ):
        _loss(
            _controls(),
            _controls(),
            supervision=_supervision(
                route_field=field,
                drivable_field=torch.zeros_like(field),
            ),
            map_valid=False,
            route_valid=True,
        )


def test_mixed_batch_averages_only_available_map_terms_per_sample():
    field = _field_from_lateral_band().repeat(2, 1, 1)
    predicted = _controls(curvature=0.04).repeat(2, 1, 1)
    target = _controls().repeat(2, 1, 1)
    initial_speed = torch.full((2,), 5.0)
    logged_positions, _, _ = integrate_controls_torch(
        target,
        initial_speed,
    )
    supervision = {
        "distance_to_corridor_m": field,
        "distance_to_drivable_m": field,
        "available": torch.tensor([True, False]),
        "drivable_available": torch.tensor([True, False]),
    }

    terms = RolloutAlignedLoss()(
        predicted,
        target,
        initial_speed,
        logged_positions,
        supervision,
        torch.tensor([True, False]),
        torch.tensor([True, False]),
    )

    assert terms["map_sample_count"].item() == 1
    assert terms["route_sample_count"].item() == 1
    assert terms["drivable_sample_count"].item() == 1
    assert terms["map"].item() > 0.0
    assert terms["comfort"].item() == 0.0
    assert terms["constraint"].item() == pytest.approx(
        terms["map"].item() / 4.0
    )


def test_unavailable_map_diagnostics_do_not_overflow_half_precision():
    predicted = _controls(acceleration=65504.0).to(torch.float16)

    terms = _loss(
        predicted,
        predicted.clone(),
        supervision=_supervision(available=False),
        map_valid=False,
        route_valid=False,
    )

    assert torch.isfinite(terms["map"])
    assert torch.isfinite(terms["route"])
    assert torch.isfinite(terms["drivable"])


def test_missing_drivable_field_keeps_route_term_available():
    field = _field_from_lateral_band()
    supervision = _supervision(
        route_field=field,
        drivable_field=torch.full_like(field, 30.0),
    )
    supervision["drivable_available"] = torch.tensor([False])

    terms = _loss(
        _controls(curvature=0.04),
        _controls(),
        supervision=supervision,
    )

    assert terms["route_sample_count"].item() == 1
    assert terms["drivable_sample_count"].item() == 0
    assert terms["map"].item() == terms["route"].item()
