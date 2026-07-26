"""Tests for rollout-aligned planner loss terms."""

from __future__ import annotations

import numpy as np
import torch

from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from training.losses.rollout_aligned_loss import RolloutAlignedLoss


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
    }


def _loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    supervision: dict[str, torch.Tensor] | None = None,
    map_valid: bool = True,
    route_valid: bool = True,
) -> dict[str, torch.Tensor]:
    return RolloutAlignedLoss()(
        predicted,
        target,
        torch.tensor([5.0], dtype=torch.float32),
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
