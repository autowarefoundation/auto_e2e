"""Tests for differentiable selected-route control supervision."""

import numpy as np
import pytest
import torch

from evaluation.metrics import integrate_trajectory
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from navigation.supervision import ROUTE_SUPERVISION_ARTIFACT_VERSION
from training.losses.control_rollout import integrate_controls_torch
from training.losses.route_consistency_loss import (
    RouteConsistencyLoss,
    ego_points_to_grid,
)


GEOMETRY = DEFAULT_NAVIGATION_GEOMETRY
TIMESTEPS = 64


def _controls(acceleration=0.0, curvature=0.0, *, requires_grad=False):
    value = torch.zeros(1, TIMESTEPS, 2)
    value[..., 0] = acceleration
    value[..., 1] = curvature
    return value.requires_grad_(requires_grad)


def _straight_route_supervision(*, destination_visible=False):
    _, y_left = GEOMETRY.pixel_center_grids()
    half_width = GEOMETRY.route_corridor_width_m
    distance = np.maximum(np.abs(y_left) - half_width, 0.0)
    corridor = np.abs(y_left) <= half_width
    return {
        "distance_to_corridor_m": torch.from_numpy(
            distance.astype(np.float32)
        ).unsqueeze(0),
        "route_heading_sin": torch.zeros(
            1,
            GEOMETRY.height_px,
            GEOMETRY.width_px,
        ),
        "route_heading_cos": torch.from_numpy(
            corridor.astype(np.float32)
        ).unsqueeze(0),
        "route_heading_valid": torch.from_numpy(corridor).unsqueeze(0),
        "destination_xy_m": torch.tensor([[100.0, 0.0]]),
        "destination_visible": torch.tensor([destination_visible]),
        "available": torch.tensor([True]),
    }


def _loss(
    predicted,
    target,
    *,
    initial_speed=5.0,
    supervision=None,
    route_valid=True,
    intersection=False,
):
    return RouteConsistencyLoss()(
        predicted,
        target,
        torch.tensor([initial_speed]),
        supervision or _straight_route_supervision(),
        torch.tensor([route_valid]),
        torch.tensor([intersection]),
    )


def test_route_loss_declares_current_supervision_contract():
    assert (
        RouteConsistencyLoss().metadata()["artifact_version"]
        == ROUTE_SUPERVISION_ARTIFACT_VERSION
    )


@pytest.mark.parametrize(
    ("acceleration", "curvature", "initial_speed"),
    [
        (0.0, 0.0, 5.0),
        (-3.0, 0.0, 2.0),
        (0.5, 0.04, 8.0),
        (0.5, -0.04, 8.0),
    ],
)
def test_torch_control_integration_matches_numpy(
    acceleration,
    curvature,
    initial_speed,
):
    controls = _controls(acceleration, curvature)

    positions, _, _ = integrate_controls_torch(
        controls,
        torch.tensor([initial_speed]),
    )
    expected = integrate_trajectory(
        np.full(TIMESTEPS, acceleration),
        np.full(TIMESTEPS, curvature),
        initial_speed,
    )

    np.testing.assert_allclose(
        positions[0].numpy(),
        expected,
        rtol=0.0,
        atol=5e-5,
    )


def test_control_rollout_accepts_flattened_controls():
    controls = _controls(0.5, -0.04)
    structured = integrate_controls_torch(
        controls,
        torch.tensor([8.0]),
    )
    flattened = integrate_controls_torch(
        controls.flatten(start_dim=1),
        torch.tensor([8.0]),
    )

    for structured_value, flattened_value in zip(
        structured,
        flattened,
        strict=True,
    ):
        torch.testing.assert_close(structured_value, flattened_value)


def test_control_rollout_is_float32_under_autocast():
    controls = _controls(0.5, 0.04).to(dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        positions, headings, speeds = integrate_controls_torch(
            controls,
            torch.tensor([8.0], dtype=torch.bfloat16),
        )

    assert positions.dtype == torch.float32
    assert headings.dtype == torch.float32
    assert speeds.dtype == torch.float32


@pytest.mark.parametrize(
    ("controls", "speed", "message"),
    [
        (
            torch.tensor([[[float("nan"), 0.0]]]),
            torch.tensor([1.0]),
            "controls contain non-finite",
        ),
        (
            torch.zeros(1, 1, 2),
            torch.tensor([float("inf")]),
            "initial_speed contains non-finite",
        ),
    ],
)
def test_control_rollout_rejects_non_finite_input(
    controls,
    speed,
    message,
):
    with pytest.raises(ValueError, match=message):
        integrate_controls_torch(controls, speed)


def test_control_rollout_clamps_finite_negative_initial_speed():
    positions, _, speeds = integrate_controls_torch(
        torch.zeros(1, 1, 2),
        torch.tensor([-1.0]),
    )

    torch.testing.assert_close(speeds, torch.zeros_like(speeds))
    torch.testing.assert_close(positions, torch.zeros_like(positions))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the device fail-fast contract",
)
def test_control_rollout_rejects_non_finite_cuda_input_immediately():
    controls = torch.tensor(
        [[[float("nan"), 0.0]]],
        device="cuda",
    )

    with pytest.raises(ValueError, match="controls contain non-finite"):
        integrate_controls_torch(
            controls,
            torch.tensor([1.0], device="cuda"),
        )


def test_control_rollout_documents_zero_gradient_after_stop_clamp():
    controls = torch.tensor(
        [[[-1.0, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions, _, _ = integrate_controls_torch(
        controls,
        torch.tensor([0.0]),
    )

    positions.sum().backward()

    assert controls.grad is not None
    assert controls.grad[0, 0, 0].item() == 0.0
    assert controls.grad[0, 0, 1].item() == 0.0


def test_vectorized_rollout_matches_recurrent_stop_and_restart():
    controls = torch.tensor(
        [[
            [-20.0, 0.02],
            [5.0, 0.03],
            [5.0, -0.01],
            [-20.0, 0.04],
            [5.0, -0.02],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    initial_speed = torch.tensor([1.0])

    actual = integrate_controls_torch(controls, initial_speed)

    speed = initial_speed
    heading = torch.zeros_like(speed)
    x = torch.zeros_like(speed)
    y = torch.zeros_like(speed)
    reference_positions = []
    reference_headings = []
    reference_speeds = []
    for step in range(controls.shape[1]):
        speed = torch.clamp_min(
            speed + controls[:, step, 0] * 0.1,
            0.0,
        )
        heading = heading + speed * controls[:, step, 1] * 0.1
        x = x + speed * torch.cos(heading) * 0.1
        y = y + speed * torch.sin(heading) * 0.1
        reference_positions.append(torch.stack((x, y), dim=-1))
        reference_headings.append(heading)
        reference_speeds.append(speed)
    reference = (
        torch.stack(reference_positions, dim=1),
        torch.stack(reference_headings, dim=1),
        torch.stack(reference_speeds, dim=1),
    )

    for actual_value, reference_value in zip(
        actual,
        reference,
        strict=True,
    ):
        torch.testing.assert_close(actual_value, reference_value)

    actual_gradient = torch.autograd.grad(
        sum(value.sum() for value in actual),
        controls,
        retain_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(value.sum() for value in reference),
        controls,
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_control_rollout_matches_clamp_subgradient_at_zero_speed():
    controls = torch.zeros(
        1,
        4,
        2,
        dtype=torch.float32,
        requires_grad=True,
    )
    initial_speed = torch.zeros(1, dtype=torch.float32)
    actual = integrate_controls_torch(controls, initial_speed)

    speed = initial_speed
    reference_speeds = []
    for step in range(controls.shape[1]):
        speed = torch.clamp_min(
            speed + controls[:, step, 0] * 0.1,
            0.0,
        )
        reference_speeds.append(speed)
    reference = torch.stack(reference_speeds, dim=1)

    torch.testing.assert_close(actual[2], reference)
    actual_gradient = torch.autograd.grad(
        actual[2].sum(),
        controls,
        retain_graph=True,
    )[0]
    reference_gradient = torch.autograd.grad(
        reference.sum(),
        controls,
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA recurrence parity requires a GPU",
)
def test_cuda_rollout_matches_recurrent_values_and_gradients():
    controls = torch.tensor(
        [[
            [-10.0, 0.03],
            [0.0, 0.02],
            [5.0, -0.01],
            [5.0, -0.02],
        ]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    reference_controls = controls.detach().clone().requires_grad_(True)
    initial_speed = torch.tensor([1.0], device="cuda")

    actual = integrate_controls_torch(controls, initial_speed)
    speed = initial_speed
    heading = torch.zeros_like(speed)
    x = torch.zeros_like(speed)
    y = torch.zeros_like(speed)
    reference_positions = []
    reference_headings = []
    reference_speeds = []
    for step in range(reference_controls.shape[1]):
        speed = torch.clamp_min(
            speed + reference_controls[:, step, 0] * 0.1,
            0.0,
        )
        heading = (
            heading
            + speed * reference_controls[:, step, 1] * 0.1
        )
        x = x + speed * torch.cos(heading) * 0.1
        y = y + speed * torch.sin(heading) * 0.1
        reference_positions.append(torch.stack((x, y), dim=-1))
        reference_headings.append(heading)
        reference_speeds.append(speed)
    reference = (
        torch.stack(reference_positions, dim=1),
        torch.stack(reference_headings, dim=1),
        torch.stack(reference_speeds, dim=1),
    )

    for actual_value, reference_value in zip(
        actual,
        reference,
        strict=True,
    ):
        torch.testing.assert_close(actual_value, reference_value)
    actual_gradient = torch.autograd.grad(
        sum(value.sum() for value in actual),
        controls,
    )[0]
    reference_gradient = torch.autograd.grad(
        sum(value.sum() for value in reference),
        reference_controls,
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)


def test_grid_coordinates_match_geometry_pixel_centers():
    points = torch.tensor([[
        [0.0, 0.0],
        [20.0, 5.0],
        [GEOMETRY.x_max_m, GEOMETRY.y_max_m],
    ]])

    grid = ego_points_to_grid(points, GEOMETRY)[0].numpy()
    pixels = GEOMETRY.ego_to_pixel(points[0].numpy())
    expected = np.column_stack([
        2.0 * (pixels[:, 1] + 0.5) / GEOMETRY.width_px - 1.0,
        2.0 * (pixels[:, 0] + 0.5) / GEOMETRY.height_px - 1.0,
    ])

    np.testing.assert_allclose(grid, expected, rtol=0.0, atol=1e-6)


def test_out_of_bounds_motion_has_explicit_positive_gradient():
    predicted = _controls(20.0, 0.0, requires_grad=True)
    supervision = _straight_route_supervision()
    supervision["distance_to_corridor_m"].zero_()

    terms = _loss(
        predicted,
        _controls(),
        supervision=supervision,
    )
    terms["total"].backward()

    assert terms["eligible_count"].item() == 1
    assert terms["corridor"].item() > 0.0
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item() > 0.0


def test_wrong_junction_branch_costs_more_than_selected_route():
    target = _controls()
    selected = _controls(requires_grad=True)
    wrong_branch = _controls(0.0, 0.05, requires_grad=True)

    selected_terms = _loss(selected, target, intersection=True)
    wrong_terms = _loss(wrong_branch, target, intersection=True)

    assert selected_terms["branch_active_count"].item() == 1
    assert selected_terms["total"].item() == pytest.approx(0.0)
    assert wrong_terms["branch"].item() > selected_terms["branch"].item()
    assert wrong_terms["total"].item() > selected_terms["total"].item()


def test_destination_hinge_penalizes_only_lost_progress():
    target = _controls()
    supervision = _straight_route_supervision(destination_visible=True)
    farther = _controls(-0.5, 0.0, requires_grad=True)
    closer = _controls(0.5, 0.0, requires_grad=True)

    farther_terms = _loss(
        farther,
        target,
        supervision=supervision,
    )
    closer_terms = _loss(
        closer,
        target,
        supervision=supervision,
    )

    assert farther_terms["destination"].item() > 0.0
    assert closer_terms["destination"].item() == pytest.approx(0.0)


def test_stationary_trajectory_has_no_heading_penalty():
    predicted = _controls(0.0, 1.0, requires_grad=True)

    terms = _loss(
        predicted,
        _controls(),
        initial_speed=0.0,
    )

    assert terms["eligible_count"].item() == 1
    assert terms["heading_active_count"].item() == 0
    assert terms["heading"].item() == pytest.approx(0.0)


def test_inconsistent_target_rejects_route_auxiliary():
    predicted = _controls(requires_grad=True)
    target = _controls(0.0, 0.1)

    terms = _loss(predicted, target)
    terms["total"].backward()

    assert terms["candidate_count"].item() == 1
    assert terms["eligible_count"].item() == 0
    assert terms["compliance_rejected_count"].item() == 1
    assert terms["total"].item() == pytest.approx(0.0)
    assert predicted.grad is not None
    assert predicted.grad.abs().sum().item() == pytest.approx(0.0)


def test_empty_route_set_returns_differentiable_zero():
    predicted = _controls(requires_grad=True)

    terms = _loss(predicted, _controls(), route_valid=False)
    terms["total"].backward()

    assert terms["candidate_count"].item() == 0
    assert terms["eligible_count"].item() == 0
    assert terms["total"].item() == pytest.approx(0.0)
    assert predicted.grad is not None
    assert predicted.grad.abs().sum().item() == pytest.approx(0.0)
