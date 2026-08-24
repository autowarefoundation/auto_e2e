"""Reactive-only nuPlan/L2D multi-task contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data_processing.reactive_training_artifacts import (
    BEV_SEGMENTATION_ARTIFACT_VERSION,
    BEV_SEGMENTATION_TAXONOMY_VERSION,
    decode_bev_segmentation,
    decode_trajectory_xy,
    encode_bev_segmentation,
    encode_trajectory_xy,
)
from model_components.auxiliary_heads import BEVSegmentationHead
from model_components.losses import (
    BEVSegmentationAuxiliaryLoss,
    RouteReconstructionLoss,
    TrajectoryXYImitationLoss,
)
from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
from training.reactive_multitask import (
    ReactiveMultitaskObjective,
    ReactiveTrainingStage,
    configure_model_for_stage,
    reactive_model_kwargs,
)
from training.reactive_stage_runner import (
    evaluate_reactive_multitask,
    evaluate_reactive_transfer_matrix_models,
    evaluate_reactive_xy,
    load_stage_a_parent,
    reactive_model_state_sha256,
    run_reactive_epoch,
    save_reactive_checkpoint,
)


def test_model_state_digest_supports_scalar_tensors():
    state = {
        "counter": torch.tensor(3, dtype=torch.long),
        "scale": torch.tensor(1.5, dtype=torch.bfloat16),
    }

    digest = reactive_model_state_sha256(state)

    assert len(digest) == 64
    assert digest == reactive_model_state_sha256(state)
    assert digest != reactive_model_state_sha256({
        **state,
        "counter": torch.tensor(4, dtype=torch.long),
    })
    assert digest != reactive_model_state_sha256({
        **state,
        "counter": torch.tensor([3], dtype=torch.long),
    })


def test_bev_artifact_encoding_does_not_follow_taxonomy_version():
    assert BEV_SEGMENTATION_ARTIFACT_VERSION == (
        "bev_segmentation_" + "v2"
    )
    assert BEV_SEGMENTATION_TAXONOMY_VERSION == (
        "bev_segmentation_" + "v3"
    )
    assert (
        BEV_SEGMENTATION_ARTIFACT_VERSION
        != BEV_SEGMENTATION_TAXONOMY_VERSION
    )


def _inputs(device: torch.device, *, batch_size: int = 2, views: int = 8):
    return {
        "visual": torch.randn(
            batch_size,
            views,
            3,
            256,
            256,
            device=device,
        ),
        "map": torch.rand(
            batch_size,
            14,
            256,
            256,
            device=device,
        ),
        "route": torch.rand(
            batch_size,
            2,
            256,
            256,
            device=device,
        ),
        "visual_history": torch.randn(
            batch_size,
            896,
            device=device,
        ),
        "egomotion": torch.randn(
            batch_size,
            256,
            device=device,
        ),
    }


def _model(build_mock_model, device, *, views: int = 8):
    return build_mock_model(
        num_views=views,
        device=device,
        map_context_channels=14,
        route_channels=2,
        map_type="semantic_raster",
        planner_mode="gru",
        enable_bev_segmentation=True,
        enable_route_reconstruction=True,
    )


def _forward(model, values, **kwargs):
    return model(
        values["visual"],
        values["map"],
        values["visual_history"],
        values["egomotion"],
        route_mask=values["route"],
        map_valid=torch.ones(
            values["visual"].shape[0],
            dtype=torch.bool,
            device=values["visual"].device,
        ),
        route_valid=torch.ones(
            values["visual"].shape[0],
            dtype=torch.bool,
            device=values["visual"].device,
        ),
        mode="train",
        **kwargs,
    )


def _stage_batch(
    device: torch.device,
    *,
    include_bev: bool,
    batch_size: int = 1,
    views: int = 8,
) -> dict[str, object]:
    batch: dict[str, object] = {
        "sample_uid": [
            f"synthetic-sample-{index}"
            for index in range(batch_size)
        ],
        "visual_tiles": torch.randn(
            batch_size,
            views,
            3,
            256,
            256,
            device=device,
        ),
        "map_context": torch.rand(
            batch_size,
            14,
            8,
            8,
            device=device,
        ),
        "route_mask": torch.rand(
            batch_size,
            2,
            8,
            8,
            device=device,
        ),
        "map_valid": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "route_valid": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "route_channel_valid": torch.ones(
            batch_size,
            2,
            dtype=torch.bool,
            device=device,
        ),
        "visual_history": torch.randn(
            batch_size,
            896,
            device=device,
        ),
        "egomotion_history": torch.randn(
            batch_size,
            256,
            device=device,
        ),
        "trajectory_xy_m": torch.zeros(
            batch_size,
            64,
            2,
            device=device,
        ),
        "trajectory_valid": torch.ones(
            batch_size,
            64,
            dtype=torch.bool,
            device=device,
        ),
        "initial_speed_mps": torch.ones(
            batch_size,
            device=device,
        ),
        "bev_segmentation_available": torch.full(
            (batch_size,),
            include_bev,
            dtype=torch.bool,
            device=device,
        ),
    }
    if include_bev:
        batch["bev_segmentation_target"] = torch.rand(
            batch_size,
            8,
            8,
            8,
            device=device,
        )
        batch["bev_segmentation_valid"] = torch.ones(
            batch_size,
            8,
            8,
            8,
            dtype=torch.bool,
            device=device,
        )
    return batch


def test_common_geometry_matches_camera_bev_contract():
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    assert (geometry.height_px, geometry.width_px) == (450, 300)
    assert geometry.meters_per_pixel == pytest.approx(0.4)
    assert geometry.matching_pc_range == (
        -60.0,
        -60.0,
        -5.0,
        120.0,
        60.0,
        3.0,
    )
    points = np.asarray([[0.0, 0.0], [10.0, -4.0]])
    assert np.allclose(
        geometry.pixel_to_ego(geometry.ego_to_pixel(points)),
        points,
    )


def test_reactive_model_contract_uses_bevformer_t1_latent_grid():
    kwargs = reactive_model_kwargs(
        ReactiveTrainingStage.NUPLAN_FULL,
        num_views=8,
    )

    assert "image_feature_size" not in kwargs
    assert kwargs["view_fusion_kwargs"] == {
        "architecture": "bevformer_v2_t1",
        "activation_checkpointing": True,
        "bev_h": 256,
        "bev_w": 256,
        "feedforward_channels": 512,
        "image_size": 256,
        "num_encoder_layers": 6,
        "num_heads": 8,
        "num_levels": 4,
        "num_points": 8,
        "pc_range": [-60.0, -60.0, -5.0, 120.0, 60.0, 3.0],
        "query_chunk_size": 4096,
    }
    assert kwargs["map_fusion_mode"] == "deformable"
    assert kwargs["route_encoder_hidden_channels"] == 96
    assert kwargs["auxiliary_output_size"] == (450, 300)


def test_bev_head_uses_full_residual_spatial_blocks(device):
    head = BEVSegmentationHead(
        embed_dim=16,
        hidden_channels=8,
        num_classes=3,
        num_groups=4,
    ).to(device)
    spatial_convolutions = [
        module
        for module in head.modules()
        if (
            isinstance(module, torch.nn.Conv2d)
            and module.kernel_size == (3, 3)
        )
    ]

    assert len(spatial_convolutions) == 4
    assert all(module.groups == 1 for module in spatial_convolutions)
    assert head.decoder[-1].bias is not None

    image_bev = torch.randn(
        2,
        16,
        7,
        5,
        device=device,
        requires_grad=True,
    )
    logits = head(image_bev)
    logits.square().mean().backward()

    assert logits.shape == (2, 3, 7, 5)
    assert image_bev.grad is not None
    assert torch.isfinite(image_bev.grad).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_reactive_model_emits_both_auxiliary_heads(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device)
    trajectory, auxiliary = _forward(model, _inputs(device))

    assert trajectory.shape == (2, 128)
    assert auxiliary["bev_segmentation_logits"].shape == (2, 8, 8, 8)
    assert auxiliary["route_reconstruction_logits"].shape == (2, 2, 8, 8)


def test_bev_logits_do_not_depend_on_navigation(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).eval()
    values = _inputs(device)
    _, first = _forward(model, values)
    values["map"] = torch.rand_like(values["map"])
    values["route"] = torch.rand_like(values["route"])
    _, second = _forward(model, values)

    assert torch.equal(
        first["bev_segmentation_logits"],
        second["bev_segmentation_logits"],
    )


def test_route_loss_reaches_route_gate_but_not_camera_or_map(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).train()
    values = _inputs(device)
    _, auxiliary = _forward(model, values)
    loss = RouteReconstructionLoss()(
        auxiliary["route_reconstruction_logits"],
        F.interpolate(values["route"], size=(8, 8), mode="nearest"),
        torch.ones(2, 2, dtype=torch.bool, device=device),
    )
    loss.backward()

    reactive = model.Reactive_E2E
    route_gate = reactive.NavigationEncoder.route_gate
    assert route_gate.grad is not None
    assert bool((route_gate.grad != 0).any())
    assert reactive.MapBEVFusion.alpha.grad is None
    assert any(
        parameter.grad is not None
        for parameter in reactive.NavigationEncoder.RouteEncoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in reactive.NavigationEncoder.MapEncoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in reactive.Backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in reactive.FeatureFusion.parameters()
    )


def test_bev_loss_reaches_camera_but_not_navigation(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).train()
    _, auxiliary = _forward(model, _inputs(device))
    logits = auxiliary["bev_segmentation_logits"]
    target = torch.rand_like(logits)
    loss = BEVSegmentationAuxiliaryLoss([1.0] * 8).to(device)(
        logits,
        target,
        torch.ones_like(logits, dtype=torch.bool),
    )
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.Reactive_E2E.Backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.Reactive_E2E.NavigationEncoder.parameters()
    )
    assert model.Reactive_E2E.MapBEVFusion.alpha.grad is None


def test_all_invalid_losses_are_differentiable_zero():
    bev_logits = torch.randn(2, 8, 4, 4, requires_grad=True)
    bev = BEVSegmentationAuxiliaryLoss([1.0] * 8)(
        bev_logits,
        torch.zeros_like(bev_logits),
        torch.zeros_like(bev_logits, dtype=torch.bool),
    )
    route_logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    route = RouteReconstructionLoss()(
        route_logits,
        torch.zeros_like(route_logits),
        torch.zeros(2, 2, dtype=torch.bool),
    )
    controls = torch.randn(2, 128, requires_grad=True)
    trajectory = TrajectoryXYImitationLoss()(
        controls,
        torch.zeros(2, 64, 2),
        torch.zeros(2, 64, dtype=torch.bool),
        torch.zeros(2),
    )
    total = bev + route + trajectory
    total.backward()

    assert total.item() == 0.0
    assert bev_logits.grad is not None
    assert route_logits.grad is not None
    assert controls.grad is not None


def test_bev_loss_stays_fp32_and_handles_empty_targets():
    loss_fn = BEVSegmentationAuxiliaryLoss([64.0])
    target = torch.zeros(1, 1, 16, 16)
    valid = torch.ones_like(target, dtype=torch.bool)
    negative_logits = torch.full(
        target.shape,
        -20.0,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positive_logits = torch.full(
        target.shape,
        20.0,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        negative_components = loss_fn.components(
            negative_logits,
            target,
            valid,
        )
        negative_loss = negative_components["total"]
        positive_loss = loss_fn(positive_logits, target, valid)
    negative_loss.backward()

    assert negative_loss.dtype == torch.float32
    assert negative_loss.item() < 1e-5
    assert negative_components["dice"].item() == 0.0
    assert negative_loss.item() == pytest.approx(
        0.5 * negative_components["bce"].item()
    )
    assert positive_loss.item() > 1.0
    assert torch.isfinite(negative_logits.grad).all()


def test_repeat_importance_preserves_non_bev_objective_mean():
    objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.L2D_CONTINUATION,
        bev_pos_weight=[1.0] * 8,
    )
    predicted_controls = torch.zeros(1, 128)
    auxiliary = {
        "route_reconstruction_logits": torch.zeros(1, 2, 4, 4),
    }
    batch = {
        "trajectory_xy_m": torch.ones(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.zeros(1),
        "route_mask": torch.zeros(1, 2, 4, 4),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    baseline = objective(
        predicted_controls,
        auxiliary,
        batch,
    )

    weighted_terms = []
    for repeat, importance in ((1, 2.0), (3, 2.0 / 3.0)):
        weighted_batch = {
            **batch,
            "bev_sampling_importance": torch.tensor([importance]),
        }
        terms = objective(
            predicted_controls,
            auxiliary,
            weighted_batch,
        )
        weighted_terms.extend([terms] * repeat)

    for name in ("trajectory", "route_reconstruction"):
        weighted_mean = torch.stack([
            terms[name] for terms in weighted_terms
        ]).mean()
        assert weighted_mean.item() == pytest.approx(
            baseline[name].item()
        )


def test_multitask_total_applies_every_objective_weight():
    objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_pos_weight=[1.0],
        trajectory_weight=0.25,
        bev_weight=0.5,
        route_weight=0.75,
    )
    controls = torch.zeros(1, 128)
    bev_logits = torch.zeros(1, 1, 4, 4)
    route_logits = torch.zeros(1, 2, 4, 4)
    batch = {
        "trajectory_xy_m": torch.ones(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.zeros(1),
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": torch.ones_like(bev_logits),
        "bev_segmentation_valid": torch.ones_like(
            bev_logits,
            dtype=torch.bool,
        ),
        "route_mask": torch.zeros_like(route_logits),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
    }

    terms = objective(
        controls,
        {
            "bev_segmentation_logits": bev_logits,
            "route_reconstruction_logits": route_logits,
        },
        batch,
    )

    expected = (
        0.25 * terms["trajectory"]
        + 0.5 * terms["bev_segmentation"]
        + 0.75 * terms["route_reconstruction"]
    )
    assert torch.equal(terms["total"], expected)


def test_bev_only_objective_skips_inactive_inputs_and_gradients():
    objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_pos_weight=[1.0],
        trajectory_weight=0.0,
        bev_weight=1.0,
        route_weight=0.0,
    )
    controls = torch.randn(1, 128, requires_grad=True)
    bev_logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    batch = {
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": torch.ones_like(bev_logits),
        "bev_segmentation_valid": torch.ones_like(
            bev_logits,
            dtype=torch.bool,
        ),
    }

    terms = objective(
        controls,
        {"bev_segmentation_logits": bev_logits},
        batch,
    )
    terms["total"].backward()

    assert objective.compute_bev_segmentation
    assert not objective.compute_route_reconstruction
    assert terms["trajectory"].item() == 0.0
    assert terms["route_reconstruction"].item() == 0.0
    assert controls.grad is not None
    assert torch.count_nonzero(controls.grad).item() == 0
    assert bev_logits.grad is not None
    assert torch.count_nonzero(bev_logits.grad).item() > 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trajectory_weight": float("nan")}, "trajectory_weight"),
        ({"bev_weight": float("inf")}, "bev_weight"),
        ({"route_weight": -1.0}, "route_weight"),
        ({"route_weight": 1_001.0}, "route_weight"),
        ({"corridor_pos_weight": 0.5}, "corridor_pos_weight"),
    ],
)
def test_multitask_objective_rejects_invalid_weights(overrides, match):
    with pytest.raises(ValueError, match=match):
        ReactiveMultitaskObjective(
            ReactiveTrainingStage.NUPLAN_FULL,
            bev_pos_weight=[1.0],
            **overrides,
        )


def test_perfect_multitask_predictions_approach_zero():
    bev_target = torch.zeros(1, 8, 4, 4)
    bev_target[:, :, 1:3, 1:3] = 1.0
    bev_logits = torch.where(
        bev_target > 0.5,
        torch.full_like(bev_target, 20.0),
        torch.full_like(bev_target, -20.0),
    )
    bev_loss = BEVSegmentationAuxiliaryLoss([1.0] * 8)(
        bev_logits,
        bev_target,
        torch.ones_like(bev_target, dtype=torch.bool),
    )

    route_target = torch.zeros(1, 2, 4, 4)
    route_target[:, 0, 1:3, 1:3] = 1.0
    route_target[:, 1, 2, 1] = 1.0
    route_logits = torch.where(
        route_target > 0.5,
        torch.full_like(route_target, 20.0),
        torch.full_like(route_target, -20.0),
    )
    route_loss = RouteReconstructionLoss()(
        route_logits,
        route_target,
        torch.ones(1, 2, dtype=torch.bool),
    )

    controls = torch.zeros(1, 128)
    trajectory_loss = TrajectoryXYImitationLoss()
    target_xy = trajectory_loss.predicted_xy(
        controls,
        torch.ones(1),
    )
    xy_loss = trajectory_loss(
        controls,
        target_xy,
        torch.ones(1, 64, dtype=torch.bool),
        torch.ones(1),
    )

    assert bev_loss.item() < 1e-5
    assert route_loss.item() < 1e-5
    assert xy_loss.item() == pytest.approx(0.0)


def test_route_loss_components_preserve_masked_total():
    logits = torch.zeros(2, 2, 4, 6)
    target = torch.zeros_like(logits)
    target[0, 0, 1:3, 2:4] = 1.0
    target[1, 1, 2, 4] = 1.0
    channel_valid = torch.tensor([
        [True, False],
        [False, True],
    ])
    loss_function = RouteReconstructionLoss()

    components = loss_function.components(
        logits,
        target,
        channel_valid,
    )
    expected_total = (
        0.5 * components["corridor_bce"]
        + 0.5 * components["corridor_dice"]
        + 0.25 * components["destination_focal"]
    ) / 2.0

    assert components["total"].dtype == torch.float32
    assert torch.equal(components["total"], expected_total)
    assert torch.equal(
        loss_function(logits, target, channel_valid),
        components["total"],
    )


def test_zero_weight_destination_only_route_loss_is_finite():
    logits = torch.zeros(1, 2, 4, 6)
    target = torch.zeros_like(logits)
    target[0, 1, 2, 4] = 1.0
    channel_valid = torch.tensor([[False, True]])

    components = RouteReconstructionLoss(
        destination_weight=0.0,
    ).components(logits, target, channel_valid)

    assert components["total"].item() == 0.0
    assert torch.isfinite(components["total"])
    assert components["destination_focal"].item() > 0.0


def test_route_destination_focal_is_resolution_invariant():
    losses = []
    for height, width in ((32, 32), (450, 300)):
        logits = torch.zeros(1, 2, height, width)
        target = torch.zeros_like(logits)
        target[:, 1, height // 2, width // 2] = 1.0
        loss = RouteReconstructionLoss()(
            logits,
            target,
            torch.tensor([[False, True]]),
        )
        losses.append(loss)

    expected = 0.25 * 2.0 * (-np.log(0.5)) * 0.5**2
    assert losses[0].item() == pytest.approx(expected)
    assert losses[1].item() == pytest.approx(expected)


def test_route_destination_focal_stays_fp32_for_bf16_extremes():
    logits = torch.full(
        (1, 2, 32, 32),
        100.0,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.zeros_like(logits)
    target[:, 1, 16, 16] = 1.0

    loss = RouteReconstructionLoss()(
        logits,
        target,
        torch.tensor([[False, True]]),
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_route_corridor_loss_stays_fp32_for_bf16_production_raster():
    logits = torch.zeros(
        1,
        2,
        450,
        300,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.zeros_like(logits)
    target[:, 0, 120:330, 110:190] = 1.0
    channel_valid = torch.tensor([[True, False]])
    loss_function = RouteReconstructionLoss(corridor_pos_weight=8.0)

    loss = loss_function(logits, target, channel_valid)
    reference = loss_function(
        logits.detach().float(),
        target.float(),
        channel_valid,
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(reference.item())
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_trajectory_loss_reaches_all_reactive_modules(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).train()
    with torch.no_grad():
        model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.5)
    values = _inputs(device)
    controls, _ = _forward(model, values)
    loss = TrajectoryXYImitationLoss()(
        controls,
        torch.zeros(2, 64, 2, device=device),
        torch.ones(2, 64, dtype=torch.bool, device=device),
        torch.ones(2, device=device),
    )
    loss.backward()

    reactive = model.Reactive_E2E
    assert any(
        parameter.grad is not None
        for parameter in reactive.Backbone.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in reactive.NavigationEncoder.parameters()
    )
    assert reactive.MapBEVFusion.alpha.grad is not None
    assert bool((reactive.MapBEVFusion.alpha.grad != 0).any())
    assert any(
        parameter.grad is not None
        for parameter in reactive.TrajectoryPlanner.parameters()
    )


def test_route_changes_reconstruction_and_planner_output(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).eval()
    with torch.no_grad():
        model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.5)
    values = _inputs(device)
    first_controls, first_auxiliary = _forward(model, values)
    values["route"] = torch.flip(values["route"], dims=(-2, -1))
    second_controls, second_auxiliary = _forward(model, values)

    assert not torch.equal(
        first_auxiliary["route_reconstruction_logits"],
        second_auxiliary["route_reconstruction_logits"],
    )
    assert not torch.equal(first_controls, second_controls)


def test_stage_a_optimizer_smoke(build_mock_model, device):
    model = _model(build_mock_model, device).train()
    configure_model_for_stage(model, ReactiveTrainingStage.NUPLAN_FULL)
    values = _inputs(device)
    trajectory, auxiliary = _forward(model, values)
    objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_pos_weight=[1.0] * 8,
        bev_weight=0.1,
        route_weight=0.01,
    ).to(device)
    target_xy = objective.trajectory_loss.predicted_xy(
        torch.zeros_like(trajectory),
        torch.ones(2, device=device),
    ).detach()
    batch = {
        "trajectory_xy_m": target_xy,
        "trajectory_valid": torch.ones(
            2,
            64,
            dtype=torch.bool,
            device=device,
        ),
        "initial_speed_mps": torch.ones(2, device=device),
        "route_mask": F.interpolate(
            values["route"],
            size=(8, 8),
            mode="nearest",
        ),
        "route_channel_valid": torch.ones(
            2,
            2,
            dtype=torch.bool,
            device=device,
        ),
        "bev_segmentation_target": torch.rand(
            2,
            8,
            8,
            8,
            device=device,
        ),
        "bev_segmentation_valid": torch.ones(
            2,
            8,
            8,
            8,
            dtype=torch.bool,
            device=device,
        ),
        "bev_segmentation_available": torch.ones(
            2,
            dtype=torch.bool,
            device=device,
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    terms = objective(trajectory, auxiliary, batch)
    optimizer.zero_grad()
    terms["total"].backward()
    optimizer.step()

    assert torch.isfinite(terms["total"])
    assert terms["trajectory"].item() >= 0.0
    assert terms["bev_segmentation"].item() >= 0.0
    assert terms["route_reconstruction"].item() >= 0.0


def test_stage_b_skips_and_freezes_bev_head(build_mock_model, device):
    model = _model(build_mock_model, device).train()
    configure_model_for_stage(
        model,
        ReactiveTrainingStage.L2D_CONTINUATION,
    )
    calls = 0

    def record_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.Reactive_E2E.BEVSegmentationHead.register_forward_hook(
        record_call
    )
    try:
        _, auxiliary = _forward(
            model,
            _inputs(device, views=8),
            compute_bev_segmentation=False,
        )
    finally:
        handle.remove()

    assert calls == 0
    assert "bev_segmentation_logits" not in auxiliary
    assert all(
        not parameter.requires_grad
        for parameter in model.Reactive_E2E.BEVSegmentationHead.parameters()
    )


def test_bev_only_stage_excludes_and_preserves_non_bev_parameters(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).train()
    configure_model_for_stage(
        model,
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_only=True,
    )
    reactive = model.Reactive_E2E
    trainable = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    expected_trainable = {
        id(parameter)
        for module in (
            reactive.Backbone,
            reactive.FeatureFusion,
            reactive.BEVSegmentationHead,
        )
        for parameter in module.parameters()
    }
    frozen_navigation = {
        name: parameter.detach().clone()
        for name, parameter in reactive.NavigationEncoder.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=1e-3,
        weight_decay=0.1,
    )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    _, auxiliary = _forward(
        model,
        _inputs(device),
        compute_route_reconstruction=False,
    )
    logits = auxiliary["bev_segmentation_logits"]
    loss = BEVSegmentationAuxiliaryLoss([1.0] * 8).to(device)(
        logits,
        torch.ones_like(logits),
        torch.ones_like(logits, dtype=torch.bool),
    )
    loss.backward()
    optimizer.step()

    assert trainable == expected_trainable
    assert optimizer_parameters == expected_trainable
    for name, parameter in reactive.NavigationEncoder.named_parameters():
        assert torch.equal(parameter.detach(), frozen_navigation[name])


def test_packed_reactive_targets_round_trip():
    xy = np.arange(128, dtype=np.float32).reshape(64, 2)
    trajectory_valid = np.ones(64, dtype=np.bool_)
    encoded_xy = encode_trajectory_xy(xy, trajectory_valid)
    decoded_xy, decoded_valid = decode_trajectory_xy(encoded_xy)
    assert np.array_equal(decoded_xy, xy)
    assert np.array_equal(decoded_valid, trajectory_valid)

    target = np.linspace(
        0.0,
        1.0,
        num=8 * 5 * 4,
        dtype=np.float32,
    ).reshape(8, 5, 4)
    valid = np.ones_like(target, dtype=np.bool_)
    encoded_bev = encode_bev_segmentation(target, valid)
    decoded_target, decoded_bev_valid = decode_bev_segmentation(encoded_bev)
    assert np.max(np.abs(decoded_target - target)) <= 1.0 / 255.0
    assert np.array_equal(decoded_bev_valid, valid)


def test_semantic_artifact_accepts_prequantized_frames():
    from Platform.pipelines.semantic_occupancy import (
        encode_semantic_occupancy,
        quantize_semantic_occupancy,
    )

    probability = np.linspace(
        0.0,
        1.0,
        num=8 * 5 * 4,
        dtype=np.float32,
    ).reshape(1, 8, 5, 4)
    quantized = quantize_semantic_occupancy(probability[0])

    assert quantized.dtype == np.uint8
    assert quantized.shape == (8, 5, 4)
    assert encode_semantic_occupancy(
        ["sample-a"],
        np.stack([quantized]),
    ) == encode_semantic_occupancy(["sample-a"], probability)


def test_stage_a_to_stage_b_to_semantic_artifact_smoke(
    build_mock_model,
    device,
    tmp_path,
):
    from Platform.pipelines.semantic_occupancy import (
        decode_semantic_occupancy,
        encode_semantic_occupancy,
        infer_semantic_occupancy,
    )

    stage_a_model = _model(build_mock_model, device).train()
    stage_a_objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.NUPLAN_FULL,
        bev_pos_weight=[1.0] * 8,
        bev_weight=0.1,
        route_weight=0.01,
    ).to(device)
    stage_a_optimizer = torch.optim.AdamW(
        stage_a_model.parameters(),
        lr=1e-4,
    )
    stage_a_metrics = run_reactive_epoch(
        stage_a_model,
        [_stage_batch(device, include_bev=True)],
        stage_a_objective,
        stage_a_optimizer,
        device=device,
    )
    assert np.isfinite(stage_a_metrics["total"])

    checkpoint_path = tmp_path / "stage-a.pt"
    checkpoint_sha256 = save_reactive_checkpoint(
        checkpoint_path,
        stage_a_model,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        dataset_manifest_sha256="a" * 64,
        epoch=1,
        model_config={
            "num_views": 8,
            "trajectory_weight": 1.0,
            "bev_weight": 0.1,
            "route_weight": 0.01,
            "corridor_pos_weight": 1.0,
            "training_seed": 149,
            "scheduler_identity": "selection_plateau_v1",
            "overfit_bev_only": False,
            "overfit_fixed_lr": False,
            "bev_pos_weights": [1.0] * 8,
            "bev_repeat_factors": [1] * 8,
            "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
            "is_pretrained": False,
            "allow_random_bevformer_init": True,
        },
        optimizer=stage_a_optimizer,
        metrics=stage_a_metrics,
    )
    assert len(checkpoint_sha256) == 64

    stage_b_model = _model(
        build_mock_model,
        device,
        views=6,
    ).train()
    lineage = load_stage_a_parent(stage_b_model, checkpoint_path)
    assert lineage["stage_a_parent_checkpoint_sha256"] == (
        checkpoint_sha256
    )
    assert lineage["bevformer_v2_initialization_mode"] == (
        "explicit_random_init"
    )
    configure_model_for_stage(
        stage_b_model,
        ReactiveTrainingStage.L2D_CONTINUATION,
    )
    frozen_bev = {
        name: parameter.detach().clone()
        for name, parameter in (
            stage_b_model.Reactive_E2E.BEVSegmentationHead.named_parameters()
        )
    }
    stage_b_optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in stage_b_model.parameters()
            if parameter.requires_grad
        ],
        lr=3e-5,
    )
    assert not stage_b_optimizer.state
    stage_b_objective = ReactiveMultitaskObjective(
        ReactiveTrainingStage.L2D_CONTINUATION,
        bev_pos_weight=[1.0] * 8,
        bev_weight=0.0,
        route_weight=0.01,
    ).to(device)
    stage_b_batch = _stage_batch(
        device,
        include_bev=False,
        views=6,
    )
    stage_b_metrics = run_reactive_epoch(
        stage_b_model,
        [stage_b_batch],
        stage_b_objective,
        stage_b_optimizer,
        device=device,
    )
    assert stage_b_metrics["bev_segmentation"] == 0.0
    assert stage_b_optimizer.state
    for name, parameter in (
        stage_b_model.Reactive_E2E.BEVSegmentationHead.named_parameters()
    ):
        assert torch.equal(parameter.detach(), frozen_bev[name])

    sample_uids, probability, teacher, valid_mask = (
        infer_semantic_occupancy(
            stage_b_model,
            [stage_b_batch],
            device=device,
        )
    )
    payload = encode_semantic_occupancy(
        sample_uids,
        probability,
        teacher=teacher,
        valid_mask=valid_mask,
    )
    decoded = decode_semantic_occupancy(payload)
    assert sample_uids == ["synthetic-sample-0"]
    assert decoded.probability.shape == (1, 8, 8, 8)
    assert decoded.teacher is None
    assert decoded.valid_mask is None
    assert np.max(np.abs(
        decoded.probability - probability
    )) <= 1.0 / 255.0


def test_multitask_evaluator_reports_partial_horizons_and_route_use(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).eval()
    with torch.no_grad():
        model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.5)
    batch = _stage_batch(
        device,
        include_bev=True,
        batch_size=2,
    )
    batch["trajectory_valid"][:, 50:] = False
    report = evaluate_reactive_multitask(
        model,
        [batch],
        device=device,
    )

    assert report["schema_version"] == (
        "reactive_multitask_evaluation_v1"
    )
    assert report["sample_count"] == 2
    assert len(report["sample_uid_sha256"]) == 64
    assert report["trajectory"]["ade_5s_sample_count"] == 2
    assert report["trajectory"]["fde_5s_sample_count"] == 2
    assert report["trajectory"]["fde_6p4s_m"] is None
    assert report["bev_segmentation"]["available"] is True
    assert set(report["bev_segmentation"]["per_class"]) == {
        "drivable_area",
        "lane_boundary",
        "intersection",
        "crosswalk",
        "stop_line",
        "vehicle",
        "vulnerable_road_user",
        "other_obstacle",
    }
    assert report["route"]["corridor_valid_sample_count"] == 2
    assert report["route"]["route_zero_sample_count"] == 2
    assert report["route"]["route_swap_sample_count"] == 2
    assert report["route"]["route_input_gradient_mean_abs"] > 0.0


def test_stage_a_b_cross_dataset_retention_matrix_smoke(
    build_mock_model,
    device,
):
    stage_a_model = _model(build_mock_model, device, views=8).eval()
    stage_b_model = _model(build_mock_model, device, views=6).eval()
    with torch.no_grad():
        stage_a_model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.5)
        stage_b_model.Reactive_E2E.MapBEVFusion.alpha.fill_(0.5)
    nuplan_batch = _stage_batch(
        device,
        include_bev=True,
        batch_size=2,
        views=8,
    )
    nuplan_batch["sample_uid"] = ["nuplan-a", "nuplan-b"]
    l2d_batch = _stage_batch(
        device,
        include_bev=False,
        batch_size=2,
        views=6,
    )
    l2d_batch["sample_uid"] = ["l2d-a", "l2d-b"]

    matrix = evaluate_reactive_transfer_matrix_models(
        stage_a_model,
        stage_b_model,
        {
            "nuplan": lambda: [nuplan_batch],
            "l2d": lambda: [l2d_batch],
        },
        device=device,
    )

    assert set(matrix) == {"stage_a", "stage_b"}
    assert set(matrix["stage_a"]) == {"nuplan", "l2d"}
    assert matrix["stage_a"]["nuplan"]["sample_uid_sha256"] == (
        matrix["stage_b"]["nuplan"]["sample_uid_sha256"]
    )
    assert matrix["stage_a"]["l2d"]["sample_uid_sha256"] == (
        matrix["stage_b"]["l2d"]["sample_uid_sha256"]
    )
    assert matrix["stage_a"]["nuplan"]["bev_segmentation"][
        "available"
    ]
    assert not matrix["stage_b"]["l2d"]["bev_segmentation"]["available"]


def test_checkpoint_selection_evaluation_skips_auxiliary_heads(
    build_mock_model,
    device,
):
    model = _model(build_mock_model, device).eval()
    batch = _stage_batch(device, include_bev=True)
    calls = {"bev": 0, "route": 0}

    def record_bev(_module, _inputs, _output):
        calls["bev"] += 1

    def record_route(_module, _inputs, _output):
        calls["route"] += 1

    bev_handle = model.Reactive_E2E.BEVSegmentationHead.register_forward_hook(
        record_bev
    )
    route_handle = (
        model.Reactive_E2E.RouteReconstructionHead.register_forward_hook(
            record_route
        )
    )
    try:
        metrics = evaluate_reactive_xy(
            model,
            [batch],
            device=device,
        )
    finally:
        bev_handle.remove()
        route_handle.remove()

    assert metrics["ade_6p4s_m"] >= 0.0
    assert metrics["fde_6p4s_m"] >= 0.0
    assert calls == {"bev": 0, "route": 0}


def test_stage_b_rejects_stage_a_without_bevformer_provenance(
    build_mock_model,
    device,
    tmp_path,
):
    model = _model(build_mock_model, device)
    checkpoint_path = tmp_path / "legacy-stage-a.pt"
    save_reactive_checkpoint(
        checkpoint_path,
        model,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        dataset_manifest_sha256="a" * 64,
        epoch=1,
        model_config={
            "trajectory_weight": 1.0,
            "bev_weight": 1.0,
            "route_weight": 1.0,
            "corridor_pos_weight": 1.0,
            "training_seed": 149,
            "scheduler_identity": "selection_plateau_v1",
            "overfit_bev_only": False,
            "overfit_fixed_lr": False,
            "bev_pos_weights": [2.0] * 8,
            "bev_repeat_factors": [1] * 8,
            "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
            "is_pretrained": False,
            "allow_random_bevformer_init": False,
        },
    )

    with pytest.raises(
        ValueError,
        match="lacks BEVFormer initialization provenance",
    ):
        load_stage_a_parent(model, checkpoint_path)


def test_stage_b_inherits_bevformer_initialization_provenance(
    build_mock_model,
    device,
    tmp_path,
):
    model = _model(build_mock_model, device)
    checkpoint_path = tmp_path / "pretrained-stage-a.pt"
    source_sha256 = "b" * 64
    initialization = {
        "source_sha256": source_sha256,
        "source_url": "https://example.invalid/bevformer.pth",
        "source_repository": "https://example.invalid/repository",
        "weight_license_spdx": "NOASSERTION",
        "training_data_license_spdx": "CC-BY-NC-SA-4.0",
    }
    save_reactive_checkpoint(
        checkpoint_path,
        model,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        dataset_manifest_sha256="a" * 64,
        epoch=1,
        model_config={
            "trajectory_weight": 1.0,
            "bev_weight": 1.0,
            "route_weight": 1.0,
            "corridor_pos_weight": 1.0,
            "training_seed": 149,
            "scheduler_identity": "selection_plateau_v1",
            "overfit_bev_only": False,
            "overfit_fixed_lr": False,
            "bev_pos_weights": [2.0] * 8,
            "bev_repeat_factors": [1] * 8,
            "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
            "bevformer_v2_initialization": initialization,
            "is_pretrained": True,
            "allow_random_bevformer_init": False,
        },
        lineage={
            "bevformer_v2_parent_checkpoint_sha256": source_sha256,
        },
    )

    inherited = load_stage_a_parent(model, checkpoint_path)

    assert inherited["bevformer_v2_initialization"] == initialization
    assert (
        inherited["bevformer_v2_parent_checkpoint_sha256"]
        == source_sha256
    )


def test_stage_b_rejects_stage_a_that_did_not_optimize_bev(
    build_mock_model,
    device,
    tmp_path,
):
    model = _model(build_mock_model, device)
    checkpoint_path = tmp_path / "bev-disabled-stage-a.pt"
    save_reactive_checkpoint(
        checkpoint_path,
        model,
        stage=ReactiveTrainingStage.NUPLAN_FULL,
        dataset_manifest_sha256="a" * 64,
        epoch=1,
        model_config={
            "trajectory_weight": 1.0,
            "bev_weight": 0.0,
            "route_weight": 1.0,
            "corridor_pos_weight": 1.0,
            "training_seed": 149,
            "scheduler_identity": "selection_plateau_v1",
            "overfit_bev_only": False,
            "overfit_fixed_lr": False,
            "bev_pos_weights": [2.0] * 8,
            "bev_repeat_factors": [1] * 8,
            "bev_taxonomy_version": BEV_SEGMENTATION_TAXONOMY_VERSION,
        },
    )

    with pytest.raises(ValueError, match="objective provenance"):
        load_stage_a_parent(model, checkpoint_path)
