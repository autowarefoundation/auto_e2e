"""Reactive-only nuPlan/L2D multi-task contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data_processing.reactive_training_artifacts import (
    decode_bev_segmentation,
    decode_trajectory_xy,
    encode_bev_segmentation,
    encode_trajectory_xy,
)
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
)
from training.reactive_stage_runner import (
    evaluate_reactive_multitask,
    evaluate_reactive_transfer_matrix_models,
    evaluate_reactive_xy,
    load_stage_a_parent,
    run_reactive_epoch,
    save_reactive_checkpoint,
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


def test_route_loss_reaches_gate_but_not_camera(
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

    alpha = model.Reactive_E2E.MapBEVFusion.alpha
    assert alpha.grad is not None
    assert bool((alpha.grad != 0).any())
    assert all(
        parameter.grad is None
        for parameter in model.Reactive_E2E.Backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.Reactive_E2E.FeatureFusion.parameters()
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
        model_config={"num_views": 8},
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
        "lane_area",
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
