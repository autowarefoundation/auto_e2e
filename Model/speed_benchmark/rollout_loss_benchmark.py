"""Benchmark action-only and rollout-aligned loss forward/backward cost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.checkpoint_selection import (
    aggregate_validation_records,
    freeze_component_availability,
    score_checkpoint,
)
from model_components.losses import TrajectoryImitationLoss
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from training.losses import RolloutAlignedLoss
from training.losses.control_rollout import integrate_controls_torch


def _measure_cuda_forward(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _measure_cuda_backward(
    loss_factory,
    predicted: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        predicted.grad = None
        loss_factory().backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        predicted.grad = None
        loss_factory().backward()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _measure_cuda_training_step(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _selector_records(
    *,
    sample_count: int = 3820,
    scene_count: int = 40,
) -> list[dict[str, object]]:
    records = []
    for index in range(sample_count):
        records.append({
            "sample_uid": f"sample-{index:06d}",
            "split_group_uid": f"scene-{index % scene_count:03d}",
            "ade_3s_m": 1.0 + (index % 11) * 0.01,
            "fde_3s_m": 2.0 + (index % 17) * 0.01,
            "comfort_excess": 0.01,
            "offroad_excess": 0.02,
            "route_gap": 0.03,
            "wrong_branch_excess": 0.04,
            "destination_error_m": 2.0,
            "diagnostic_target_offroad_rate": 0.1,
            "diagnostic_target_route_compliance": 0.7,
            "diagnostic_raster_tolerance_m": 0.5,
        })
    return records


def _measure_selector(
    *,
    warmup: int,
    iterations: int,
) -> float:
    records = _selector_records()

    def operation() -> None:
        aggregates = aggregate_validation_records(records)
        availability = freeze_component_availability(aggregates)
        score_checkpoint(aggregates, availability)

    for _ in range(warmup):
        operation()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _measure_full_model_steps(
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    from model_components.auto_e2e import AutoE2E
    from model_components.reasoning.reasoning_taxonomy import (
        DEFAULT_TAXONOMY,
        LabelMode,
    )
    from training.losses.horizon_reasoning_loss import (
        HorizonReasoningLoss,
    )

    model = AutoE2E(
        backbone="swin_v2_tiny",
        num_views=6,
        is_pretrained=False,
        view_fusion_kwargs=(
            DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
        ),
        map_context_channels=14,
        route_channels=2,
        enable_route_conditioning=True,
        enable_world_model=True,
        enable_reasoning=True,
        reasoning_mode="pooled_latent",
    ).to(device)
    model.train()
    world_model = model.World_Action_Model_E2E
    if world_model is None:
        raise RuntimeError("full-step benchmark requires the World Model")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=5e-5,
    )
    generator = torch.Generator(device=device).manual_seed(152)
    visual = torch.randn(
        batch_size,
        6,
        3,
        256,
        256,
        generator=generator,
        device=device,
    )
    map_context = torch.randn(
        batch_size,
        14,
        256,
        256,
        generator=generator,
        device=device,
    )
    route_mask = torch.zeros(
        batch_size,
        2,
        256,
        256,
        device=device,
    )
    visual_history = torch.randn(
        batch_size,
        896,
        generator=generator,
        device=device,
    )
    egomotion_history = torch.randn(
        batch_size,
        256,
        generator=generator,
        device=device,
    )
    egomotion_history[:, -4] = 8.0
    history_frames = torch.randn(
        batch_size,
        4,
        6,
        3,
        256,
        256,
        generator=generator,
        device=device,
    )
    future_frames = torch.randn(
        batch_size,
        4,
        6,
        3,
        256,
        256,
        generator=generator,
        device=device,
    )
    target = torch.randn(
        batch_size,
        64,
        2,
        generator=generator,
        device=device,
    ) * torch.tensor([0.25, 0.01], device=device)
    initial_speed = torch.full((batch_size,), 8.0, device=device)
    logged_positions, _, _ = integrate_controls_torch(
        target,
        initial_speed,
    )
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    field_shape = (
        batch_size,
        geometry.height_px,
        geometry.width_px,
    )
    route_supervision = {
        "distance_to_corridor_m": torch.zeros(
            field_shape,
            device=device,
        ),
        "distance_to_drivable_m": torch.zeros(
            field_shape,
            device=device,
        ),
        "available": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        "drivable_available": torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
    }
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    action_loss = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=0.99,
        temporal_weight_normalization="mean_one",
        signal_scales=(0.778, 0.0350),
    ).to(device)
    aligned_loss = RolloutAlignedLoss().to(device)
    reasoning_loss = HorizonReasoningLoss()
    reasoning_targets = {}
    for group in DEFAULT_TAXONOMY.groups:
        shape = (batch_size, 5, len(group))
        if group.mode is LabelMode.MULTI:
            target_values = torch.zeros(shape, device=device)
            target_values[:, :, -1] = 1.0
        else:
            target_values = torch.zeros(
                batch_size,
                5,
                dtype=torch.long,
                device=device,
            )
        reasoning_targets[group.name] = target_values
    reasoning_weights = torch.ones(batch_size, 5, device=device)
    reasoning_confidence = torch.ones(batch_size, 5, device=device)

    def step(*, aligned: bool) -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(
            visual,
            map_context,
            visual_history,
            egomotion_history,
            route_mask=route_mask,
            map_valid=valid,
            route_valid=valid,
            geometry_type="pseudo",
            mode="train",
            trajectory_target=target,
            history_frames=history_frames,
            future_frames=future_frames,
        )
        predicted, auxiliary = output
        loss = action_loss(predicted, target)
        future_state = auxiliary["future_state_pred"]
        loss = loss + world_model.jepa_loss(
            future_state,
            future_frames,
        )
        reasoning_terms = reasoning_loss(
            auxiliary["reasoning_pred"],
            reasoning_targets,
            source_weights=reasoning_weights,
            confidence_targets=reasoning_confidence,
        )
        loss = loss + 0.05 * reasoning_terms["total"]
        if aligned:
            terms = aligned_loss(
                predicted,
                target,
                initial_speed,
                logged_positions,
                route_supervision,
                valid,
                valid,
            )
            loss = (
                loss
                + 0.5 * terms["rollout"]
                + 0.05 * terms["constraint"]
            )
        loss.backward()
        optimizer.step()

    baseline_ms = _measure_cuda_training_step(
        lambda: step(aligned=False),
        warmup=warmup,
        iterations=iterations,
    )
    treatment_ms = _measure_cuda_training_step(
        lambda: step(aligned=True),
        warmup=warmup,
        iterations=iterations,
    )
    return baseline_ms, treatment_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--full-step-batch-size", type=int, default=1)
    parser.add_argument("--full-step-warmup", type=int, default=3)
    parser.add_argument("--full-step-iterations", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    geometry = DEFAULT_NAVIGATION_GEOMETRY
    generator = torch.Generator(device=device).manual_seed(149)
    target = torch.randn(
        args.batch_size,
        64,
        2,
        generator=generator,
        device=device,
    ) * torch.tensor([0.25, 0.01], device=device)
    predicted = (
        target
        + torch.randn(
            target.shape,
            generator=generator,
            device=device,
        ) * torch.tensor([0.05, 0.002], device=device)
    ).detach().requires_grad_(True)
    initial_speed = torch.full(
        (args.batch_size,),
        8.0,
        device=device,
    )
    shape = (
        args.batch_size,
        geometry.height_px,
        geometry.width_px,
    )
    route_supervision = {
        "distance_to_corridor_m": torch.zeros(shape, device=device),
        "distance_to_drivable_m": torch.zeros(shape, device=device),
        "available": torch.ones(
            args.batch_size,
            device=device,
            dtype=torch.bool,
        ),
        "drivable_available": torch.ones(
            args.batch_size,
            device=device,
            dtype=torch.bool,
        ),
    }
    map_valid = torch.ones(
        args.batch_size,
        device=device,
        dtype=torch.bool,
    )
    route_valid = map_valid.clone()
    action_loss = TrajectoryImitationLoss(
        loss_type="smooth_l1",
        temporal_decay=0.99,
        temporal_weight_normalization="mean_one",
        signal_scales=(0.778, 0.0350),
    ).to(device)
    aligned_loss = RolloutAlignedLoss().to(device)
    logged_positions, _, _ = integrate_controls_torch(
        target,
        initial_speed,
    )

    def action_only() -> torch.Tensor:
        return action_loss(predicted, target)

    def rollout_forward() -> torch.Tensor:
        positions, headings, speeds = integrate_controls_torch(
            predicted,
            initial_speed,
        )
        return positions.sum() + headings.sum() + speeds.sum()

    def treatment() -> torch.Tensor:
        terms = aligned_loss(
            predicted,
            target,
            initial_speed,
            logged_positions,
            route_supervision,
            map_valid,
            route_valid,
        )
        return (
            action_loss(predicted, target)
            + 0.5 * terms["rollout"]
            + 0.05 * terms["constraint"]
        )

    rollout_forward_ms = _measure_cuda_forward(
        rollout_forward,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    rollout_forward_backward_ms = _measure_cuda_backward(
        rollout_forward,
        predicted,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    baseline_ms = _measure_cuda_backward(
        action_only,
        predicted,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    treatment_ms = _measure_cuda_backward(
        treatment,
        predicted,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    selector_ms = _measure_selector(
        warmup=min(args.warmup, 3),
        iterations=min(args.iterations, 20),
    )
    full_step_baseline_ms, full_step_treatment_ms = (
        _measure_full_model_steps(
            device=device,
            batch_size=args.full_step_batch_size,
            warmup=args.full_step_warmup,
            iterations=args.full_step_iterations,
        )
    )
    print(json.dumps({
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "rollout_forward_ms": rollout_forward_ms,
        "rollout_forward_backward_ms": rollout_forward_backward_ms,
        "action_only_forward_backward_ms": baseline_ms,
        "rollout_aligned_forward_backward_ms": treatment_ms,
        "loss_only_regression_percent": (
            (treatment_ms / baseline_ms - 1.0) * 100.0
        ),
        "full_step_batch_size": args.full_step_batch_size,
        "full_step_iterations": args.full_step_iterations,
        "full_step_includes_world_model_jepa": True,
        "full_step_includes_reasoning": True,
        "action_only_full_training_step_ms": full_step_baseline_ms,
        "rollout_aligned_full_training_step_ms": (
            full_step_treatment_ms
        ),
        "full_training_step_regression_percent": (
            (full_step_treatment_ms / full_step_baseline_ms - 1.0)
            * 100.0
        ),
        "validation_aggregation_and_selector_ms": selector_ms,
        "validation_sample_count": 3820,
        "validation_scene_count": 40,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
