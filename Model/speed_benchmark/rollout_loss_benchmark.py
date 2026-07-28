"""Benchmark action-only and rollout-aligned loss forward/backward cost."""

from __future__ import annotations

import argparse
import json
import time

import torch

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
            "fde_6_4s_m": 4.0 + (index % 17) * 0.02,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
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
        "validation_aggregation_and_selector_ms": selector_ms,
        "validation_sample_count": 3820,
        "validation_scene_count": 40,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
