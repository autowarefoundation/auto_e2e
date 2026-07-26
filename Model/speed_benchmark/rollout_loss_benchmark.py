"""Benchmark action-only and rollout-aligned loss forward/backward cost."""

from __future__ import annotations

import argparse
import json

import torch

from model_components.losses import TrajectoryImitationLoss
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY
from training.losses import RolloutAlignedLoss


def _measure(
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

    def action_only() -> torch.Tensor:
        return action_loss(predicted, target)

    def treatment() -> torch.Tensor:
        terms = aligned_loss(
            predicted,
            target,
            initial_speed,
            route_supervision,
            map_valid,
            route_valid,
        )
        return (
            action_loss(predicted, target)
            + 0.5 * terms["rollout"]
            + 0.05 * terms["constraint"]
        )

    baseline_ms = _measure(
        action_only,
        predicted,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    treatment_ms = _measure(
        treatment,
        predicted,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps({
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "action_only_forward_backward_ms": baseline_ms,
        "rollout_aligned_forward_backward_ms": treatment_ms,
        "loss_only_regression_percent": (
            (treatment_ms / baseline_ms - 1.0) * 100.0
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
