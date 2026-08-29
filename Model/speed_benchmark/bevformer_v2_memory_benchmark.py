"""Measure production BEVFormer V2 forward/backward CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model_components.auto_e2e import AutoE2E
from model_components.bevformer_v2_pretrained import (
    BEVFORMER_V2_T8_CHECKPOINT_SHA256,
    load_bevformer_v2_t8_checkpoint,
)
from navigation.geometry import (
    AUTOE2E_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)
from reactive_training_contracts import (
    REACTIVE_BEVFORMER_HISTORY_FRAMES,
    REACTIVE_CAMERA_IMAGE_SIZE,
    REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
)
from training.reactive_multitask import (
    ReactiveTrainingStage,
    reactive_model_kwargs,
)


def _inputs(
    *,
    batch_size: int,
    num_views: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    geometry = AUTOE2E_NAVIGATION_GEOMETRY
    return {
        "camera_tiles": torch.randn(
            batch_size,
            num_views,
            3,
            REACTIVE_CAMERA_IMAGE_SIZE,
            REACTIVE_CAMERA_IMAGE_SIZE,
            device=device,
        ),
        "camera_history_tiles": torch.randn(
            batch_size,
            REACTIVE_BEVFORMER_HISTORY_FRAMES,
            num_views,
            3,
            REACTIVE_CAMERA_IMAGE_SIZE,
            REACTIVE_CAMERA_IMAGE_SIZE,
            device=device,
        ),
        "front_camera_tile": torch.randn(
            batch_size,
            3,
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            REACTIVE_FRONT_CAMERA_IMAGE_SIZE,
            device=device,
        ),
        "map_context": torch.rand(
            batch_size,
            MAP_CHANNEL_COUNT,
            geometry.height_px,
            geometry.width_px,
            device=device,
        ),
        "route_mask": torch.rand(
            batch_size,
            ROUTE_CHANNEL_COUNT,
            geometry.height_px,
            geometry.width_px,
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
    }


def run_benchmark(
    checkpoint_path: str | Path,
    *,
    batch_size: int = 1,
    num_views: int = 8,
    backward: bool = True,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("BEVFormer memory benchmark requires CUDA")
    if batch_size <= 0 or num_views <= 0:
        raise ValueError("batch_size and num_views must be positive")
    device = torch.device("cuda")
    model = AutoE2E(
        backbone="res_net_50",
        embed_dim=256,
        is_pretrained=False,
        **reactive_model_kwargs(
            ReactiveTrainingStage.NUPLAN_FULL,
            num_views=num_views,
        ),
    ).to(device)
    initialization = load_bevformer_v2_t8_checkpoint(
        model,
        checkpoint_path,
        expected_sha256=BEVFORMER_V2_T8_CHECKPOINT_SHA256,
    )
    view_fusion = model.Reactive_E2E.FeatureFusion.view_fusion
    model.train(backward)
    inputs = _inputs(
        batch_size=batch_size,
        num_views=num_views,
        device=device,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        trajectory, auxiliary = model(
            **inputs,
            mode="train",
            compute_bev_segmentation=True,
            compute_route_reconstruction=True,
        )
        loss = trajectory.float().square().mean()
        loss = loss + sum(
            value.float().square().mean()
            for value in auxiliary.values()
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    backward_seconds = None
    if backward:
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
    gib = 1024**3
    return {
        "backward": backward,
        "backward_seconds": backward_seconds,
        "batch_size": batch_size,
        "bev_latent_shape": [
            int(view_fusion.bev_h),
            int(view_fusion.bev_w),
        ],
        "cuda_device": torch.cuda.get_device_name(device),
        "forward_seconds": forward_seconds,
        "initialization_source_sha256": initialization.source_sha256,
        "loss": float(loss.detach().cpu().item()),
        "num_views": num_views,
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / gib
        ),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "supervision_shape": [
            AUTOE2E_NAVIGATION_GEOMETRY.height_px,
            AUTOE2E_NAVIGATION_GEOMETRY.width_px,
        ],
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        args.checkpoint,
        batch_size=args.batch_size,
        num_views=args.num_views,
        backward=not args.forward_only,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
