"""One full-size training step of each stage on a GPU, with synthetic inputs.

These run the real 1344x768 six-camera T8 model with the pretrained DINOv2, so
they need a GPU and the timm weights; CI skips them (``integration``). They
check that every loss is finite, that gradients reach the parts each stage
trains and not the frozen ones, and print the step time and peak memory used
to size the full runs.
"""

import time

import pytest
import torch
import torch.nn as nn

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


def _batch(batch: int, device) -> dict:
    from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES

    projection = torch.zeros(batch, 6, 3, 4, device=device)
    # Six cameras around the ego at 1.5 m height, looking along their yaw.
    for view, yaw in enumerate((0.0, 0.9, -0.9, 2.2, -2.2, 3.14159)):
        c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        rotation = torch.tensor([[s, -c, 0.0], [0.0, 0.0, -1.0], [c, s, 0.0]])
        intrinsic = torch.tensor([[800.0, 0.0, 672.0], [0.0, 800.0, 384.0], [0.0, 0.0, 1.0]])
        extrinsic = torch.cat([rotation, -rotation @ torch.tensor([[1.0], [0.0], [1.5]])], dim=1)
        projection[:, view] = (intrinsic @ extrinsic).to(device)
    classes = len(BEV_SEGMENTATION_CLASSES)
    boxes = torch.tensor([[12.0, 1.5, 0.8, 4.5, 1.9, 1.6, 0.1, 5.0, 0.0]], device=device)
    trajectory = torch.stack([torch.arange(1, 65, device=device, dtype=torch.float32) * 0.8,
                              torch.zeros(64, device=device)], dim=-1)
    egomotion = torch.zeros(batch, 64, 4, device=device)
    egomotion[..., 0] = 8.0
    return {
        "visual_tiles": torch.randn(batch, 6, 3, 768, 1344, device=device),
        "camera_history_tiles": torch.randn(batch, 7, 6, 3, 768, 1344, device=device),
        "camera_projection_matrix": projection,
        "camera_history_projection_matrix": projection[:, None].expand(-1, 7, -1, -1, -1).clone(),
        "map_context": (torch.rand(batch, 14, 450, 300, device=device) > 0.9).float(),
        "route_mask": (torch.rand(batch, 2, 450, 300, device=device) > 0.9).float(),
        "map_valid": torch.ones(batch, dtype=torch.bool, device=device),
        "route_valid": torch.ones(batch, dtype=torch.bool, device=device),
        "route_channel_valid": torch.ones(batch, 2, dtype=torch.bool, device=device),
        "egomotion_history": egomotion.reshape(batch, -1),
        "initial_speed_mps": torch.full((batch,), 8.0, device=device),
        "trajectory_xy_m": trajectory[None].expand(batch, -1, -1).clone(),
        "trajectory_valid": torch.ones(batch, 64, dtype=torch.bool, device=device),
        "planner_supervised": torch.tensor([True, False][:batch], device=device),
        "bev_segmentation_target": (torch.rand(batch, classes, 220, 160, device=device) > 0.8).float(),
        "bev_segmentation_valid": torch.ones(batch, classes, 220, 160, dtype=torch.bool, device=device),
        "autobev_depth": torch.rand(batch, 6, 96, 168, device=device) * 60,
        "autobev_boxes": [boxes.clone() for _ in range(batch)],
        "autobev_box_labels": [torch.zeros(1, dtype=torch.long, device=device) for _ in range(batch)],
    }


class _Wrapped(nn.Module):
    """Stands in for DistributedDataParallel: the trainer reads ``.module``."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _timed_step(run_step) -> tuple[dict, float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    terms = run_step()
    terms["total"].backward()
    torch.cuda.synchronize()
    return terms, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30


def test_stage1_step():
    from model_components.autobev import AutoBEV
    from training.objective import AutoBEVObjective
    from training.trainer import _stage1_terms

    device = torch.device("cuda")
    model = AutoBEV(pretrained_backbone=True).to(device).train()
    objective = AutoBEVObjective().to(device)
    generator = torch.Generator().manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        terms, seconds, peak = _timed_step(lambda: _stage1_terms(model, objective, _batch(1, device), generator))
    assert torch.isfinite(terms["total"])
    assert model.backbone.vit.blocks[0].attn.qkv.weight.grad is not None
    assert model.center.heatmap.weight.grad is not None
    print(f"stage 1: {seconds:.2f} s per sample, peak {peak:.1f} GiB")


def test_stage2_step():
    from model_components import AutoE2E
    from training.objective import AutoE2EObjective
    from training.schedule import Schedule, optimizer, stage2_groups
    from training.trainer import _stage2_terms

    device = torch.device("cuda")
    model = AutoE2E(pretrained_backbone=True).to(device).train()
    groups = stage2_groups(model)
    schedule = Schedule(total_steps=100, stage=2, groups=groups, base_lr=1e-4)
    optimizer(groups, 1e-4)
    objective = AutoE2EObjective().to(device)
    generator = torch.Generator().manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        terms, seconds, peak = _timed_step(
            lambda: _stage2_terms(_Wrapped(model), objective, _batch(2, device), generator, schedule, 0))
    for name in ("total", "trajectory", "route", "pose", "consistency", "autobev"):
        assert torch.isfinite(terms[name]), name
    reactive = model.Reactive_E2E
    assert reactive.TrajectoryPlanner.control_head.weight.grad is not None
    assert reactive.MapBEVFusion.out_proj.weight.grad is not None
    assert reactive.AutoBEV.backbone.vit.blocks[0].attn.qkv.weight.grad is None  # lower DINOv2 frozen
    assert reactive.AutoBEV.backbone.vit.blocks[11].attn.qkv.weight.grad is not None
    print(f"stage 2: {seconds / 2:.2f} s per sample, peak {peak:.1f} GiB")
