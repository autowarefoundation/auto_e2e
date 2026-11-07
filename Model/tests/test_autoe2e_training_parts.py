"""Distortion mix, schedule and metric pieces of AutoE2E training, on small synthetic tensors."""

import math

import torch
import torch.nn as nn

from evaluation.autoe2e_metrics import bev_counts, calibrate_thresholds, summarize_bev, trajectory_errors
from training import schedule as sched
from training.distortion_mix import camera_augment, photometric


def _images(batch: int = 2) -> dict:
    # Small images keep the warps cheap; the zoom math is size-independent.
    return {
        "visual_tiles": torch.randn(batch, 6, 3, 768, 1344) * 0.1,
        "camera_history_tiles": torch.randn(batch, 7, 6, 3, 768, 1344) * 0.1,
        "camera_projection_matrix": torch.randn(batch, 6, 3, 4),
        "camera_history_projection_matrix": torch.randn(batch, 7, 6, 3, 4),
        "autobev_depth": torch.rand(batch, 6, 96, 168) * 50,
    }


def test_inactive_samples_pass_camera_augment_and_photometric_unchanged():
    batch = _images()
    generator = torch.Generator().manual_seed(0)
    active = torch.tensor([True, False])
    zoomed = camera_augment(batch, active, generator)
    assert torch.allclose(zoomed["visual_tiles"][1], batch["visual_tiles"][1], atol=1e-5)
    assert torch.equal(zoomed["camera_projection_matrix"][1], batch["camera_projection_matrix"][1])
    assert torch.equal(zoomed["autobev_depth"][1], batch["autobev_depth"][1])
    assert torch.equal(zoomed["autobev_image_window"][1], torch.tensor([[0.0, 0.0, 1344.0, 768.0]] * 6))
    assert not torch.equal(zoomed["camera_projection_matrix"][0], batch["camera_projection_matrix"][0])
    jittered = photometric(batch, active, generator)
    assert torch.allclose(jittered["visual_tiles"][1], batch["visual_tiles"][1], atol=1e-5)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)


def test_schedule_freezes_autobev_then_thaws_with_cleared_state():
    toy = _Toy()
    groups = [sched.Group("autobev/decay", [toy.a.weight], 0.2, True),
              sched.Group("planner/decay", [toy.b.weight], 1.0, False)]
    optimizer = sched.optimizer(groups, 1e-3)
    schedule = sched.Schedule(total_steps=200, stage=2, groups=groups, base_lr=1e-3)
    thaw = schedule.thaw_step
    assert thaw == math.ceil(200 * sched.FREEZE_FRACTION[2])
    schedule.apply(optimizer, 0)
    assert optimizer.param_groups[0]["lr"] == 0.0 and optimizer.param_groups[1]["lr"] > 0.0
    toy.a.weight.grad = torch.ones_like(toy.a.weight)
    toy.b.weight.grad = torch.ones_like(toy.b.weight)
    optimizer.step()
    assert toy.a.weight in optimizer.state
    schedule.apply(optimizer, thaw)
    assert toy.a.weight not in optimizer.state and toy.b.weight in optimizer.state
    assert optimizer.param_groups[0]["lr"] > 0.0
    schedule.autobev_backoff = 2
    before = schedule.factor(thaw + 10, groups[0])
    schedule.autobev_backoff = 0
    assert math.isclose(before * 4, schedule.factor(thaw + 10, groups[0]))
    assert schedule.auxiliary_weight(0) == sched.AUXILIARY_WEIGHT[0]
    assert math.isclose(schedule.auxiliary_weight(199), sched.AUXILIARY_WEIGHT[1])


def test_band_iou_and_calibration():
    target = torch.zeros(1, 8, 220, 160)
    target[0, 0, 100:150, 60:100] = 1.0
    logits = torch.full_like(target, -6.0)
    logits[0, 0, 100:150, 60:100] = 0.0  # probability 0.5 exactly on the true cells
    valid = torch.ones_like(target, dtype=torch.bool)
    counts = bev_counts(logits, target, valid)
    totals = {key: value.sum(0) for key, value in counts.items()}
    thresholds = calibrate_thresholds(totals)
    assert thresholds["drivable_area"] < 0.5
    report = summarize_bev(totals, thresholds)
    assert report["bev/0-30/drivable_area"] == 1.0


def test_trajectory_errors_respect_validity():
    predicted = torch.zeros(2, 64, 2)
    target = torch.zeros(2, 64, 2)
    target[:, :, 0] = 1.0
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[1, 20:] = False
    errors = trajectory_errors(predicted, target, valid)
    assert torch.allclose(errors["ade/1.0s"], torch.tensor([1.0, 1.0]))
    assert math.isnan(float(errors["fde/6.4s"][1])) and float(errors["fde/6.4s"][0]) == 1.0
