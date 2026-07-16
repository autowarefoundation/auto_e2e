"""End-to-end reasoning training-step smoke (issue #98).

The unit tests prove each part in isolation; this proves the parts actually
connect into a training step that produces a finite combined loss which
DECREASES under optimization — i.e. reasoning loss really flows to the head.

Mock backbone + mock teacher, no GPU / network / shards. Mirrors the wiring in
the Flyte train_il task: model(mode="train") → (traj, aux) → HorizonReasoningLoss
on aux["reasoning_pred"] against tensorized mock-teacher labels.
"""

from __future__ import annotations

import torch

from data_processing.reasoning_label_generation.mock_teacher import MockTeacher
from data_processing.reasoning_label_generation.targets import (
    collate_reasoning_targets,
    record_to_target_tensors,
)
from data_processing.reasoning_label_generation.teacher_client import TeacherRequest
from training.losses.horizon_reasoning_loss import HorizonReasoningLoss

NUM_VIEWS = 7
B = 3


def _batch_inputs():
    return (
        torch.randn(B, NUM_VIEWS, 3, 256, 256),
        torch.randn(B, 3, 256, 256),
        torch.randn(B, 896),
        torch.randn(B, 256),
        torch.randn(B, 128),  # trajectory_target
    )


def _target_batch():
    teacher = MockTeacher()
    per_sample = [
        record_to_target_tensors(teacher.label(TeacherRequest(f"s{i}", "l2d")))
        for i in range(B)
    ]
    return collate_reasoning_targets(per_sample)


def test_reasoning_targets_have_signal():
    tb = _target_batch()
    # Source weights are > 0 (mock provenance is teacher_gt=0.5 × conf∈[0.5,1]).
    assert tb.source_weights.min() > 0
    # Multi-label cause target has at least one active class per horizon.
    assert tb.targets["cause"].sum() > 0


def test_full_train_step_combined_loss_decreases(build_mock_model):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    model.train()
    traj_loss_fn = torch.nn.SmoothL1Loss()
    reason_loss_fn = HorizonReasoningLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    visual, map_input, vis_hist, ego, target = _batch_inputs()
    tb = _target_batch()

    def step():
        opt.zero_grad()
        out = model(visual, map_input, vis_hist, ego, mode="train",
                    trajectory_target=target)
        trajectory, aux = out
        loss = traj_loss_fn(trajectory, target)
        terms = reason_loss_fn(
            aux["reasoning_pred"], tb.targets,
            source_weights=tb.source_weights,
            confidence_targets=tb.confidence_targets,
        )
        total = loss + 0.5 * terms["total"]
        total.backward()
        opt.step()
        return float(total.detach()), float(terms["total"].detach())

    first_total, first_reason = step()
    assert torch.isfinite(torch.tensor(first_total))
    assert first_reason > 0  # reasoning loss is a real, non-trivial signal

    # A handful of steps on the SAME batch must reduce the combined loss —
    # confirming gradients from the reasoning loss actually reach the head.
    last_total = first_total
    for _ in range(15):
        last_total, _ = step()
    assert last_total < first_total


def test_reasoning_grad_reaches_head(build_mock_model):
    device = torch.device("cpu")
    model = build_mock_model(
        num_views=NUM_VIEWS, device=device,
        enable_reasoning=True, reasoning_mode="pooled_latent",
    )
    model.train()
    reason_loss_fn = HorizonReasoningLoss()
    visual, map_input, vis_hist, ego, target = _batch_inputs()
    tb = _target_batch()

    _, aux = model(visual, map_input, vis_hist, ego, mode="train",
                   trajectory_target=target)
    terms = reason_loss_fn(
        aux["reasoning_pred"], tb.targets,
        source_weights=tb.source_weights, confidence_targets=tb.confidence_targets,
    )
    terms["total"].backward()

    head = model.Reactive_E2E.ReasoningHead
    grads = [p.grad for n, p in head.named_parameters()
             if "heads.cause" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
