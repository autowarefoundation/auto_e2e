"""Unit tests for the optional System 2 causal reasoning head (Issue #17).

The module is auxiliary and independent: it does NOT modify AutoE2E's
forward pass or its (trajectory, ego_hidden, future) 3-tuple contract.
Key property under test: gradients flow end-to-end (no torch.no_grad in
the cascade), so the head can shape the shared representation and the
reasoning latent can condition the planner.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_components.causal_reasoning import (
    CAUSAL_CLASSES,
    CausalReasoningModule,
    causal_consistency_loss,
)


class _MockBackbone(nn.Module):
    """Minimal stand-in for Backbone (4 channels-first feature maps),
    self-contained so this test file does not depend on conftest imports."""

    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone_channels = 1440
        self._stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
            nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
            nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
            nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
        ])

    def forward(self, image):
        outs, x = [], image
        for stage in self._stages:
            x = stage(x)
            outs.append(x)
        return outs


EMBED_DIM = 256
B = 4


def test_class_vocabulary():
    assert CAUSAL_CLASSES == (
        "intersection", "pedestrian", "traffic_light", "obstacle", "clear",
    )


def test_output_shapes():
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(B, EMBED_DIM)
    reasoning_latent, decision_logits = module(context)
    assert decision_logits.shape == (B, 5)
    assert reasoning_latent.shape == (B, EMBED_DIM)
    assert torch.isfinite(decision_logits).all()
    assert torch.isfinite(reasoning_latent).all()


def test_custom_latent_dim():
    module = CausalReasoningModule(embed_dim=EMBED_DIM, latent_dim=128)
    context = torch.randn(B, EMBED_DIM)
    reasoning_latent, decision_logits = module(context)
    assert reasoning_latent.shape == (B, 128)
    assert decision_logits.shape == (B, 5)


def test_reason_method_matches_cascade_latent():
    """reason() is the cascade hook — it must expose the same latent the
    decision head consumes."""
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    module.eval()
    context = torch.randn(B, EMBED_DIM)
    with torch.no_grad():
        latent_only = module.reason(context)
        latent_full, _ = module(context)
    assert torch.allclose(latent_only, latent_full)


def test_gradients_flow_to_all_parameters_and_context():
    """No torch.no_grad anywhere: gradients must reach every parameter of
    the module AND the upstream context vector (cascade requirement)."""
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(B, EMBED_DIM, requires_grad=True)
    reasoning_latent, decision_logits = module(context)
    labels = torch.randint(0, 5, (B,))
    loss = causal_consistency_loss(decision_logits, labels)
    loss = loss + reasoning_latent.pow(2).mean()
    loss.backward()

    assert context.grad is not None, "Gradient must reach upstream context"
    assert torch.isfinite(context.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_produce_context():
    """Test that the module produces a well-formed SceneContext with
    confidence and provenance, as requested by the architectural review."""
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(B, EMBED_DIM)
    scene_context = module.produce_context(context)
    
    assert scene_context.causal_reasoning is not None
    cr = scene_context.causal_reasoning
    assert cr.reasoning_latent.shape == (B, EMBED_DIM)
    assert cr.causal_class_logits.shape == (B, 5)
    assert cr.confidence.shape == (B,)
    assert (cr.confidence >= 0).all() and (cr.confidence <= 1).all()
    assert cr.provenance == "vlm_causal_head"


def test_loss_lower_for_correct_labels():
    """Cross-entropy must be lower for labels matching the argmax decisions
    than for deliberately wrong labels."""
    torch.manual_seed(0)
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(B, EMBED_DIM)
    _, decision_logits = module(context)
    correct = decision_logits.argmax(dim=1)
    wrong = (correct + 1) % 5
    loss_correct = causal_consistency_loss(decision_logits, correct)
    loss_wrong = causal_consistency_loss(decision_logits, wrong)
    assert loss_correct.item() < loss_wrong.item()


def test_loss_decreases_with_training():
    """A few optimisation steps on fixed pseudo-labels must reduce the
    consistency loss (head is trainable end-to-end)."""
    torch.manual_seed(0)
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(16, EMBED_DIM)
    labels = torch.randint(0, 5, (16,))
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    _, logits = module(context)
    initial = causal_consistency_loss(logits, labels).item()
    for _ in range(30):
        optimizer.zero_grad()
        _, logits = module(context)
        loss = causal_consistency_loss(logits, labels)
        loss.backward()
        optimizer.step()
    final = causal_consistency_loss(module(context)[1], labels).item()
    assert final < initial


def test_label_smoothing_changes_loss_and_keeps_gradients():
    """With label_smoothing > 0 the loss must differ from the unsmoothed one
    (the VLM pseudo-labels are noisy, so smoothing is the intended regime)
    and must remain differentiable end-to-end."""
    torch.manual_seed(0)
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    context = torch.randn(B, EMBED_DIM, requires_grad=True)
    _, decision_logits = module(context)
    labels = torch.randint(0, 5, (B,))

    plain = causal_consistency_loss(decision_logits, labels)
    smoothed = causal_consistency_loss(
        decision_logits, labels, label_smoothing=0.1,
    )
    assert not torch.isclose(plain, smoothed), (
        "label_smoothing > 0 must change the loss value"
    )

    smoothed.backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"


def test_class_weights_change_loss():
    """Optional per-class weights (e.g. to upweight rare long-tail classes)
    must be honoured by the loss."""
    torch.manual_seed(0)
    module = CausalReasoningModule(embed_dim=EMBED_DIM)
    _, decision_logits = module(torch.randn(B, EMBED_DIM))
    labels = torch.randint(0, 5, (B,))
    plain = causal_consistency_loss(decision_logits, labels)
    weighted = causal_consistency_loss(
        decision_logits, labels,
        class_weights=torch.tensor([4.0, 1.0, 1.0, 1.0, 0.5]),
    )
    assert not torch.isclose(plain, weighted)


def test_autoe2e_contract_and_reasoning_head(device):
    """Post-refactor (#86/#88) AutoE2E.forward returns the trajectory only — the
    planner's internal ego embedding is no longer surfaced. This verifies the
    new contract, and exercises the (independent) CausalReasoningModule on a
    256-d embedding so its auxiliary loss stays differentiable."""
    from unittest.mock import patch

    from model_components.auto_e2e import AutoE2E

    with patch("model_components.reactive_e2e.Backbone", _MockBackbone):
        model = AutoE2E(num_views=8,
                        view_fusion_kwargs={"bev_h": 8, "bev_w": 8}).to(device)

    x = torch.randn(2, 8, 3, 256, 256, device=device)
    map_input = torch.randn(2, 3, 256, 256, device=device)
    vis = torch.randn(2, 896, device=device)
    ego = torch.randn(2, 256, device=device)
    # forward(camera_tiles, map_input, visual_history, egomotion_history)
    out = model(x, map_input, vis, ego, mode="infer")

    # New contract: a single trajectory tensor [B, num_timesteps*num_signals].
    assert torch.is_tensor(out) and out.shape == (2, 128)

    # The auxiliary reasoning head consumes a 256-d embedding; its consistency
    # loss must remain differentiable wrt that embedding.
    head = CausalReasoningModule(embed_dim=EMBED_DIM).to(device)
    embedding = torch.randn(2, EMBED_DIM, device=device, requires_grad=True)
    _, decision_logits = head(embedding)
    labels = torch.randint(0, 5, (2,), device=device)
    causal_consistency_loss(decision_logits, labels).backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
