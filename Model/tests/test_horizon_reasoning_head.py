"""Tests for HorizonReasoningHead (issue #98, v2).

Synthetic tensors, no GPU / network / teacher. Covers:
    * output shapes: horizon_tokens [B,5,256], reasoning_latent [B,256], every
      structured logit [B,5,C], confidence [B,5,1];
    * optional route/map tokens change N_context but not output shapes;
    * gradient flows to all head + decoder params;
    * no teacher module is importable from the head;
    * teacher-embedding alignment head is off by default, on when requested.
"""

from __future__ import annotations

import torch

from model_components.reasoning.horizon_reasoning_head import HorizonReasoningHead
from model_components.reasoning.reasoning_taxonomy import DEFAULT_TAXONOMY


def _head(**kw) -> HorizonReasoningHead:
    return HorizonReasoningHead(**kw)


def test_output_shapes():
    B = 3
    head = _head()
    pred = head(torch.randn(B, 896), torch.randn(B, 256))
    assert pred.horizon_tokens.shape == (B, 5, 256)
    assert pred.reasoning_latent.shape == (B, 256)
    assert pred.confidence_logits.shape == (B, 5, 1)
    assert pred.relation_to_ego_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("relation_to_ego"))
    assert pred.hazard_event_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("hazard_event"))
    assert pred.cause_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("cause"))
    assert pred.longitudinal_response_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("longitudinal_response"))
    assert pred.lateral_response_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("lateral_response"))
    assert pred.tactical_response_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("tactical_response"))
    assert pred.rule_response_logits.shape == (B, 5, DEFAULT_TAXONOMY.num_classes("rule_response"))


def test_optional_context_does_not_change_output_shape():
    B = 2
    head = _head(route_context_dim=16, map_context_dim=32)
    pred = head(
        torch.randn(B, 896), torch.randn(B, 256),
        route_context=torch.randn(B, 16), map_context=torch.randn(B, 32),
    )
    assert pred.horizon_tokens.shape == (B, 5, 256)
    # Works too if the optional inputs are omitted at call time.
    pred2 = head(torch.randn(B, 896), torch.randn(B, 256))
    assert pred2.horizon_tokens.shape == (B, 5, 256)


def test_gradient_flows_to_heads_and_decoder():
    head = _head()
    pred = head(torch.randn(4, 896), torch.randn(4, 256))
    loss = (
        pred.cause_logits.sum()
        + pred.reasoning_latent.sum()
        + pred.confidence_logits.sum()
    )
    loss.backward()
    grads = {n: p.grad for n, p in head.named_parameters() if p.requires_grad}
    assert any("heads.cause" in n and g is not None and g.abs().sum() > 0 for n, g in grads.items())
    assert any("horizon_queries" in n and g is not None and g.abs().sum() > 0 for n, g in grads.items())
    assert any("decoder" in n and g is not None and g.abs().sum() > 0 for n, g in grads.items())


def test_no_teacher_import_in_head_module():
    import model_components.reasoning.horizon_reasoning_head as m
    src = open(m.__file__).read()
    for forbidden in ("teachers", "openai", "qwen", "videollama", "cosmos", "urllib.request"):
        assert forbidden not in src.lower(), f"head must not reference '{forbidden}'"


def test_teacher_embedding_head_off_by_default():
    head = _head()
    pred = head(torch.randn(2, 896), torch.randn(2, 256))
    assert pred.student_reasoning_embedding is None


def test_teacher_embedding_head_on_when_requested():
    head = _head(teacher_embedding_dim=512)
    pred = head(torch.randn(2, 896), torch.randn(2, 256))
    assert pred.student_reasoning_embedding is not None
    assert pred.student_reasoning_embedding.shape == (2, 5, 512)
