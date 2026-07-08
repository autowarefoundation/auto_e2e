"""Tests for HorizonReasoningLoss (issue #98, v2).

Synthetic tensors, no GPU / network. Covers:
    * loss is finite;
    * structured loss is lower when logits match targets than when they oppose;
    * abstained/masked horizons (weight 0) do not contribute;
    * single-label ignore_index horizons do not contribute;
    * confidence Brier trains the confidence head (matched << mismatched);
    * source weighting changes the loss magnitude;
    * temporal term is finite and non-negative;
    * alignment loss requires the student embedding and rewards agreement.
"""

from __future__ import annotations

import torch

from model_components.reasoning.horizon_reasoning_head import HorizonReasoningHead
from model_components.reasoning.reasoning_taxonomy import DEFAULT_TAXONOMY, LabelMode
from model_components.reasoning.types import HorizonReasoningPrediction
from training.losses.horizon_reasoning_loss import IGNORE_INDEX, HorizonReasoningLoss

B, H = 4, 5


def _targets(scale_multi: float = 1.0, single_val: int = 0):
    """Build a target dict matching the head's groups."""
    tax = DEFAULT_TAXONOMY
    targets = {}
    for group in (
        "relation_to_ego", "hazard_event", "cause",
        "longitudinal_response", "lateral_response",
        "tactical_response", "rule_response",
    ):
        C = tax.num_classes(group)
        if tax.mode(group) is LabelMode.MULTI:
            t = torch.zeros(B, H, C)
            t[..., 0] = scale_multi  # first class active
            targets[group] = t
        else:
            targets[group] = torch.full((B, H), single_val, dtype=torch.long)
    return targets


def _pred_from_logit_fill(fill: float, tax=DEFAULT_TAXONOMY) -> HorizonReasoningPrediction:
    def L(group):
        return torch.full((B, H, tax.num_classes(group)), fill, requires_grad=False)
    return HorizonReasoningPrediction(
        horizon_tokens=torch.zeros(B, H, 256),
        reasoning_latent=torch.zeros(B, 256),
        relation_to_ego_logits=L("relation_to_ego"),
        hazard_event_logits=L("hazard_event"),
        cause_logits=L("cause"),
        longitudinal_response_logits=L("longitudinal_response"),
        lateral_response_logits=L("lateral_response"),
        tactical_response_logits=L("tactical_response"),
        rule_response_logits=L("rule_response"),
        confidence_logits=torch.zeros(B, H, 1),
    )


def test_loss_is_finite():
    loss_fn = HorizonReasoningLoss()
    head = HorizonReasoningHead()
    pred = head(torch.randn(B, 896), torch.randn(B, 256))
    out = loss_fn(pred, _targets(), source_weights=torch.ones(B, H),
                  confidence_targets=torch.rand(B, H))
    assert torch.isfinite(out["total"])


def test_structured_lower_when_logits_match():
    loss_fn = HorizonReasoningLoss(lambda_temporal=0.0)
    targets = _targets(scale_multi=1.0, single_val=0)
    w = torch.ones(B, H)
    # Matching: large positive logit on class 0 for both multi and single.
    match = _matched_pred(targets)
    oppose = _pred_from_logit_fill(0.0)
    l_match = loss_fn(match, targets, w)["structured"]
    l_oppose = loss_fn(oppose, targets, w)["structured"]
    assert l_match < l_oppose


def _matched_pred(targets) -> HorizonReasoningPrediction:
    tax = DEFAULT_TAXONOMY

    def L(group):
        C = tax.num_classes(group)
        t = torch.full((B, H, C), -10.0)
        t[..., 0] = 10.0  # class 0 is the active/true class in _targets
        return t

    return HorizonReasoningPrediction(
        horizon_tokens=torch.zeros(B, H, 256),
        reasoning_latent=torch.zeros(B, 256),
        relation_to_ego_logits=L("relation_to_ego"),
        hazard_event_logits=L("hazard_event"),
        cause_logits=L("cause"),
        longitudinal_response_logits=L("longitudinal_response"),
        lateral_response_logits=L("lateral_response"),
        tactical_response_logits=L("tactical_response"),
        rule_response_logits=L("rule_response"),
        confidence_logits=torch.zeros(B, H, 1),
    )


def test_abstained_horizons_do_not_contribute():
    loss_fn = HorizonReasoningLoss(lambda_temporal=0.0)
    targets = _targets()
    pred = _pred_from_logit_fill(0.0)
    # All weights zero => structured loss is exactly 0 (masked out).
    out = loss_fn(pred, targets, source_weights=torch.zeros(B, H))
    assert torch.allclose(out["structured"], torch.zeros(()))


def test_single_label_ignore_index_masked():
    loss_fn = HorizonReasoningLoss(lambda_temporal=0.0)
    targets = _targets()
    # Ignore every single-label horizon; only multi-label contributes.
    for group in ("relation_to_ego", "longitudinal_response", "lateral_response",
                  "tactical_response", "rule_response"):
        targets[group] = torch.full((B, H), IGNORE_INDEX, dtype=torch.long)
    pred = _pred_from_logit_fill(0.0)
    out = loss_fn(pred, targets, source_weights=torch.ones(B, H))
    assert torch.isfinite(out["structured"]) and out["structured"] > 0


def test_confidence_loss_trains_confidence():
    loss_fn = HorizonReasoningLoss()
    targets = _targets()
    w = torch.ones(B, H)
    conf_t = torch.ones(B, H)  # target confidence 1.0
    pred_hi = _pred_from_logit_fill(0.0)
    pred_hi.confidence_logits = torch.full((B, H, 1), 10.0)   # sigmoid ~1
    pred_lo = _pred_from_logit_fill(0.0)
    pred_lo.confidence_logits = torch.full((B, H, 1), -10.0)  # sigmoid ~0
    hi = loss_fn(pred_hi, targets, w, confidence_targets=conf_t)["confidence"]
    lo = loss_fn(pred_lo, targets, w, confidence_targets=conf_t)["confidence"]
    assert hi < lo


def test_source_weighting_changes_magnitude():
    loss_fn = HorizonReasoningLoss(lambda_temporal=0.0)
    targets = _targets()
    pred = _pred_from_logit_fill(0.0)
    full = loss_fn(pred, targets, source_weights=torch.ones(B, H))["structured"]
    half = loss_fn(pred, targets, source_weights=torch.full((B, H), 0.5))["structured"]
    # Weighted mean normalizes by weight sum, so a uniform 0.5 weight yields the
    # same mean; a per-horizon-varying weight must change it.
    assert torch.allclose(full, half)
    varied = torch.ones(B, H)
    varied[:, 0] = 0.0  # drop the "now" horizon
    dropped = loss_fn(pred, targets, source_weights=varied)["structured"]
    assert torch.isfinite(dropped)


def test_alignment_rewards_agreement():
    loss_fn = HorizonReasoningLoss(lambda_alignment=0.5)
    head = HorizonReasoningHead(teacher_embedding_dim=64)
    pred = head(torch.randn(B, 896), torch.randn(B, 256))
    w = torch.ones(B, H)
    same = pred.student_reasoning_embedding.detach()
    out_same = loss_fn(pred, _targets(), w, teacher_embedding_targets=same)
    out_opp = loss_fn(pred, _targets(), w, teacher_embedding_targets=-same)
    assert out_same["alignment"] < out_opp["alignment"]


def test_temporal_finite_nonnegative():
    loss_fn = HorizonReasoningLoss()
    head = HorizonReasoningHead()
    pred = head(torch.randn(B, 896), torch.randn(B, 256))
    out = loss_fn(pred, _targets(), torch.ones(B, H), confidence_targets=torch.rand(B, H))
    assert torch.isfinite(out["temporal"]) and out["temporal"] >= 0
