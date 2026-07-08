"""Tests for the review fixes on the reasoning supervision PR (#98 / #109).

- MultiTeacher must reject teachers whose label tuples differ in order.
- confidence_brier_loss supervises the per-horizon confidence head (#110 hook).
"""

from __future__ import annotations

import pytest
import torch

from model_components.reasoning.scenario_taxonomy import (
    DEFAULT_TAXONOMY,
    ScenarioTaxonomy,
    TaxonomyGroup,
)
from model_components.reasoning.teachers import DeterministicTeacher, MultiTeacher
from training.losses.reasoning_loss import confidence_brier_loss

B = 2


class TestMultiTeacherLabelOrder:
    def test_same_group_names_but_different_label_order_raises(self):
        tax_a = ScenarioTaxonomy()
        tax_b = ScenarioTaxonomy()
        # Same group names, but reverse the label order of one group.
        man = tax_b["maneuver"]
        tax_b._groups["maneuver"] = TaxonomyGroup(
            name="maneuver", labels=tuple(reversed(man.labels))
        )
        t1 = DeterministicTeacher(taxonomy=tax_a)
        t2 = DeterministicTeacher(taxonomy=tax_b)
        with pytest.raises(ValueError, match="label tuples differ"):
            MultiTeacher([t1, t2])

    def test_identical_taxonomies_accepted(self):
        t1 = DeterministicTeacher()
        t2 = DeterministicTeacher()
        # Must not raise (labels identical in content and order).
        MultiTeacher([t1, t2])


class TestConfidenceBrierLoss:
    def test_zero_for_perfect_confidence(self):
        logits = torch.full((B, 5), 10.0)  # sigmoid ~ 1.0
        target = torch.ones(B, 5)
        assert confidence_brier_loss(logits, target).item() < 1e-3

    def test_penalises_confident_but_wrong(self):
        logits = torch.full((B, 5), 10.0)  # sigmoid ~ 1.0
        target = torch.zeros(B, 5)
        assert confidence_brier_loss(logits, target).item() > 0.9

    def test_reduction_none_shape(self):
        loss = confidence_brier_loss(
            torch.zeros(B, 5), torch.zeros(B, 5), reduction="none"
        )
        assert loss.shape == (B, 5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            confidence_brier_loss(torch.zeros(B, 5), torch.zeros(B, 4))

    def test_backward_flows_to_confidence_head(self):
        logits = torch.zeros(B, 5, requires_grad=True)
        target = torch.full((B, 5), 0.7)
        confidence_brier_loss(logits, target).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_taxonomy_is_importable_default(self):
        # sanity: the default taxonomy still exposes the 3 groups used above.
        assert DEFAULT_TAXONOMY.group_names == ["maneuver", "edge_case", "weather_env"]
