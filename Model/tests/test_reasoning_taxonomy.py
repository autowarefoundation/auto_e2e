"""Tests for the compositional reasoning taxonomy (issue #98, v2).

Pure-Python, no torch/GPU/network. Covers:
    * all seven action-relevant v1 groups exist with the right mode;
    * every group has an abstain label (unknown_* / no_* / none);
    * index order is stable and append-only (extend keeps existing indices);
    * register_group is append-only (duplicate name rejected);
    * validate_exact_match passes for identical taxonomies and fails on any
      difference in groups / order / labels / mode (R10).
"""

from __future__ import annotations

import pytest

from model_components.reasoning.reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    LabelMode,
    ReasoningTaxonomy,
    TaxonomyGroup,
)

_EXPECTED_GROUPS = {
    "relation_to_ego": LabelMode.SINGLE,
    "hazard_event": LabelMode.MULTI,
    "cause": LabelMode.MULTI,
    "longitudinal_response": LabelMode.SINGLE,
    "lateral_response": LabelMode.SINGLE,
    "tactical_response": LabelMode.SINGLE,
    "rule_response": LabelMode.SINGLE,
}


def test_all_v1_groups_exist_with_correct_mode():
    tax = ReasoningTaxonomy()
    assert tax.group_names() == list(_EXPECTED_GROUPS.keys())
    for name, mode in _EXPECTED_GROUPS.items():
        assert tax.mode(name) is mode


def test_every_group_has_abstain_label():
    tax = ReasoningTaxonomy()
    for name in tax.group_names():
        labels = tax.labels(name)
        assert any(
            l.startswith("unknown") or l.startswith("no_") or l == "none"
            for l in labels
        ), f"group '{name}' lacks an abstain label"


def test_index_is_stable_and_lookup_roundtrips():
    tax = ReasoningTaxonomy()
    for name in tax.group_names():
        for i, label in enumerate(tax.labels(name)):
            assert tax.index(name, label) == i


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        TaxonomyGroup(name="bad", labels=("a", "a", "unknown_x"))


def test_group_without_abstain_label_rejected():
    with pytest.raises(ValueError, match="abstain"):
        TaxonomyGroup(name="bad", labels=("a", "b"))


def test_extend_is_append_only_and_preserves_indices():
    tax = ReasoningTaxonomy()
    before = {l: i for i, l in enumerate(tax.labels("cause"))}
    tax.extend("cause", ["debris_on_road"])
    # existing indices unchanged
    for label, idx in before.items():
        assert tax.index("cause", label) == idx
    # new label appended at the end
    assert tax.index("cause", "debris_on_road") == len(before)


def test_extend_rejects_overlapping_labels():
    tax = ReasoningTaxonomy()
    with pytest.raises(ValueError, match="already exist"):
        tax.extend("cause", ["red_light"])


def test_register_group_rejects_duplicate_name():
    tax = ReasoningTaxonomy()
    with pytest.raises(ValueError, match="already registered"):
        tax.register_group("cause", ["unknown_x"])


def test_register_new_context_group_appends():
    tax = ReasoningTaxonomy()
    n = len(tax.group_names())
    tax.register_group("road_topology", ["intersection", "roundabout", "unknown_topology"])
    assert tax.group_names()[-1] == "road_topology"
    assert len(tax.group_names()) == n + 1
    assert tax.num_classes("road_topology") == 3


def test_validate_exact_match_passes_for_identical():
    a = ReasoningTaxonomy()
    b = ReasoningTaxonomy()
    a.validate_exact_match(b)  # no raise


def test_validate_exact_match_fails_on_extra_group():
    a = ReasoningTaxonomy()
    b = ReasoningTaxonomy()
    b.register_group("extra", ["x", "unknown_x"])
    with pytest.raises(ValueError, match="group mismatch"):
        a.validate_exact_match(b)


def test_validate_exact_match_fails_on_reordered_labels():
    a = ReasoningTaxonomy()
    b = ReasoningTaxonomy()
    # Rebuild b's cause group with a reordered label tuple (same set, wrong order).
    reordered = tuple(reversed(a.labels("cause")))
    b._groups["cause"] = TaxonomyGroup(name="cause", labels=reordered, mode=LabelMode.MULTI)
    with pytest.raises(ValueError, match="label mismatch"):
        a.validate_exact_match(b)


def test_default_taxonomy_is_shared_instance():
    assert isinstance(DEFAULT_TAXONOMY, ReasoningTaxonomy)
    assert DEFAULT_TAXONOMY.num_classes("cause") == 27
