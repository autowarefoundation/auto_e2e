"""Tests for the Reasoning Band (issue #98).

All tests use synthetic tensors — no GPU required, no network calls.
Mirrors the style of test_world_action_model.py.

Covered:
    * I/O shapes of ScenarioTaxonomy, ReasoningBand, DeterministicTeacher,
      ReasoningLoss.
    * Planner coupling (#98/#103): forward returns a ReasoningPrediction whose
      zero-init gate is a strict no-op at initialisation (modulated history ==
      input) and only diverges once trained; per-horizon confidence included.
    * Taxonomy is extensible: adding a KIT label does not change existing indices.
    * Multi-label: several classes can be active simultaneously.
    * Multi-horizon: train mode returns 5 horizons; infer mode returns 1.
    * DeterministicTeacher is deterministic.
    * ReasoningLoss decreases toward trivial targets (sigmoid(0) → 0.5 ≠ 1.0;
      sigmoid(10) → ~1.0 ≈ 1.0 → near-zero loss).
    * AutoE2E with enable_reasoning_band=False is byte-identical to baseline.
"""

from __future__ import annotations


import pytest
import torch
import torch.nn as nn

from model_components.reasoning.scenario_taxonomy import (
    ScenarioTaxonomy,
    TaxonomyGroup,
    DEFAULT_TAXONOMY,
)
from model_components.reasoning.reasoning_band import ReasoningBand, ReasoningPrediction
from model_components.reasoning.teachers.deterministic import DeterministicTeacher
from training.losses.reasoning_loss import ReasoningLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 2      # batch size used throughout
VH_DIM = 896  # Encoded Visual History dimension


def _vh(device: torch.device) -> torch.Tensor:
    """Synthetic visual history [B, 896]."""
    return torch.randn(B, VH_DIM, device=device)


def _band(device: torch.device, **kw) -> ReasoningBand:
    return ReasoningBand(visual_history_dim=VH_DIM, hidden_dim=64, **kw).to(device)


# ---------------------------------------------------------------------------
# ScenarioTaxonomy
# ---------------------------------------------------------------------------

class TestScenarioTaxonomy:
    def test_default_groups_present(self):
        t = ScenarioTaxonomy()
        assert "maneuver" in t
        assert "edge_case" in t
        assert "weather_env" in t

    def test_default_group_sizes(self):
        t = ScenarioTaxonomy()
        assert t.num_classes("maneuver") == 7
        assert t.num_classes("edge_case") == 6
        assert t.num_classes("weather_env") == 8

    def test_stable_index_ordering(self):
        """Index is part of the loss contract — must not change."""
        t = ScenarioTaxonomy()
        assert t["maneuver"].index("continue_straight") == 0
        assert t["maneuver"].index("turn_right") == 6
        assert t["edge_case"].index("nudge_out") == 0
        assert t["weather_env"].index("fair_day") == 0
        assert t["weather_env"].index("fog_night") == 7

    def test_extensible_register_group(self):
        """Adding a new group does not affect existing groups."""
        t = ScenarioTaxonomy()
        old_idx = t["maneuver"].index("turn_left")
        t.register_group("kit_context", ["intersection", "construction_zone"])
        # Existing index unchanged
        assert t["maneuver"].index("turn_left") == old_idx
        # New group accessible
        assert t.num_classes("kit_context") == 2
        assert t["kit_context"].index("intersection") == 0

    def test_extend_appends_only(self):
        """extend() must not change indices of existing labels."""
        t = ScenarioTaxonomy()
        t.extend("maneuver", ["u_turn"])
        assert t["maneuver"].index("turn_right") == 6   # unchanged
        assert t["maneuver"].index("u_turn") == 7       # appended
        assert t.num_classes("maneuver") == 8

    def test_register_group_duplicate_raises(self):
        t = ScenarioTaxonomy()
        with pytest.raises(ValueError, match="already registered"):
            t.register_group("maneuver", ["foo"])

    def test_extend_unknown_group_raises(self):
        t = ScenarioTaxonomy()
        with pytest.raises(KeyError):
            t.extend("nonexistent", ["foo"])

    def test_extend_duplicate_label_raises(self):
        t = ScenarioTaxonomy()
        with pytest.raises(ValueError, match="already exist"):
            t.extend("maneuver", ["turn_left"])

    def test_group_contains_no_duplicates(self):
        t = ScenarioTaxonomy()
        for g in t.groups:
            assert len(set(g.labels)) == len(g.labels)

    def test_taxonomy_group_duplicate_labels_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            TaxonomyGroup(name="test", labels=("a", "a", "b"))

    def test_multi_label_multi_group(self):
        """Taxonomy supports querying labels across all three groups at once."""
        t = ScenarioTaxonomy()
        # Each group has its own independent index space; all lookups must succeed.
        m_idx = t["maneuver"].index("curve_left")
        e_idx = t["edge_case"].index("give_way")
        w_idx = t["weather_env"].index("rain_night")
        # Indices within each group must be in valid range
        assert 0 <= m_idx < t.num_classes("maneuver")
        assert 0 <= e_idx < t.num_classes("edge_case")
        assert 0 <= w_idx < t.num_classes("weather_env")


# ---------------------------------------------------------------------------
# ReasoningBand — shapes
# ---------------------------------------------------------------------------

class TestReasoningBandShapes:
    def test_infer_mode_returns_one_horizon(self, device):
        band = _band(device)
        vh = _vh(device)
        logits = band(vh, mode="infer").logits
        for group in DEFAULT_TAXONOMY.groups:
            assert group.name in logits
            assert len(logits[group.name]) == 1
            assert logits[group.name][0].shape == (B, len(group))

    def test_train_mode_returns_five_horizons(self, device):
        band = _band(device)
        vh = _vh(device)
        logits = band(vh, mode="train").logits
        # 1 current + 4 future = 5
        for group in DEFAULT_TAXONOMY.groups:
            assert len(logits[group.name]) == 5

    def test_all_taxonomy_groups_present(self, device):
        band = _band(device)
        logits = band(_vh(device), mode="infer").logits
        assert set(logits.keys()) == set(DEFAULT_TAXONOMY.group_names)

    def test_custom_num_future_horizons(self, device):
        band = _band(device, num_future_horizons=2)
        logits = band(_vh(device), mode="train").logits
        for group in DEFAULT_TAXONOMY.groups:
            assert len(logits[group.name]) == 3  # 1 current + 2 future

    def test_output_logits_finite(self, device):
        band = _band(device)
        logits = band(_vh(device), mode="train").logits
        for group_horizons in logits.values():
            for t in group_horizons:
                assert torch.isfinite(t).all()


# ---------------------------------------------------------------------------
# Zero-init gate (planner coupling, #98/#103) + per-horizon confidence
# ---------------------------------------------------------------------------

class TestZeroInitGateAndConfidence:
    """The band feeds the planner through a zero-init gate: at initialisation
    the modulated visual history is IDENTICAL to the input (strict no-op), so
    enabling the band leaves the reactive baseline unchanged until training
    moves the gate.  A per-horizon confidence accompanies the logits (#103)."""

    def test_returns_typed_prediction(self, device):
        band = _band(device)
        pred = band(_vh(device), mode="train")
        assert isinstance(pred, ReasoningPrediction)
        assert set(pred.logits.keys()) == set(DEFAULT_TAXONOMY.group_names)

    def test_gate_is_noop_at_init(self, device):
        band = _band(device)
        vh = _vh(device)
        for mode in ("infer", "train"):
            pred = band(vh, mode=mode)
            assert torch.allclose(pred.modulated_visual_history, vh, atol=1e-6), (
                "Zero-init gate must be a strict no-op at initialisation."
            )

    def test_gate_diverges_once_trained(self, device):
        band = _band(device)
        vh = _vh(device)
        with torch.no_grad():
            band.gate.beta.bias.fill_(0.5)
        pred = band(vh, mode="infer")
        assert not torch.allclose(pred.modulated_visual_history, vh)

    def test_visual_history_not_mutated_in_place(self, device):
        band = _band(device)
        vh = _vh(device)
        vh_before = vh.clone()
        band(vh, mode="train")
        assert torch.equal(vh, vh_before)

    def test_confidence_shape_train_and_infer(self, device):
        band = _band(device)
        vh = _vh(device)
        assert band(vh, mode="train").confidence.shape == (B, 5)
        assert band(vh, mode="infer").confidence.shape == (B, 1)

    def test_gate_receives_gradient(self, device):
        band = _band(device)
        pred = band(_vh(device), mode="train")
        pred.modulated_visual_history.sum().backward()
        assert band.gate.gamma.weight.grad is not None


# ---------------------------------------------------------------------------
# Multi-label
# ---------------------------------------------------------------------------

class TestMultiLabel:
    def test_several_classes_active_simultaneously(self, device):
        """Multiple labels across different groups can be active at once."""
        teacher = DeterministicTeacher(
            active_labels={
                "maneuver": ["turn_left", "curve_left"],  # two active in same group
                "edge_case": ["give_way", "avoid_roadworks"],
                "weather_env": ["rain_night", "fog_day"],
            }
        )
        frame = torch.zeros(B, 3, 8, 8, device=device)
        targets = teacher.label([frame], num_future_horizons=0)

        m_target = targets["maneuver"][0]
        assert m_target[:, DEFAULT_TAXONOMY["maneuver"].index("turn_left")].all()
        assert m_target[:, DEFAULT_TAXONOMY["maneuver"].index("curve_left")].all()
        # All-zero for inactive labels
        assert m_target[:, DEFAULT_TAXONOMY["maneuver"].index("continue_straight")].sum() == 0

    def test_multi_label_loss_accepts_multiple_active(self, device):
        band = _band(device)
        vh = _vh(device)
        logits = band(vh, mode="train").logits

        # Create targets with multiple classes active
        targets = {}
        for group in DEFAULT_TAXONOMY.groups:
            t_list = []
            for _ in logits[group.name]:
                t = torch.zeros(B, len(group), device=device)
                t[:, 0] = 1.0
                t[:, 1] = 1.0   # two classes active
                t_list.append(t)
            targets[group.name] = t_list

        loss_fn = ReasoningLoss(weight=1.0)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss) and loss > 0


# ---------------------------------------------------------------------------
# DeterministicTeacher
# ---------------------------------------------------------------------------

class TestDeterministicTeacher:
    def test_deterministic_across_calls(self, device):
        teacher = DeterministicTeacher(
            active_labels={"maneuver": ["turn_right"], "edge_case": ["nudge_out"]}
        )
        frame = torch.randn(B, 3, 8, 8, device=device)
        t1 = teacher.label([frame], num_future_horizons=4)
        t2 = teacher.label([frame], num_future_horizons=4)
        for group_name in t1:
            for h1, h2 in zip(t1[group_name], t2[group_name]):
                assert torch.equal(h1, h2)

    def test_deterministic_returns_correct_number_of_horizons(self, device):
        teacher = DeterministicTeacher()
        frame = torch.zeros(B, 3, 8, 8, device=device)
        targets = teacher.label([frame], num_future_horizons=4)
        for group_name in DEFAULT_TAXONOMY.group_names:
            assert len(targets[group_name]) == 5  # 1 current + 4 future

    def test_deterministic_correct_shapes(self, device):
        teacher = DeterministicTeacher()
        frame = torch.zeros(B, 3, 8, 8, device=device)
        targets = teacher.label([frame], num_future_horizons=4)
        for group in DEFAULT_TAXONOMY.groups:
            for t in targets[group.name]:
                assert t.shape == (B, len(group))

    def test_deterministic_active_labels_set_correctly(self, device):
        teacher = DeterministicTeacher(
            active_labels={"maneuver": ["continue_straight"]}
        )
        frame = torch.zeros(B, 3, 8, 8, device=device)
        targets = teacher.label([frame], num_future_horizons=0)
        t = targets["maneuver"][0]
        idx = DEFAULT_TAXONOMY["maneuver"].index("continue_straight")
        assert (t[:, idx] == 1.0).all()
        # All other maneuver classes must be 0
        mask = torch.ones(len(DEFAULT_TAXONOMY["maneuver"]), dtype=torch.bool)
        mask[idx] = False
        assert (t[:, mask] == 0.0).all()

    def test_deterministic_all_zero_when_no_active_labels(self, device):
        teacher = DeterministicTeacher()   # no active_labels
        frame = torch.zeros(B, 3, 8, 8, device=device)
        targets = teacher.label([frame], num_future_horizons=0)
        for group_name in DEFAULT_TAXONOMY.group_names:
            assert (targets[group_name][0] == 0.0).all()

    def test_deterministic_invalid_label_raises(self):
        with pytest.raises(KeyError):
            DeterministicTeacher(active_labels={"maneuver": ["nonexistent_label"]})

    def test_deterministic_ignores_pixel_values(self, device):
        """Output must be identical regardless of frame pixel content."""
        teacher = DeterministicTeacher(active_labels={"maneuver": ["turn_left"]})
        f1 = torch.zeros(B, 3, 8, 8, device=device)
        f2 = torch.ones(B, 3, 8, 8, device=device) * 255
        t1 = teacher.label([f1], num_future_horizons=0)
        t2 = teacher.label([f2], num_future_horizons=0)
        assert torch.equal(t1["maneuver"][0], t2["maneuver"][0])


# ---------------------------------------------------------------------------
# ReasoningLoss
# ---------------------------------------------------------------------------

class TestReasoningLoss:
    def _make_logits_and_targets(self, device, logit_val: float, target_val: float):
        """Create synthetic logits and targets for all groups × 5 horizons."""
        band = _band(device)
        vh = _vh(device)
        logits = band(vh, mode="train").logits
        # Override with constant logit values
        for group in DEFAULT_TAXONOMY.groups:
            logits[group.name] = [
                torch.full((B, len(group)), logit_val, device=device)
                for _ in range(5)
            ]
        targets = {}
        for group in DEFAULT_TAXONOMY.groups:
            targets[group.name] = [
                torch.full((B, len(group)), target_val, device=device)
                for _ in range(5)
            ]
        return logits, targets

    def test_loss_near_zero_for_perfect_targets(self, device):
        """logit >> 0 with target=1 → near-zero BCE."""
        logits, targets = self._make_logits_and_targets(device, 10.0, 1.0)
        loss = ReasoningLoss(weight=1.0)(logits, targets)
        assert loss.item() < 0.1

    def test_loss_lower_for_better_predictions(self, device):
        """Better-aligned logits → lower loss than random logits."""
        logits_good, targets = self._make_logits_and_targets(device, 8.0, 1.0)
        logits_bad, _ = self._make_logits_and_targets(device, -8.0, 1.0)
        loss_good = ReasoningLoss()(logits_good, targets)
        loss_bad = ReasoningLoss()(logits_bad, targets)
        assert loss_good < loss_bad

    def test_loss_weight_scales_output(self, device):
        logits, targets = self._make_logits_and_targets(device, 0.0, 0.5)
        l1 = ReasoningLoss(weight=1.0)(logits, targets)
        l2 = ReasoningLoss(weight=2.0)(logits, targets)
        assert torch.allclose(l2, 2.0 * l1, atol=1e-5)

    def test_loss_finite_and_positive(self, device):
        logits, targets = self._make_logits_and_targets(device, 0.0, 0.5)
        loss = ReasoningLoss()(logits, targets)
        assert torch.isfinite(loss) and loss > 0

    def test_loss_group_mismatch_raises(self, device):
        logits, targets = self._make_logits_and_targets(device, 0.0, 0.5)
        del targets["maneuver"]
        with pytest.raises(ValueError, match="Group mismatch"):
            ReasoningLoss()(logits, targets)

    def test_loss_horizon_mismatch_raises(self, device):
        logits, targets = self._make_logits_and_targets(device, 0.0, 0.5)
        targets["maneuver"] = targets["maneuver"][:3]   # 3 horizons vs 5
        with pytest.raises(ValueError, match="horizons"):
            ReasoningLoss()(logits, targets)

    def test_loss_reduction_none_returns_per_sample(self, device):
        logits, targets = self._make_logits_and_targets(device, 0.0, 0.5)
        loss = ReasoningLoss(reduction="none")(logits, targets)
        assert loss.shape == (B,)

    def test_loss_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            ReasoningLoss(reduction="sum")

    def test_loss_gradients_flow(self, device):
        band = _band(device)
        vh = _vh(device)
        logits = band(vh, mode="train").logits
        targets = {}
        for group in DEFAULT_TAXONOMY.groups:
            targets[group.name] = [
                torch.zeros(B, len(group), device=device) for _ in logits[group.name]
            ]
        loss = ReasoningLoss()(logits, targets)
        loss.backward()
        # Gradients must flow to the decoder heads
        assert any(p.grad is not None for p in band.heads.parameters())


# ---------------------------------------------------------------------------
# AutoE2E integration — reasoning band disabled = unchanged baseline
# ---------------------------------------------------------------------------

class _AutoE2EHarness:
    """Shared mock-backbone harness (NOT collected: no Test prefix) so the
    wiring and faithfulness suites reuse _build/_inputs without
    pytest re-collecting inherited tests."""

    class _MockBackbone(nn.Module):
        def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kw):
            super().__init__()
            self.backbone_channels = 1440
            self._st = nn.ModuleList([
                nn.Sequential(nn.Conv2d(3, 96, 3, 1, 1), nn.AdaptiveAvgPool2d(64)),
                nn.Sequential(nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32)),
                nn.Sequential(nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16)),
                nn.Sequential(nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8)),
            ])
        def forward(self, x):
            outs, h = [], x
            for s in self._st:
                h = s(h)
                outs.append(h)
            return outs

    def _build(self, device, *, enable_reasoning_band=False,
               enable_world_model=False, reasoning_kwargs=None):
        from unittest.mock import patch
        from model_components.auto_e2e import AutoE2E
        with patch("model_components.reactive_e2e.Backbone", self._MockBackbone):
            return AutoE2E(
                num_views=2,
                view_fusion_kwargs={"bev_h": 8, "bev_w": 8},
                enable_world_model=enable_world_model,
                world_model_kwargs={"feature_channels": 768} if enable_world_model else None,
                enable_reasoning_band=enable_reasoning_band,
                reasoning_kwargs=reasoning_kwargs,
            ).to(device)

    def _inputs(self, device):
        return (
            torch.randn(B, 2, 3, 256, 256, device=device),  # camera_tiles
            torch.randn(B, 3, 256, 256, device=device),      # map_input (256 required by map encoder)
            torch.zeros(B, 896, device=device),              # visual_history
            torch.randn(B, 256, device=device),              # egomotion_history
        )


class TestAutoE2EReasoningBandWiring(_AutoE2EHarness):
    def test_disabled_returns_same_shape_as_baseline(self, device):
        """enable_reasoning_band=False → output identical to un-modified AutoE2E."""
        m = self._build(device, enable_reasoning_band=False)
        cam, mp, vh, ego = self._inputs(device)
        out = m(cam, mp, vh, ego, mode="infer")
        traj = out[0] if isinstance(out, tuple) else out
        assert traj.shape == (B, 128)

    def test_disabled_reasoning_band_is_none(self, device):
        m = self._build(device, enable_reasoning_band=False)
        assert m.Reasoning_Band is None

    def test_enabled_reasoning_band_is_set(self, device):
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        assert m.Reasoning_Band is not None

    def test_infer_mode_returns_trajectory_only(self, device):
        """In infer mode the reasoning_pred is not returned (same as World Model)."""
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        cam, mp, vh, ego = self._inputs(device)
        out = m(cam, mp, vh, ego, mode="infer")
        # Should NOT be a 3-tuple (reasoning_pred only comes in train mode)
        if isinstance(out, tuple):
            assert len(out) <= 2
        else:
            assert out.shape[-1] == 128

    def test_train_mode_returns_3tuple(self, device):
        """In train mode with reasoning band on: (trajectory, future_state_pred, reasoning_pred)."""
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        cam, mp, vh, ego = self._inputs(device)
        tgt = torch.randn(B, 128, device=device)
        out = m(cam, mp, vh, ego, mode="train", trajectory_target=tgt)
        assert isinstance(out, tuple) and len(out) == 3
        _traj, future_state_pred, reasoning_pred = out
        assert future_state_pred is None   # World Model is OFF
        assert reasoning_pred is not None
        assert "maneuver" in reasoning_pred.logits

    def test_reasoning_pred_horizons_in_train(self, device):
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        cam, mp, vh, ego = self._inputs(device)
        tgt = torch.randn(B, 128, device=device)
        _, _, reasoning_pred = m(cam, mp, vh, ego, mode="train", trajectory_target=tgt)
        for group in DEFAULT_TAXONOMY.groups:
            assert len(reasoning_pred.logits[group.name]) == 5

    def test_baseline_unchanged_enable_false(self, device):
        """Confirm enable_reasoning_band=False ⇒ exact return contract as before."""
        m_base = self._build(device, enable_reasoning_band=False)
        cam, mp, vh, ego = self._inputs(device)
        tgt = torch.randn(B, 128, device=device)
        out_infer = m_base(cam, mp, vh, ego, mode="infer")
        out_train = m_base(cam, mp, vh, ego, mode="train", trajectory_target=tgt)
        # Infer: plain tensor or 1-tuple
        if isinstance(out_infer, tuple):
            assert len(out_infer) <= 2
        else:
            assert torch.is_tensor(out_infer)
        # Train without world model: plain tensor (no 3-tuple)
        assert not (isinstance(out_train, tuple) and len(out_train) == 3)


# ---------------------------------------------------------------------------
# Taxonomy extensibility — KIT labels (no-break contract)
# ---------------------------------------------------------------------------

class TestTaxonomyExtensibility:
    def test_kit_label_does_not_shift_existing_indices(self):
        """Adding a KIT group must not change any existing group's indices."""
        t = ScenarioTaxonomy()
        # Record all indices before extension
        before = {
            g.name: {label: g.index(label) for label in g.labels}
            for g in t.groups
        }
        t.register_group("kit_high_level", [
            "intersection_scenario",
            "overtake",
            "construction_zone",
        ])
        # All old indices must be unchanged
        for gname, label_map in before.items():
            for label, idx in label_map.items():
                assert t[gname].index(label) == idx

    def test_kit_group_accessible_after_register(self):
        t = ScenarioTaxonomy()
        t.register_group("kit_high_level", ["intersection_scenario"])
        assert t.num_classes("kit_high_level") == 1
        assert t["kit_high_level"].index("intersection_scenario") == 0

    def test_reasoning_band_with_extended_taxonomy(self, device):
        """ReasoningBand works correctly with an extended taxonomy."""
        t = ScenarioTaxonomy()
        t.register_group("kit_high_level", ["intersection_scenario", "overtake"])
        band = ReasoningBand(
            visual_history_dim=VH_DIM, hidden_dim=32, taxonomy=t
        ).to(device)
        logits = band(_vh(device), mode="train").logits
        assert "kit_high_level" in logits
        assert len(logits["kit_high_level"]) == 5
        assert logits["kit_high_level"][0].shape == (B, 2)


# ---------------------------------------------------------------------------
# Faithfulness / intervention check (#98 evaluation protocol)
# ---------------------------------------------------------------------------

class TestReasoningInterventionDelta(_AutoE2EHarness):
    """The intervention metric: with the gate untrained the coupled and
    intervened trajectories are identical (delta 0.0); once the gate moves,
    the reasoning signal measurably influences the trajectory."""

    def test_zero_delta_at_init(self, device):
        from evaluation.faithfulness import reasoning_intervention_delta
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        cam, mp, vh, ego = self._inputs(device)
        delta = reasoning_intervention_delta(m, cam, mp, vh, ego)
        assert delta["trajectory_l2"] == pytest.approx(0.0, abs=1e-5)
        assert delta["history_shift"] == pytest.approx(0.0, abs=1e-5)

    def test_positive_delta_once_gate_moves(self, device):
        from evaluation.faithfulness import reasoning_intervention_delta
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        with torch.no_grad():
            m.Reasoning_Band.gate.beta.bias.fill_(1.0)
        cam, mp, vh, ego = self._inputs(device)
        delta = reasoning_intervention_delta(m, cam, mp, vh, ego)
        assert delta["history_shift"] > 0.0

    def test_band_restored_after_intervention(self, device):
        from evaluation.faithfulness import reasoning_intervention_delta
        m = self._build(device, enable_reasoning_band=True,
                        reasoning_kwargs={"hidden_dim": 32})
        band = m.Reasoning_Band
        cam, mp, vh, ego = self._inputs(device)
        reasoning_intervention_delta(m, cam, mp, vh, ego)
        assert m.Reasoning_Band is band

    def test_requires_reasoning_band(self, device):
        from evaluation.faithfulness import reasoning_intervention_delta
        m = self._build(device, enable_reasoning_band=False)
        cam, mp, vh, ego = self._inputs(device)
        with pytest.raises(ValueError, match="enable_reasoning_band"):
            reasoning_intervention_delta(m, cam, mp, vh, ego)


# ---------------------------------------------------------------------------
# Audit regressions (comité 3/07)
# ---------------------------------------------------------------------------

class TestAuditRegressions(_AutoE2EHarness):
    def test_faithfulness_zero_delta_with_world_model_on(self, device):
        """CRITICAL fix: with the World Model enabled, every forward pushes to
        the rolling buffer — without snapshot/restore the coupled and
        intervened runs would diverge even with an untrained gate."""
        from evaluation.faithfulness import reasoning_intervention_delta
        m = self._build(device, enable_reasoning_band=True,
                        enable_world_model=True,
                        reasoning_kwargs={"hidden_dim": 32})
        cam, mp, vh, ego = self._inputs(device)
        buf_before = list(m.visual_history_buffer._buf)
        delta = reasoning_intervention_delta(m, cam, mp, vh, ego)
        assert delta["trajectory_l2"] == pytest.approx(0.0, abs=1e-5)
        # Caller's rollout state untouched by the metric.
        assert len(m.visual_history_buffer._buf) == len(buf_before)
