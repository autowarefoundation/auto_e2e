"""Tests for the reasoning-band supervision pipeline (issue #98/#103).

Covers the pieces that run WITHOUT any VLM (CPU-only, no downloads):
    * ReasoningLoss with loss_type="asl" (Asymmetric Loss for the class
      imbalance) alongside the default BCE.
    * MultiTeacher: agreement-fraction fusion of two teachers (the soft
      confidence signal).
    * Qwen2-VL prompt/parse helpers (pure functions; the model call itself is
      GPU/offline work).
    * long_tail_split: stratified evaluation subset for #98's protocol.
"""

from __future__ import annotations

import pytest
import torch

from evaluation.splits import long_tail_split
from model_components.reasoning.scenario_taxonomy import (
    DEFAULT_TAXONOMY,
    ScenarioTaxonomy,
)
from model_components.reasoning.teachers import (
    DeterministicTeacher,
    MultiTeacher,
)
from model_components.reasoning.teachers.qwen2vl import (
    build_scenario_prompt,
    labels_to_targets,
    parse_scenario_response,
)
from training.losses.reasoning_loss import ReasoningLoss

B = 2


def _frames(n: int = 1) -> list[torch.Tensor]:
    return [torch.zeros(B, 3, 8, 8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Asymmetric Loss (ASL) option
# ---------------------------------------------------------------------------

class TestAsymmetricLoss:
    def _logits_targets(self, logit_val: float, target_val: float):
        logits = {
            g.name: [torch.full((B, len(g)), logit_val)]
            for g in DEFAULT_TAXONOMY.groups
        }
        targets = {
            g.name: [torch.full((B, len(g)), target_val)]
            for g in DEFAULT_TAXONOMY.groups
        }
        return logits, targets

    def test_asl_near_zero_for_perfect_predictions(self):
        logits, targets = self._logits_targets(10.0, 1.0)
        loss = ReasoningLoss(loss_type="asl")(logits, targets)
        assert loss.item() < 0.01

    def test_asl_penalises_confident_mistakes(self):
        good, targets = self._logits_targets(10.0, 1.0)
        bad, _ = self._logits_targets(-10.0, 1.0)
        loss_fn = ReasoningLoss(loss_type="asl")
        assert loss_fn(bad, targets).item() > loss_fn(good, targets).item()

    def test_asl_downweights_easy_negatives_vs_bce(self):
        """The point of ASL: an easy negative (low prob, target 0) contributes
        ~0 loss, while BCE still charges it."""
        logits, targets = self._logits_targets(-2.0, 0.0)
        asl = ReasoningLoss(loss_type="asl")(logits, targets).item()
        bce = ReasoningLoss(loss_type="bce")(logits, targets).item()
        assert asl < bce

    def test_reduction_none_shape(self):
        logits, targets = self._logits_targets(0.0, 0.5)
        loss = ReasoningLoss(loss_type="asl", reduction="none")(logits, targets)
        assert loss.shape == (B,)

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            ReasoningLoss(loss_type="focal")

    def test_asl_backward(self):
        logits = {
            g.name: [torch.zeros(B, len(g), requires_grad=True)]
            for g in DEFAULT_TAXONOMY.groups
        }
        targets = {
            g.name: [torch.ones(B, len(g))] for g in DEFAULT_TAXONOMY.groups
        }
        ReasoningLoss(loss_type="asl")(logits, targets).backward()
        assert logits["maneuver"][0].grad is not None


# ---------------------------------------------------------------------------
# MultiTeacher — agreement fusion
# ---------------------------------------------------------------------------

class TestMultiTeacher:
    def test_agreement_fraction(self):
        t1 = DeterministicTeacher(
            active_labels={"maneuver": ["turn_left"], "weather_env": ["rain_day"]}
        )
        t2 = DeterministicTeacher(
            active_labels={"maneuver": ["turn_left", "curve_left"]}
        )
        fused = MultiTeacher([t1, t2]).label(_frames(), num_future_horizons=0)
        man = DEFAULT_TAXONOMY["maneuver"]
        wea = DEFAULT_TAXONOMY["weather_env"]
        assert fused["maneuver"][0][0, man.index("turn_left")] == 1.0   # both
        assert fused["maneuver"][0][0, man.index("curve_left")] == 0.5  # one of two
        assert fused["weather_env"][0][0, wea.index("rain_day")] == 0.5

    def test_needs_two_teachers(self):
        with pytest.raises(ValueError, match="at least 2"):
            MultiTeacher([DeterministicTeacher()])

    def test_taxonomy_mismatch_raises(self):
        other = ScenarioTaxonomy()
        other.register_group("kit_context", ["intersection"])
        with pytest.raises(ValueError, match="same taxonomy"):
            MultiTeacher([
                DeterministicTeacher(),
                DeterministicTeacher(taxonomy=other),
            ])

    def test_horizons_preserved(self):
        fused = MultiTeacher(
            [DeterministicTeacher(), DeterministicTeacher()]
        ).label(_frames(5), num_future_horizons=4)
        for g in DEFAULT_TAXONOMY.groups:
            assert len(fused[g.name]) == 5


# ---------------------------------------------------------------------------
# Qwen2-VL prompt/parse helpers (pure — no model)
# ---------------------------------------------------------------------------

class TestQwenPromptAndParse:
    def test_prompt_lists_every_label(self):
        prompt = build_scenario_prompt(DEFAULT_TAXONOMY)
        for group in DEFAULT_TAXONOMY.groups:
            assert group.name in prompt
            for label in group.labels:
                assert label in prompt

    def test_parse_happy_path_with_chatter(self):
        text = (
            'Sure! Here is the labelling:\n'
            '{"maneuver": ["turn_left"], "edge_case": [], '
            '"weather_env": ["rain_night"]}\nHope that helps.'
        )
        parsed = parse_scenario_response(text, DEFAULT_TAXONOMY)
        assert parsed["maneuver"] == ["turn_left"]
        assert parsed["edge_case"] == []
        assert parsed["weather_env"] == ["rain_night"]

    def test_parse_drops_unknown_labels(self):
        text = '{"maneuver": ["turn_left", "warp_speed"], "edge_case": 3}'
        parsed = parse_scenario_response(text, DEFAULT_TAXONOMY)
        assert parsed["maneuver"] == ["turn_left"]
        assert parsed["edge_case"] == []

    def test_parse_garbage_abstains(self):
        parsed = parse_scenario_response("no json here", DEFAULT_TAXONOMY)
        assert all(v == [] for v in parsed.values())

    def test_labels_to_targets(self):
        per_sample = [
            {"maneuver": ["turn_left"], "edge_case": [], "weather_env": []},
            {"maneuver": [], "edge_case": ["give_way"], "weather_env": []},
        ]
        targets = labels_to_targets(per_sample, DEFAULT_TAXONOMY)
        man = DEFAULT_TAXONOMY["maneuver"]
        edge = DEFAULT_TAXONOMY["edge_case"]
        assert targets["maneuver"][0, man.index("turn_left")] == 1.0
        assert targets["edge_case"][1, edge.index("give_way")] == 1.0
        assert targets["maneuver"][1].sum() == 0.0


# ---------------------------------------------------------------------------
# Long-tail evaluation split (#98 protocol)
# ---------------------------------------------------------------------------

class TestLongTailSplit:
    def test_stratifies_by_membership(self):
        scenarios = [
            ["continue_straight", "fair_day"],
            ["give_way", "rain_day"],
            ["turn_left"],
            ["close_to_vru", "fog_night"],
        ]
        rare = list(DEFAULT_TAXONOMY["edge_case"].labels)
        long_tail, nominal = long_tail_split(scenarios, rare)
        assert long_tail == [1, 3]
        assert nominal == [0, 2]

    def test_empty_long_tail_classes_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            long_tail_split([["a"]], [])


# ---------------------------------------------------------------------------
# Teacher registry — all backends reachable by name
# ---------------------------------------------------------------------------

class TestTeacherRegistry:
    def test_all_backends_registered(self):
        from model_components.reasoning.teachers import _TEACHER_REGISTRY
        for key in ("deterministic", "multi", "qwen2vl", "videollama3"):
            assert key in _TEACHER_REGISTRY

    def test_videollama3_is_explicit_about_pending_wiring(self):
        from model_components.reasoning.teachers.videollama3 import VideoLlama3Teacher
        with pytest.raises(NotImplementedError, match="labelling pipeline"):
            VideoLlama3Teacher().label([torch.zeros(1, 3, 8, 8)])


class TestParseTrailingBrace:
    def test_trailing_brace_after_object_still_parses(self):
        """Audit fix: a greedy regex would swallow the trailing '}' and abstain."""
        text = '{"maneuver": ["turn_left"], "edge_case": [], "weather_env": []} bye }'
        parsed = parse_scenario_response(text, DEFAULT_TAXONOMY)
        assert parsed["maneuver"] == ["turn_left"]
