"""Reasoning Band — multi-label, multi-horizon scenario classification (issue #98).

The band consumes the 896-dim Encoded Visual History (the same vector the World
Action Model produces via :meth:`WorldActionModel.aggregate_history`) and
decodes it into per-group sigmoid classification heads for the current scenario
and — in training — four future horizons at +1 … +4 s (@1 Hz).

Its scenario prediction is supervised by the Video-Language-Model loss
(student/teacher, see ``Model/training/losses/reasoning_loss.py``).  In
addition, the band feeds the trajectory planner through a **zero-init gate**
(agreed in issues #98/#103): the current-scenario probabilities modulate the
visual history via a FiLM-style transform whose weights start at zero, so the
reactive baseline is byte-identical at initialisation and the coupling only
takes effect as training pushes the gate away from zero.  A per-horizon
**confidence head** (issue #103, temporal-first) accompanies the class logits.

Architecture summary:

    896-dim visual_history
        ├── trunk (shared MLP)
        │       ├── per-group × per-horizon sigmoid heads → multi-label logits
        │       └── confidence head                       → [B, horizons]
        └── zero-init gate (conditioned on current-scenario probabilities)
                └── modulated visual_history              → trajectory planner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .scenario_taxonomy import ScenarioTaxonomy, DEFAULT_TAXONOMY


# ---------------------------------------------------------------------------
# Typed output containers
# ---------------------------------------------------------------------------

# ReasoningOutput[group_name][horizon_index] = raw logits [B, num_classes]
ReasoningOutput = Dict[str, List[torch.Tensor]]


@dataclass
class ReasoningPrediction:
    """Typed result of one reasoning-band forward pass (1 Hz tick).

    Fields:
        logits: dict mapping each taxonomy group to a list of per-horizon raw
            logits ``[B, num_classes]`` (apply ``torch.sigmoid`` for
            probabilities).  Train mode: ``1 + num_future_horizons`` entries;
            other modes: 1 (current scenario only).
        confidence: ``[B, num_horizons]`` raw logits for the per-horizon
            confidence (issue #103, temporal-first).  Same horizon count as
            ``logits``.  NOTE: this output is **not supervised in the core PR** —
            it is a trainable placeholder.  Supervise it with
            :func:`~training.losses.reasoning_loss.confidence_brier_loss`
            against a target (e.g. the cross-teacher agreement fraction); the
            full supervise-and-consume-by-the-planner loop is tracked in #110.
        modulated_visual_history: ``[B, visual_history_dim]`` — the visual
            history after the zero-init gate.  Identical to the input at
            initialisation (strict no-op) so the reactive baseline is
            unchanged until training moves the gate.

    Ablation surface: ``logits`` exposes **every** horizon (now, +1 … +4 s), so
    a caller can build an all-horizon-pooled latent or a horizon-token planner
    coupling on top of this output.  The band's own gate is the current-only
    pooled baseline (option B); richer couplings (all-horizon pooled / horizon
    cross-attention) are follow-ups that reuse this same output contract, so the
    interface does not block them.
    """

    logits: ReasoningOutput
    confidence: torch.Tensor
    modulated_visual_history: torch.Tensor


# ---------------------------------------------------------------------------
# Zero-init gate (planner coupling, issues #98/#103)
# ---------------------------------------------------------------------------


class ZeroInitGate(nn.Module):
    """FiLM-style gate that modulates the visual history with scenario probs.

    ``output = visual_history * (1 + gamma) + beta`` where ``gamma`` and
    ``beta`` are linear projections of the current-scenario probability
    vector.  Both projections are zero-initialised (weights AND biases), so at
    initialisation the gate is a strict no-op — the same alpha=0 pattern the
    repo uses in ``ResidualMapFusion``.  Gradients move the gate away from
    zero only where the scenario signal helps the trajectory.

    Args:
        scenario_dim: size of the conditioning vector (total classes across
            all taxonomy groups).
        visual_history_dim: dimensionality of the visual history (default 896).
    """

    def __init__(self, scenario_dim: int, visual_history_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(scenario_dim, visual_history_dim)
        self.beta = nn.Linear(scenario_dim, visual_history_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self, visual_history: torch.Tensor, scenario_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply the gate.

        Args:
            visual_history: ``[B, visual_history_dim]``.
            scenario_probs: ``[B, scenario_dim]`` current-scenario
                probabilities in ``[0, 1]``.

        Returns:
            Modulated ``[B, visual_history_dim]`` (equal to the input at init).
        """
        gamma = self.gamma(scenario_probs)
        beta = self.beta(scenario_probs)
        return visual_history * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ReasoningBand(nn.Module):
    """Multi-label, multi-horizon scenario classification band for AutoE2E.

    Consumes the 896-dim Encoded Visual History produced by the World Action
    Model and outputs, per taxonomy group, sigmoid classification logits for
    the current scene and (in training) four future horizons at +1 … +4 s,
    plus a per-horizon confidence (issue #103).  The current-scenario
    probabilities feed the trajectory planner through a zero-init gate.

    Args:
        visual_history_dim: input dimensionality (must match the Encoded
            Visual History dimension, default 896).
        hidden_dim: width of the shared trunk (default 256).
        num_future_horizons: future horizons predicted in training
            (default 4, for h=+1..+4 s at 1 Hz).
        taxonomy: scenario label registry (defaults to
            :data:`DEFAULT_TAXONOMY`).

    Example::

        band = ReasoningBand()
        pred = band(visual_history, mode="train")   # visual_history: [B, 896]
        pred.logits["maneuver"]          # list of 5 tensors [B, 7] (h=0..4)
        pred.confidence                  # [B, 5]
        pred.modulated_visual_history    # [B, 896] (== input at init)
    """

    def __init__(
        self,
        visual_history_dim: int = 896,
        hidden_dim: int = 256,
        num_future_horizons: int = 4,
        taxonomy: Optional[ScenarioTaxonomy] = None,
    ) -> None:
        super().__init__()
        self.visual_history_dim = visual_history_dim
        self.num_future_horizons = num_future_horizons
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY

        # Shared trunk: project the visual history into a compact representation.
        self.trunk = nn.Sequential(
            nn.Linear(visual_history_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # One linear head per (group, horizon): h=0 (current) plus
        # h=1..num_future_horizons (future, train mode only).
        total_horizons = 1 + num_future_horizons
        self.heads = nn.ModuleDict(
            {
                f"{group.name}_h{h}": nn.Linear(hidden_dim, len(group))
                for group in self.taxonomy.groups
                for h in range(total_horizons)
            }
        )

        # Per-horizon confidence (raw logits; issue #103, temporal-first).
        self.confidence_head = nn.Linear(hidden_dim, total_horizons)

        # Planner coupling: zero-init gate conditioned on the current-scenario
        # probability vector (all groups concatenated).
        self.gate = ZeroInitGate(
            scenario_dim=self.taxonomy.total_classes(),
            visual_history_dim=visual_history_dim,
        )

    def forward(
        self,
        visual_history: torch.Tensor,
        mode: str = "infer",
        images: Optional[torch.Tensor] = None,
    ) -> ReasoningPrediction:
        """Run the reasoning band.

        Args:
            visual_history: ``[B, visual_history_dim]`` — the Encoded Visual
                History from :meth:`WorldActionModel.aggregate_history`.
            mode: ``"train"`` produces all ``1 + num_future_horizons``
                horizons; any other value produces only the current horizon.
            images: unused by this variant (present so the frozen-VLM variant
                can share the same call signature in ``AutoE2E``).

        Returns:
            A :class:`ReasoningPrediction`.
        """
        del images  # only the frozen-VLM variant consumes raw frames
        features = self.trunk(visual_history)
        num_horizons = 1 + self.num_future_horizons if mode == "train" else 1

        logits: ReasoningOutput = {}
        for group in self.taxonomy.groups:
            logits[group.name] = [
                self.heads[f"{group.name}_h{h}"](features)
                for h in range(num_horizons)
            ]

        confidence = self.confidence_head(features)[:, :num_horizons]

        current_probs = torch.cat(
            [torch.sigmoid(logits[g.name][0]) for g in self.taxonomy.groups],
            dim=1,
        )
        modulated = self.gate(visual_history, current_probs)

        return ReasoningPrediction(
            logits=logits,
            confidence=confidence,
            modulated_visual_history=modulated,
        )
