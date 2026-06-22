"""Optional System 2 causal reasoning head (Issue #17).

A lightweight auxiliary head that takes a context vector (e.g. the planner's
``ego_hidden``) and produces:

  * ``reasoning_latent`` — a continuous latent that can condition the
    planner downstream (cascade design),
  * ``decision_logits`` — a 5-way classification over the dominant causal
    factor of the current scene.

Training signal: ``causal_consistency_loss`` (cross-entropy) against
pseudo-labels produced by a VLM (vision-language model) prompted over the
KITScenes LongTail dataset. The labels are therefore noisy pseudo-labels,
not human ground truth; the head is auxiliary and OPTIONAL — it does not
modify AutoE2E's default forward pass or its 3-tuple return contract.

Cascade requirement: gradients must flow from the reasoning outputs back to
the trunk that produced the context vector (no ``torch.no_grad`` anywhere),
so that the auxiliary objective shapes the shared representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene_context import SceneContext, CausalReasoningContext

# Dominant causal factor classes (pseudo-labelled by a VLM on KITScenes
# LongTail). Order is part of the contract — do not reorder.
CAUSAL_CLASSES = (
    "intersection",
    "pedestrian",
    "traffic_light",
    "obstacle",
    "clear",
)
NUM_CAUSAL_CLASSES = len(CAUSAL_CLASSES)


class CausalReasoningModule(nn.Module):
    """System 2 head: context vector -> (reasoning_latent, decision_logits).

    Args:
        embed_dim: dimensionality of the input context vector
            (e.g. ``ego_hidden`` is 256 in AutoE2E).
        latent_dim: dimensionality of ``reasoning_latent``. Defaults to
            ``embed_dim`` so the latent can be summed/concatenated into the
            planner without extra projections.
        num_classes: number of causal decision classes (default 5, see
            ``CAUSAL_CLASSES``).
    """

    def __init__(self, embed_dim: int = 256, latent_dim: int | None = None,
                 num_classes: int = NUM_CAUSAL_CLASSES):
        super().__init__()
        latent_dim = embed_dim if latent_dim is None else latent_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Shared reasoning trunk. Plain differentiable ops only — gradients
        # must reach the upstream context producer (cascade design).
        self.reasoning_trunk = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Decision head on top of the reasoning latent.
        self.decision_head = nn.Linear(latent_dim, num_classes)

    def reason(self, context: torch.Tensor) -> torch.Tensor:
        """Return only the reasoning latent ``[B, latent_dim]``.

        This is the cascade hook: the latent can be fed to the planner
        (e.g. added to ``ego_hidden`` or concatenated to its input) so the
        System 2 head conditions trajectory generation. Fully
        differentiable — no ``torch.no_grad``.
        """
        return self.reasoning_trunk(context)

    def forward(self, context: torch.Tensor):
        """Full pass.

        Args:
            context: ``[B, embed_dim]`` context vector (e.g. ``ego_hidden``).

        Returns:
            reasoning_latent: ``[B, latent_dim]``
            decision_logits: ``[B, num_classes]``
        """
        reasoning_latent = self.reason(context)
        decision_logits = self.decision_head(reasoning_latent)
        return reasoning_latent, decision_logits

    def produce_context(self, context: torch.Tensor) -> SceneContext:
        """Produce structured SceneContext for the planner downstream.
        
        This satisfies the 'optional producer' architecture requested by
        the working group: instead of passing raw logits, it packages the
        reasoning with explicit confidence and provenance.
        """
        reasoning_latent, decision_logits = self.forward(context)
        confidence = F.softmax(decision_logits, dim=-1).max(dim=-1).values
        causal_context = CausalReasoningContext(
            reasoning_latent=reasoning_latent,
            causal_class_logits=decision_logits,
            confidence=confidence,
            provenance="vlm_causal_head"
        )
        return SceneContext(causal_reasoning=causal_context)


def causal_consistency_loss(decision_logits: torch.Tensor,
                            labels: torch.Tensor,
                            label_smoothing: float = 0.0,
                            class_weights: torch.Tensor | None = None,
                            ) -> torch.Tensor:
    """Cross-entropy between predicted causal decisions and pseudo-labels.

    ``labels`` are integer class indices ``[B]`` obtained by pseudo-labelling
    KITScenes LongTail scenes with a VLM (the VLM is prompted to name the
    dominant causal factor; its answer is mapped to ``CAUSAL_CLASSES``).
    Because the labels are model-generated they are noisy — treat this as an
    auxiliary consistency objective, weighted low relative to the imitation
    loss.

    Args:
        decision_logits: ``[B, num_classes]`` raw logits.
        labels: ``[B]`` integer class indices (VLM pseudo-labels).
        label_smoothing: optional smoothing factor in ``[0, 1)`` passed to
            ``F.cross_entropy``. Recommended > 0 here: the VLM pseudo-labels
            over KITScenes LongTail are noisy, and smoothing softens the
            penalty for confidently disagreeing with a wrong pseudo-label.
        class_weights: optional ``[num_classes]`` per-class weights (e.g. to
            upweight rare long-tail classes); passed as ``weight`` to
            ``F.cross_entropy``.
    """
    return F.cross_entropy(
        decision_logits, labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
