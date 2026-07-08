"""Horizon-aware, action-relevant reasoning head (issue #98, v2).

Consumes the already-computed 1 Hz Encoded Visual History ``[B, 896]`` and the
ego-motion context ``[B, 256]`` (from TemporalMemory), projects each source
into its own 256-d token, and lets five learned **horizon queries** (now, +1s,
+2s, +3s, +4s) cross-attend those context tokens through a small Transformer
decoder. Each horizon token drives per-group structured heads (relation /
hazard / cause / the four response axes) plus a per-horizon confidence, and is
pooled into a compact ``reasoning_latent [B, 256]`` for the planner.

Runtime-safe: NO teacher import, no second vision backbone, no language decoder.
Teacher supervision is generated offline (see
``data_processing/reasoning_label_generation``) and consumed as frozen labels.

Why horizon queries + cross-attention rather than one MLP: a pedestrian that is
irrelevant *now* may be action-relevant in 2 s. Each horizon needs its own
representation, and a 2-layer / 4-head / 256-d decoder is far cheaper than a
language decoder while being more expressive than a shared MLP trunk (the v1
skeleton this replaces). See ``Design/horizon_reasoning_architecture.md``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .reasoning_taxonomy import DEFAULT_TAXONOMY, ReasoningTaxonomy
from .types import HorizonReasoningPrediction

# The action-relevant core heads (group name -> attribute on the prediction).
# Optional v2 context/timing heads can be added later without touching these.
_CORE_HEADS = (
    "relation_to_ego",
    "hazard_event",
    "cause",
    "longitudinal_response",
    "lateral_response",
    "tactical_response",
    "rule_response",
)


def _context_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """LayerNorm → Linear → GELU → Linear, projecting a source to ``hidden_dim``."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class AttentionPool(nn.Module):
    """Pool ``[B, N, D]`` tokens to ``[B, D]`` via a learned query's attention."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)          # [B, 1, D]
        pooled, _ = self.attn(q, tokens, tokens)  # [B, 1, D]
        return pooled.squeeze(1)                   # [B, D]


class HorizonReasoningHead(nn.Module):
    """Predict action-relevant reasoning over five horizons from the 896 history.

    Args:
        visual_history_dim: dimensionality of the Encoded Visual History (896).
        ego_context_dim: dimensionality of the ego context from TemporalMemory (256).
        hidden_dim: shared token / decoder width (256).
        num_horizons: number of horizons (5: now, +1s..+4s).
        num_layers / num_heads / dropout: horizon-decoder config.
        route_context_dim / map_context_dim: optional extra context sources;
            omit (None) to not build the corresponding token/projection.
        taxonomy: label registry (defaults to :data:`DEFAULT_TAXONOMY`).
        teacher_embedding_dim: if set, build a training-only alignment head
            producing ``student_reasoning_embedding [B, 5, D]`` (default None).

    Forward:
        head(visual_history[B,896], ego_context[B,256],
             route_context=None, map_context=None) -> HorizonReasoningPrediction
    """

    def __init__(
        self,
        visual_history_dim: int = 896,
        ego_context_dim: int = 256,
        hidden_dim: int = 256,
        num_horizons: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        route_context_dim: Optional[int] = None,
        map_context_dim: Optional[int] = None,
        taxonomy: Optional[ReasoningTaxonomy] = None,
        teacher_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        # The horizon count is a CROSS-CUTTING constant: the label schema
        # (HORIZON_SECONDS), the tensorizer, and HorizonReasoningLoss all assume
        # exactly 5 horizons (now,+1s,+2s,+3s,+4s). Sizing the head differently
        # would crash the loss with a shape mismatch, so reject it here rather
        # than advertise a knob the rest of the stack cannot honor.
        if num_horizons != 5:
            raise ValueError(
                f"num_horizons must be 5 (fixed across schema/loss); got {num_horizons}. "
                "Changing the horizon count requires updating HORIZON_SECONDS, the "
                "target tensorizer, and HorizonReasoningLoss together."
            )
        self.taxonomy = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        # Per-source context projections (each source keeps its own semantics).
        self.visual_proj = _context_mlp(visual_history_dim, hidden_dim)
        self.ego_proj = _context_mlp(ego_context_dim, hidden_dim)
        self.route_proj = (
            _context_mlp(route_context_dim, hidden_dim)
            if route_context_dim is not None else None
        )
        self.map_proj = (
            _context_mlp(map_context_dim, hidden_dim)
            if map_context_dim is not None else None
        )

        # Five learned horizon queries: now, +1s, +2s, +3s, +4s.
        self.horizon_queries = nn.Parameter(
            torch.randn(num_horizons, hidden_dim) * 0.02
        )

        # Small cross-attention decoder: queries attend to the context tokens.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Structured heads: one Linear(hidden_dim, C) per action-relevant group.
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, self.taxonomy.num_classes(name))
             for name in _CORE_HEADS}
        )

        # Per-horizon confidence (raw logits).
        self.confidence_head = nn.Linear(hidden_dim, 1)

        # Pooled planner-facing latent.
        self.attn_pool = AttentionPool(hidden_dim, num_heads=num_heads)
        self.latent_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional training-only teacher-embedding alignment head.
        self.teacher_embedding_dim = teacher_embedding_dim
        self.align_head: Optional[nn.Module] = None
        if teacher_embedding_dim is not None:
            self.align_head = nn.Sequential(
                nn.Linear(hidden_dim, teacher_embedding_dim),
                nn.LayerNorm(teacher_embedding_dim),
            )

    def _context_tokens(
        self,
        visual_history: torch.Tensor,
        ego_context: torch.Tensor,
        route_context: Optional[torch.Tensor],
        map_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Stack the available per-source tokens into ``[B, N_context, hidden]``.

        Missing optional sources are OMITTED (no learned null token), per the
        design: the branch must work with just visual+ego in v1.
        """
        tokens = [self.visual_proj(visual_history), self.ego_proj(ego_context)]
        if self.route_proj is not None and route_context is not None:
            tokens.append(self.route_proj(route_context))
        if self.map_proj is not None and map_context is not None:
            tokens.append(self.map_proj(map_context))
        return torch.stack(tokens, dim=1)  # [B, N_context, hidden]

    def forward(
        self,
        visual_history: torch.Tensor,
        ego_context: torch.Tensor,
        route_context: Optional[torch.Tensor] = None,
        map_context: Optional[torch.Tensor] = None,
    ) -> HorizonReasoningPrediction:
        """Run the reasoning head.

        Args:
            visual_history: ``[B, visual_history_dim]`` Encoded Visual History.
            ego_context: ``[B, ego_context_dim]`` ego context from TemporalMemory.
            route_context / map_context: optional extra context (omitted if None
            or if the head was built without the corresponding projection).

        Returns:
            :class:`HorizonReasoningPrediction`.
        """
        B = visual_history.shape[0]
        context_tokens = self._context_tokens(
            visual_history, ego_context, route_context, map_context
        )  # [B, N_context, hidden]

        queries = self.horizon_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 5, hidden]
        horizon_tokens = self.decoder(queries, context_tokens)         # [B, 5, hidden]

        logits = {name: head(horizon_tokens) for name, head in self.heads.items()}
        confidence_logits = self.confidence_head(horizon_tokens)       # [B, 5, 1]

        reasoning_latent = self.latent_mlp(self.attn_pool(horizon_tokens))  # [B, hidden]

        student_embedding = (
            self.align_head(horizon_tokens) if self.align_head is not None else None
        )

        return HorizonReasoningPrediction(
            horizon_tokens=horizon_tokens,
            reasoning_latent=reasoning_latent,
            relation_to_ego_logits=logits["relation_to_ego"],
            hazard_event_logits=logits["hazard_event"],
            cause_logits=logits["cause"],
            longitudinal_response_logits=logits["longitudinal_response"],
            lateral_response_logits=logits["lateral_response"],
            tactical_response_logits=logits["tactical_response"],
            rule_response_logits=logits["rule_response"],
            confidence_logits=confidence_logits,
            student_reasoning_embedding=student_embedding,
        )
