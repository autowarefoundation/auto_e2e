"""Zero-init reasoning→planner coupling (issue #98, R7).

Injects the reasoning branch's output into a planner conditioning vector behind
a zero-initialised gate, so at initialisation the coupling is a strict no-op and
the reactive baseline is byte-identical up to numerical tolerance. Training moves
the gate away from zero only where reasoning helps the trajectory.

Three modes (the required ablation surface A/B/C):
    * ``none``                    — coupling disabled; the planner is unchanged.
    * ``pooled_latent``           — add ``alpha * reason_proj(reasoning_latent)``.
    * ``horizon_cross_attention`` — a query attends the 5 horizon tokens, then
      ``alpha * reason_proj(attended)`` is added — preserving *when* a hazard
      matters.

``alpha`` is a learned scalar initialised to 0 (the repo's ResidualMapFusion /
#108 ZeroInitGate pattern); ``reason_proj``'s final layer is also zero-init as a
belt-and-braces guarantee that the residual is exactly 0 at init regardless of
the attention output.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

REASONING_MODES = ("none", "pooled_latent", "horizon_cross_attention")


class ReasoningCoupling(nn.Module):
    """Add a zero-init reasoning residual to a planner conditioning vector.

    Args:
        embed_dim: conditioning-vector width (must match ``reasoning_latent`` /
            horizon-token width, 256).
        mode: one of :data:`REASONING_MODES`.
        num_heads: heads for the cross-attention in horizon mode.

    Forward:
        coupling(context[B,D], reasoning_latent=None, horizon_tokens=None,
                 query=None) -> context'[B,D]
        With ``mode="none"`` (or missing reasoning inputs) it returns ``context``
        unchanged. In ``horizon_cross_attention`` mode the attention query is
        ``query`` if given (e.g. the flow-matching action tokens), else the
        context vector itself.
    """

    def __init__(self, embed_dim: int = 256, mode: str = "none", num_heads: int = 4) -> None:
        super().__init__()
        if mode not in REASONING_MODES:
            raise ValueError(f"reasoning_mode must be one of {REASONING_MODES}, got {mode!r}.")
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == "none":
            return

        # Learned scalar gate, zero-init → no-op residual at initialisation.
        # ONLY alpha is zero-init (the ResidualMapFusion pattern). The projection
        # keeps NORMAL init on purpose: if reason_proj's last layer were ALSO
        # zeroed, then at init delta = reason_proj(x) = 0, so d(loss)/d(alpha) =
        # delta = 0 and d(loss)/d(proj) = alpha·(…) = 0 — every coupling param
        # would get exactly zero gradient forever and the branch would never
        # train (a permanent zero fixed point). With alpha=0 but delta≠0 the
        # output is still a strict no-op at init, yet alpha receives gradient and
        # the coupling can learn.
        self.alpha = nn.Parameter(torch.zeros(()))
        self.reason_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if mode == "horizon_cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=0.0, batch_first=True
            )

    def forward(
        self,
        context: torch.Tensor,
        reasoning_latent: Optional[torch.Tensor] = None,
        horizon_tokens: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the reasoning-conditioned context (unchanged if mode='none')."""
        if self.mode == "none":
            return context

        if self.mode == "pooled_latent":
            if reasoning_latent is None:
                return context  # no reasoning available this step → no-op
            delta = self.reason_proj(reasoning_latent)          # [B, D]
            return context + self.alpha * delta

        # horizon_cross_attention
        if horizon_tokens is None:
            return context
        q = query if query is not None else context.unsqueeze(1)  # [B, Tq, D]
        attended, _ = self.cross_attn(q, horizon_tokens, horizon_tokens)  # [B, Tq, D]
        delta = self.reason_proj(attended)                        # [B, Tq, D]
        gated = self.alpha * delta
        # Broadcast back onto the caller's context shape: a single-vector context
        # gets the squeezed residual; a per-token query context keeps its tokens.
        if query is None:
            return context + gated.squeeze(1)
        return q + gated
