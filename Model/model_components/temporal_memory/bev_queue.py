"""BEV-queue temporal-attention memory candidate (Issue #20).

@RyotaYamada proposed in #20 benchmarking several temporal-fusion designs —
BEVFormer / BEVDet4D / StreamPETR / SOLOFusion — against the ``no_memory``
baseline and the recurrent ``one_hz`` encoder, favouring a BEVFormer-style
start. BEVFormer fuses history through **temporal self-attention over a queue
of past BEV features** (Li et al. 2022, "BEVFormer", arXiv:2203.17270);
StreamPETR/SOLOFusion likewise attend over a recent-frame queue.

At the ``[B, T, feat]`` history-vector interface used by this registry (not the
spatial BEV grid), the faithful analog is a **Transformer encoder that attends
over the most recent ``queue_len`` history steps** and pools them into a single
context vector via a learnable query (CLS) token. This is the attention-based,
order-aware alternative to the recurrent 1 Hz compression — a third candidate
for the memory benchmark requested in #20.

It keeps the registry contract intact: ``forward(visual_history,
egomotion_history) -> (visual_context [B, visual_dim], egomotion_context
[B, egomotion_dim])``, and falls back to a pass-through when the history has no
time dimension, exactly like ``OneHzHistoryEncoder``.
"""

import math

import torch
import torch.nn as nn

from .base import BaseTemporalMemory


def _sinusoidal_position_encoding(length: int, dim: int,
                                  device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encoding ``[length, dim]`` (dim even).

    Temporal order matters for fusion, but self-attention is permutation
    invariant, so positions must be encoded explicitly (Vaswani et al. 2017).
    """
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class BevQueueMemory(BaseTemporalMemory):
    """Temporal self-attention over the recent history queue (BEVFormer-style).

    Args:
        visual_dim: visual history feature size (output ``visual_context`` size).
        egomotion_dim: egomotion history feature size.
        d_model: attention width (must be even; default 512).
        num_heads: attention heads (must divide ``d_model``; default 8).
        num_layers: Transformer encoder layers (default 2).
        queue_len: keep only the most recent ``queue_len`` steps (``None`` = all),
            mirroring BEVFormer's fixed-length BEV queue.
        dim_feedforward: FFN width (default ``4 * d_model``).
        dropout: attention/FFN dropout (default 0.0 for deterministic inference).
    """

    def __init__(self, visual_dim: int = 896, egomotion_dim: int = 256,
                 d_model: int = 512, num_heads: int = 8, num_layers: int = 2,
                 queue_len: int | None = None, dim_feedforward: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if queue_len is not None and queue_len < 1:
            raise ValueError(f"queue_len must be >= 1 or None, got {queue_len}")
        self.visual_dim = visual_dim
        self.egomotion_dim = egomotion_dim
        self.d_model = d_model
        self.queue_len = queue_len

        joint_dim = visual_dim + egomotion_dim
        # Project the joint history step into the attention space and back.
        self.in_proj = nn.Linear(joint_dim, d_model)
        self.out_proj = nn.Linear(d_model, joint_dim)
        # Learnable query token that pools the attended queue into one vector.
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, visual_history, egomotion_history, **kwargs):
        # Fallback: no temporal dimension -> pass through (matches OneHz/NoMemory).
        if visual_history.ndim == 2:
            return visual_history, egomotion_history

        # Join the two streams along the feature dim: [B, T, visual+ego].
        joint = torch.cat([visual_history, egomotion_history], dim=-1)

        # Keep only the most recent steps (BEVFormer's fixed-length queue). The
        # history is oldest -> most recent, so slice from the tail.
        if self.queue_len is not None:
            joint = joint[:, -self.queue_len:, :]

        B, T, _ = joint.shape
        x = self.in_proj(joint)                                   # [B, T, d]
        x = x + _sinusoidal_position_encoding(T, self.d_model, x.device)

        # Prepend the learnable query token; its output pools the queue.
        query = self.query_token.expand(B, -1, -1)                # [B, 1, d]
        x = torch.cat([query, x], dim=1)                          # [B, T+1, d]
        x = self.encoder(x)                                       # [B, T+1, d]

        context = self.out_proj(x[:, 0])                          # [B, visual+ego]
        visual_context = context[:, :self.visual_dim]
        egomotion_context = context[:, self.visual_dim:]
        return visual_context, egomotion_context
