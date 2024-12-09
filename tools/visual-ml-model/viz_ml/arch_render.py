"""Architecture-diagram renderer (arch mode) — dependency-free, left-to-right.

Takes an arch_v1 IR (schema/arch_v1.schema.json) and lays it out automatically into a
left-to-right diagram in the style of a hand-drawn paper/README architecture figure:
inputs on the LEFT, the data spine through the middle, outputs + a pink loss column on the
RIGHT, branching/merging, dashed loss/feedback edges, and an "ONLY DURING TRAINING" banner.

Pure stdlib + Python-generated inline SVG (no JS libs, no graphviz, no CDN). The output is a
self-contained HTML shell (dark theme, Save-PNG button, click-to-detail tip) defined here.

Layout = a small deterministic Sugiyama-lite pipeline:
  1. layering (x): longest-path over dataflow edges; pin inputs left, outputs right, losses
     in a dedicated far-right column; pull-right tightening; honor optional lane hints.
  2. row ordering (y): barycenter sweeps, keep lowest-crossing; honor optional row hints.
  3. coordinates: variable box heights from estimated text wrapping; per-column centering.
  4. edges: bezier forward edges with spread ports; feedback edges bow through a reserved
     top channel; loss=pink dashed, skip=thin gold, feedback=amber dashed.
  5. group banners: bounding band + pink pill over train_only members.
All ordering keys are total (tie-break on IR index) so the output is byte-stable.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .validate import _has_cycle


def _esc(s: Any) -> str:
    return html.escape(str(s)) if s is not None else ""


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# base role -> (fill, stroke) palette; the 5 arch-specific roles are overlaid in ARCH_COLORS
ROLE_COLORS = {
    "input":              ("#10233a", "#2f6fb0"),
    "output":             ("#3a1622", "#b83a3e"),
    "backbone":           ("#0e2a26", "#12a594"),
    "embedding":          ("#0e2a26", "#12a594"),
    "convolution":        ("#2a2016", "#ad7f58"),
    "self_attention":     ("#3a1718", "#e5484d"),
    "cross_attention":    ("#3a2410", "#f5a623"),
    "linear_proj":        ("#23252c", "#8b8d98"),
    "ffn_mlp_block":      ("#261a3a", "#8e4ec6"),
    "moe_block":          ("#261a3a", "#8e4ec6"),
    "normalization":      ("#0d2236", "#0091ff"),
    "activation":         ("#10241a", "#30a46c"),
    "positional_encoding": ("#2e1228", "#d6409f"),
    "recurrent":          ("#2a2410", "#c9a227"),
    "conditioning":       ("#3a2410", "#f5a623"),
    "pooler":             ("#23252c", "#8b8d98"),
    "fusion":             ("#2a2410", "#ffb224"),
    "head":               ("#3a1622", "#b83a3e"),
    "merge_add":          ("#23252c", "#c8cad0"),
    "buffer":             ("#1c2128", "#8896a6"),
    "other":              ("#1c2128", "#6e7681"),
}
