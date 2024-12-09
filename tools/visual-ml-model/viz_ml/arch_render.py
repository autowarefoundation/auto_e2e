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
