"""Stage 1 — AST pre-processor.

Extracts *normalized facts* from PyTorch source using only the stdlib `ast` module
(no torch import, works on non-installable code). These facts are NOT the final output;
they are handed to the LLM (Stage 3) so it reasons over structure instead of raw text,
and they serve as a cross-check oracle.

For each nn.Module class we extract:
  - submodule inventory: self.<name> = <Class>(<args...>)
  - register_buffer(name, ..., persistent=?) flags (ground truth)
  - a syntactic forward() skeleton: per statement, the lhs targets, the attribute calls
    invoked (e.g. self.attn, self.ln_1), and whether a `+` add appears (residual signal)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Submodule:
    var_name: str                 # e.g. "self.c_attn"
    constructor: str | None       # e.g. "nn.Linear"
    args: list[str] = field(default_factory=list)       # source text of positional args
    kwargs: dict[str, str] = field(default_factory=dict) # source text of keyword args
