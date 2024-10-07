"""Stage 3 — LLM architecture-diagram extraction.

Assembles the context (concrete config + registry-variant guidance + code bundle + AST facts
+ the arch_v1 schema) and invokes the `claude` CLI non-interactively to emit the arch_v1 IR.
Claude is the backbone: it reads __init__ AND forward() and reconstructs the blocks, the
left-to-right data flow, the training-only branches and the losses that rule-based parsing
cannot.

If the `claude` CLI is unavailable, the caller can supply a pre-computed arch IR (e.g. a
checked-in example) via `--arch` so the renderer still runs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .resolve import Bundle, bundle_to_facts_dict

_HERE = Path(__file__).resolve().parent
_ARCH_PROMPT = _HERE.parent / "prompts" / "arch_system_prompt.md"
_ARCH_SCHEMA = _HERE.parent / "schema" / "arch_v1.schema.json"


def claude_available() -> bool:
    return shutil.which("claude") is not None


def _registry_block(bundle: Bundle) -> list[str]:
    """Variant guidance: which concrete class the config selects (Registry pattern)."""
    if not bundle.registry_options:
        return []
    active = bundle.active_variant_classes()
    inactive = bundle.inactive_variant_classes()
    lines = ["## Registry / factory variants (IMPORTANT)",
             "This model selects submodules via a registry/factory. Based on the config, "
             "model the SELECTED variant as the real architecture and treat the others as "
             "inactive alternatives (omit them or mark them clearly and de-emphasize)."]
