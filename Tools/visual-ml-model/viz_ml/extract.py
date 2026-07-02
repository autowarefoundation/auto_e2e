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
    for o in bundle.registry_options:
        mark = "ACTIVE (selected by config)" if o.active else "inactive"
        lines.append(f"- `{o.registry}[\"{o.key}\"]` -> `{o.class_name}` — {mark}")
    if active:
        lines.append(f"\nSELECTED variant class(es): {', '.join(sorted(active))}. "
                     f"Build through this/these; do NOT wire the inactive ones "
                     f"({', '.join(sorted(inactive)) or 'none'}) into the active dataflow.")
    else:
        lines.append("\nNo variant was selected by the config; note this and pick the "
                     "constructor default if evident, else show all at reduced confidence.")
    return ["", "\n".join(lines), ""]


def _shared_context_parts(bundle: Bundle) -> list[str]:
    """Config + registry guidance + code bundle + AST facts handed to Claude."""
    facts = bundle_to_facts_dict(bundle)
    return [
        f"# Target module: `{bundle.entry_class}`",
        "",
        "## Concrete config (Registry assumption — constructor args are supplied)",
        "```json",
        json.dumps(bundle.config, indent=2),
        "```",
        *_registry_block(bundle),
        "## Code bundle (target class + its same-repo submodule/base classes)",
        "```python",
        bundle.bundle_source(),
        "```",
        "",
        "## AST facts (deterministic; trust these to anchor your reading)",
        "Submodule inventory, register_buffer persistence flags, and the forward() skeleton "
        "(note `has_add: true` statements — those are residual merge points).",
        "```json",
        json.dumps(facts, indent=2),
        "```",
        "",
    ]


def build_arch_prompt(bundle: Bundle) -> str:
    """Context for the left-to-right architecture diagram (paper/README-figure style)."""
    arch_schema = json.loads(_ARCH_SCHEMA.read_text(encoding="utf-8"))
    parts = [
        *_shared_context_parts(bundle),
        "## Architecture IR (arch_v1) JSON Schema (your output must validate against this)",
        "```json",
        json.dumps(arch_schema, indent=2),
        "```",
        "",
        "## Task",
        f"Emit ONE arch_v1 JSON object for `{bundle.entry_class}` configured with the config "
        "above. ~8–16 MAJOR-BLOCK nodes laid out left-to-right: one node per distinct input, "
        "the backbone, each fusion/merge, the policy/decoder head, each output, each train-only "
        "auxiliary block, each learning method, and each loss. Each node has title + 1-line desc "
        "+ a paren-string shape + role + train_only. List EVERY dataflow edge (no implied spine), "
        "plus loss/feedback/skip edges. Add a groups[] entry {id:'train', label:'ONLY DURING "
        "TRAINING'} if any node is train_only. Output ONLY the JSON object.",
    ]
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of arbitrary text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in model output")


# The architecture-reading task benefits from the strongest available model + reasoning
# budget: Claude must read every collected class, follow the dependency graph, and cross-check
# its own diagram against the code. Defaults chosen deliberately (override via CLI if needed).
#   NOTE: use the bare full name `claude-opus-4-8-v1` — the `global.anthropic.` prefixed id
#   silently falls back to an older Opus on this distribution.
DEFAULT_MODEL = "claude-opus-4-8-v1"
DEFAULT_EFFORT = "max"  # one of: low, medium, high, xhigh, max


def _run_claude(user_prompt: str, system_prompt: str, model: str | None,
                effort: str | None, timeout: int) -> str:
    cmd = ["claude", "-p", user_prompt, "--append-system-prompt", system_prompt,
           "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_HERE.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}):\n{proc.stderr[:2000]}")
    if not proc.stdout.strip():
        raise RuntimeError(f"claude CLI returned empty output. stderr: {proc.stderr[:500]}")
    return proc.stdout


def extract_arch(bundle: Bundle, model: str | None = None, effort: str | None = None,
                 timeout: int = 900) -> dict[str, Any]:
    """Invoke Claude to produce the left-to-right architecture IR (arch_v1) for the bundle.

    Defaults to the strongest model at max reasoning effort (see DEFAULT_MODEL/DEFAULT_EFFORT);
    pass explicit values to override. A higher timeout accommodates max-effort reasoning over
    large multi-file bundles."""
    if not claude_available():
        raise RuntimeError(
            "`claude` CLI not found on PATH. Either install it, or pass a pre-computed "
            "arch IR via `--arch <file.json>` to skip the LLM stage."
        )
    system_prompt = _ARCH_PROMPT.read_text(encoding="utf-8")
    user_prompt = build_arch_prompt(bundle)
    out = _run_claude(user_prompt, system_prompt,
                      model or DEFAULT_MODEL, effort or DEFAULT_EFFORT, timeout)
    return _extract_json_object(out)

