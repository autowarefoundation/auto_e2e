"""Prompt construction + response parsing for teacher endpoints (issue #98).

Builds the closed-JSON prompt over the compositional taxonomy and parses the
teacher's reply into a :class:`ReasoningLabelRecord`. Tolerant parsing: chatter
around the JSON, unknown labels, and missing horizons degrade to abstain rather
than raising — the caller's ``strict`` flag decides what to do with an
empty/malformed answer (R9).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from model_components.reasoning.reasoning_taxonomy import LabelMode, ReasoningTaxonomy

from .schema import (
    HORIZON_SECONDS,
    NUM_HORIZONS,
    SCHEMA_VERSION,
    ReasoningHorizonLabel,
    ReasoningLabelRecord,
)

_SYSTEM_PROMPT = (
    "You are an autonomous-driving reasoning labeller. Return ONLY valid JSON "
    "matching the provided schema. Use only the exact label strings given. Use "
    "unknown_* / no_* / none labels when uncertain. Do not invent labels."
)

# The core action-relevant groups the prompt asks for, in a fixed order.
_CORE_GROUPS = (
    "relation_to_ego", "hazard_event", "cause",
    "longitudinal_response", "lateral_response", "tactical_response", "rule_response",
)


def system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_clip_prompt(
    taxonomy: ReasoningTaxonomy, extra_context: Optional[str] = None
) -> str:
    """Prompt asking for all five horizons of action-relevant labels in one JSON."""
    lines = [
        f"Label a short driving clip of {NUM_HORIZONS} front-camera frames, "
        "ordered from the current frame (horizon 0 s) to the +4 s future frame.",
        "For EACH horizon, choose labels from the categories below.",
        "Single-label groups take exactly one string; multi-label groups take a list.",
        "",
        "Categories:",
    ]
    for group in _CORE_GROUPS:
        g = taxonomy[group]
        kind = "one of" if g.mode is LabelMode.SINGLE else "any of"
        lines.append(f"- {group} ({kind}): {', '.join(g.labels)}")
    if extra_context:
        lines += ["", "Scene context (ego / route / map):", extra_context]
    def _slot(group: str) -> str:
        placeholder = '"..."' if taxonomy[group].mode is LabelMode.SINGLE else "[...]"
        return f'"{group}": {placeholder}'

    example_horizon = (
        '{"horizon_sec": 0, '
        + ", ".join(_slot(g) for g in _CORE_GROUPS)
        + ', "confidence": 0.0, "evidence": "..."}'
    )
    lines += [
        "",
        "Answer with ONLY a JSON object of the form:",
        '{"horizons": [' + example_horizon + ", ...]}",
        f"with exactly {NUM_HORIZONS} horizon entries, in order (0,1,2,3,4 s).",
    ]
    return "\n".join(lines)


def _coerce_horizon(
    entry: Dict[str, Any], horizon_sec: float, taxonomy: ReasoningTaxonomy, provenance: str
) -> ReasoningHorizonLabel:
    """Coerce one raw horizon dict into a validated ReasoningHorizonLabel.

    Unknown labels are dropped; a single-label group with no valid value stays
    None (masked out downstream). This never raises — malformed content becomes
    an abstaining (empty) horizon.
    """
    kwargs: Dict[str, Any] = {"horizon_sec": horizon_sec, "provenance": provenance}
    for group in _CORE_GROUPS:
        allowed = set(taxonomy.labels(group))
        raw = entry.get(group)
        if taxonomy[group].mode is LabelMode.MULTI:
            values = raw if isinstance(raw, list) else []
            kwargs[group] = [v for v in values if isinstance(v, str) and v in allowed]
        else:
            kwargs[group] = raw if isinstance(raw, str) and raw in allowed else None
    conf = entry.get("confidence", 0.0)
    kwargs["confidence"] = float(conf) if isinstance(conf, (int, float)) else 0.0
    kwargs["confidence"] = min(1.0, max(0.0, kwargs["confidence"]))
    ev = entry.get("evidence")
    kwargs["evidence"] = ev if isinstance(ev, str) else None
    return ReasoningHorizonLabel(**kwargs)


def parse_clip_response(
    text: str,
    taxonomy: ReasoningTaxonomy,
    *,
    sample_id: str,
    dataset_name: str,
    provider: str,
    model: str,
    prompt_version: str,
    request_mode: str,
    timestamp: float = 0.0,
    provenance: str = "teacher_gt",
) -> Optional[ReasoningLabelRecord]:
    """Parse a clip response into a full record, or None if unparseable.

    Returns None (the caller decides strict-vs-abstain) when the JSON is missing,
    malformed, or does not contain exactly :data:`NUM_HORIZONS` horizon entries.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        raw, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    horizons_raw = raw.get("horizons")
    if not isinstance(horizons_raw, list) or len(horizons_raw) < NUM_HORIZONS:
        return None

    horizons: List[ReasoningHorizonLabel] = []
    for i, sec in enumerate(HORIZON_SECONDS):
        entry = horizons_raw[i] if isinstance(horizons_raw[i], dict) else {}
        horizons.append(_coerce_horizon(entry, sec, taxonomy, provenance))

    return ReasoningLabelRecord(
        schema_version=SCHEMA_VERSION,
        sample_id=sample_id,
        timestamp=timestamp,
        dataset_name=dataset_name,
        teacher_provider=provider,
        teacher_model=model,
        prompt_version=prompt_version,
        request_mode=request_mode,
        horizons=horizons,
        provenance=provenance,
    )
