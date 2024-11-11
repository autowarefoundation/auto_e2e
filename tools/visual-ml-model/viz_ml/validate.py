"""Validation — stdlib only (no jsonschema dependency).

Two layers:
  1. validate_schema(): a small JSON-Schema interpreter covering the subset arch_v1 uses
     (type, enum, const, required, properties, items, minimum/maximum, additionalProperties).
  2. validate_arch_structure(): structural invariants for the arch_v1 IR — edge endpoints
     resolve, group members resolve, the dataflow sub-graph is acyclic (so left-to-right
     layering terminates), with soft notes when inputs/outputs are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_CACHE: dict[str, Any] = {}


def load_schema(schema_path: str) -> dict[str, Any]:
    if schema_path not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[schema_path] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return _SCHEMA_CACHE[schema_path]

# ---------------------------------------------------------------------------
# minimal JSON-Schema validation (subset)
# ---------------------------------------------------------------------------

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, type_spec: Any) -> bool:
    types = type_spec if isinstance(type_spec, list) else [type_spec]
    for t in types:
        py = _JSON_TYPES.get(t)
        if py is None:
            continue
        # bool is a subclass of int; keep them distinct
        if t == "integer" and isinstance(value, bool):
            continue
        if t == "number" and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False
