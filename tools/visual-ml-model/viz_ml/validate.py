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


def _validate_node(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return

    if "const" in schema:
        if value != schema["const"]:
            errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
        return

    if "enum" in schema:
        if value not in schema["enum"]:
            errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
        # continue to type checks if any

    if "type" in schema and value is not None:
        if not _type_ok(value, schema["type"]):
            errors.append(f"{path}: expected type {schema['type']}, got {type(value).__name__}")
            return
    elif "type" in schema and value is None:
        if not _type_ok(None, schema["type"]):
            errors.append(f"{path}: null not allowed (type {schema['type']})")
            return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property '{req}'")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path}: additional property '{k}' not allowed")
        for k, v in value.items():
            if k in props:
                _validate_node(v, props[k], f"{path}.{k}", errors)

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate_node(item, schema["items"], f"{path}[{i}]", errors)
