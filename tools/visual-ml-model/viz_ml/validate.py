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


def validate_schema(ir: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_node(ir, schema, "$", errors)
    return errors


# ---------------------------------------------------------------------------
# structural invariants
# ---------------------------------------------------------------------------

def _has_cycle(adj: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in adj}

    def visit(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color.get(v) == GRAY:
                return True
            if color.get(v) == WHITE and visit(v):
                return True
        color[u] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in adj)


def validate_arch_structure(ir: dict[str, Any]) -> list[str]:
    """Structural invariants for the arch_v1 IR (top-level nodes/edges/groups).

    - every edge `from`/`to` resolves to a node id
    - every group member resolves to a node id
    - the dataflow sub-graph (kind=='dataflow') is acyclic (so layering terminates;
      feedback/loss/skip edges are excluded, mirroring the ir_v1 / flow conventions)
    - warns (does not fail) when there is no input/output node
    """
    errors: list[str] = []
    nodes = ir.get("nodes", [])
    edges = ir.get("edges", [])
    ids: dict[str, dict] = {}
    for n in nodes:
        nid = n.get("id")
        if nid in ids:
            errors.append(f"duplicate node id: {nid!r}")
        ids[nid] = n

    for e in edges:
        for end in ("from", "to"):
            ref = e.get(end)
            if ref not in ids:
                errors.append(f"edge {e.get('from')!r}->{e.get('to')!r}: {end} {ref!r} does not resolve")

    for g in ir.get("groups", []):
        for m in g.get("members", []) or []:
            if m not in ids:
                errors.append(f"group {g.get('id')!r}: member {m!r} does not resolve")

    # dataflow acyclicity (so longest-path layering terminates)
    adj: dict[str, list[str]] = {nid: [] for nid in ids}
    for e in edges:
        if e.get("kind") == "dataflow":
            s, t = e.get("from"), e.get("to")
            if s in adj and t in ids:
                adj[s].append(t)
    if _has_cycle(adj):
        errors.append("dataflow edges contain a cycle (left-to-right layering requires acyclicity; "
                      "use kind='feedback' for recurrent/loop-back edges)")

    roles = {n.get("role") for n in nodes}
    warnings: list[str] = []
    if "input" not in roles:
        warnings.append("note: no node has role 'input' (left column may be empty)")
    if "output" not in roles and "loss" not in roles:
        warnings.append("note: no 'output' or 'loss' node (right column may be empty)")
    return errors + warnings
