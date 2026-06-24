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


@dataclass
class BufferDecl:
    name: str                     # buffer name, e.g. "bias"
    persistent: bool | None       # explicit persistent= flag (default True if omitted)
    arg_summary: str = ""         # short source summary of the tensor expression


@dataclass
class ForwardStmt:
    line: int
    targets: list[str]            # lhs names, e.g. ["x"] or ["q", "k", "v"]
    calls: list[str]              # attribute calls invoked, e.g. ["self.attn", "self.ln_1"]
    has_add: bool                 # a binary '+' appears in the statement (residual signal)
    source: str                   # the raw source line(s), trimmed


@dataclass
class ClassFacts:
    name: str
    bases: list[str]
    submodules: list[Submodule] = field(default_factory=list)
    buffers: list[BufferDecl] = field(default_factory=list)
    forward_skeleton: list[ForwardStmt] = field(default_factory=list)
    has_forward: bool = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _name_of(node: ast.AST) -> str | None:
    """Render a dotted name like nn.Linear or self.attn from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _src(node: ast.AST, source: str) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if seg is not None:
            return " ".join(seg.split())
    except Exception:
        pass
    return ""


def _assign_targets(node: ast.AST) -> list[str]:
    """Names assigned to on the lhs of an assignment, including tuple unpacking."""
    out: list[str] = []

    def walk(t: ast.AST) -> None:
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                walk(e)
        else:
            n = _name_of(t)
            if n:
                out.append(n)

    walk(node)
    return out


def _calls_in(node: ast.AST) -> list[str]:
    """Attribute/name calls invoked anywhere in an expression (in source order-ish)."""
    found: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            n = _name_of(sub.func)
            if n:
                found.append(n)
    return found


def _has_add(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Add):
            return True
        # in-place: x += ...
        if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
            return True
    return False


# ---------------------------------------------------------------------------
# per-class extraction
# ---------------------------------------------------------------------------

def _extract_init(cls: ast.ClassDef, source: str, facts: ClassFacts) -> None:
    init = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return
    for stmt in ast.walk(init):
        # self.<name> = <Class>(...)
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            for tgt in stmt.targets:
                name = _name_of(tgt)
                if not name or not name.startswith("self."):
                    continue
                ctor = _name_of(stmt.value.func)
                args = [_src(a, source) for a in stmt.value.args]
                kwargs = {
                    (kw.arg or "**"): _src(kw.value, source)
                    for kw in stmt.value.keywords
                }
                facts.submodules.append(
                    Submodule(var_name=name, constructor=ctor, args=args, kwargs=kwargs)
                )
        # self.register_buffer("name", tensor, persistent=?)
        if isinstance(stmt, ast.Call):
            fn = _name_of(stmt.func)
            if fn and fn.endswith("register_buffer"):
                buf_name = None
                if stmt.args and isinstance(stmt.args[0], ast.Constant):
                    buf_name = stmt.args[0].value
                persistent: bool | None = True  # torch default when omitted
                for kw in stmt.keywords:
                    if kw.arg == "persistent" and isinstance(kw.value, ast.Constant):
                        persistent = bool(kw.value.value)
                arg_summary = _src(stmt.args[1], source) if len(stmt.args) > 1 else ""
                if buf_name is not None:
                    facts.buffers.append(
                        BufferDecl(
                            name=str(buf_name),
                            persistent=persistent,
                            arg_summary=arg_summary[:160],
                        )
                    )


def _extract_forward(cls: ast.ClassDef, source: str, facts: ClassFacts) -> None:
    fwd = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"),
        None,
    )
    if fwd is None:
        return
    facts.has_forward = True
    for stmt in fwd.body:
        # only summarize top-level statements of forward (keeps the skeleton compact)
        targets: list[str] = []
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                targets += _assign_targets(t)
        elif isinstance(stmt, ast.AugAssign):
            targets += _assign_targets(stmt.target)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            targets = ["<return>"]

        calls = _calls_in(stmt)
        # keep only self.* calls (the semantically meaningful submodule invocations) +
        # functional calls (F.*, torch.*) which the LLM may need
        meaningful = [
            c for c in calls
            if c.startswith("self.") or c.split(".")[0] in {"F", "torch", "nn"}
        ]
        facts.forward_skeleton.append(
            ForwardStmt(
                line=getattr(stmt, "lineno", -1),
                targets=targets,
                calls=meaningful,
                has_add=_has_add(stmt),
                source=_src(stmt, source)[:240],
            )
        )


def extract_classes(source: str) -> dict[str, ClassFacts]:
    """Parse source text and return AST facts for every class defined in it."""
    tree = ast.parse(source)
    result: dict[str, ClassFacts] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            facts = ClassFacts(
                name=node.name,
                bases=[b for b in (_name_of(b) for b in node.bases) if b],
            )
            _extract_init(node, source, facts)
            _extract_forward(node, source, facts)
            result[node.name] = facts
    return result


def facts_to_dict(facts: dict[str, ClassFacts]) -> dict[str, Any]:
    """JSON-serializable view of the extracted facts."""
    return {name: asdict(cf) for name, cf in facts.items()}
