"""Stage 0 — input resolution.

Given a source file, a target class name, and a concrete config, this:
  - locates the target class
  - follows SAME-REPO imports to pull in the submodule/base classes it references
  - assembles a compact "code bundle" (only the relevant .py slices) so the LLM gets the
    target module + everything it depends on, without the whole repo

MVP scope: same-repo (relative + sibling-file) imports only; no third-party following.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ast_facts import extract_classes, ClassFacts, facts_to_dict, _name_of


@dataclass
class CollectedClass:
    name: str
    file: str
    source_segment: str


@dataclass
class RegistryOption:
    """A registry/factory variant: a string key -> class, with whether it's the selected one."""
    registry: str          # e.g. "FUSION_REGISTRY"
    key: str               # e.g. "bev"
    class_name: str        # e.g. "BEVViewFusion"
    active: bool           # True if selected by config (or the only/default option)


@dataclass
class Bundle:
    entry_class: str
    entry_file: str
    config: dict[str, Any]
    classes: dict[str, CollectedClass] = field(default_factory=dict)   # name -> class
    facts: dict[str, ClassFacts] = field(default_factory=dict)         # name -> AST facts
    source_files: list[str] = field(default_factory=list)
    registry_options: list[RegistryOption] = field(default_factory=list)

    def bundle_source(self) -> str:
        """Concatenated, de-duplicated source of all collected classes (for the LLM)."""
        parts: list[str] = []
        for name, cc in self.classes.items():
            parts.append(f"# ===== class {name}  (from {cc.file}) =====\n{cc.source_segment}")
        return "\n\n".join(parts)

    def active_variant_classes(self) -> set[str]:
        return {o.class_name for o in self.registry_options if o.active}

    def inactive_variant_classes(self) -> set[str]:
        active = self.active_variant_classes()
        return {o.class_name for o in self.registry_options if not o.active} - active


def _module_classes(source: str) -> dict[str, str]:
    """name -> source segment for every class defined in `source`."""
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            seg = ast.get_source_segment(source, node)
            if seg:
                out[node.name] = seg
    return out


def _local_import_targets(source: str) -> set[str]:
    """Names imported via local/relative imports (candidates for same-repo following)."""
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def _referenced_names(facts: ClassFacts) -> set[str]:
    """Class names this class references: base classes + submodule constructors."""
    refs: set[str] = set(facts.bases)
    for sm in facts.submodules:
        if sm.constructor:
            # take the trailing identifier of a dotted ctor (nn.Linear -> Linear,
            # CausalSelfAttention -> CausalSelfAttention)
            refs.add(sm.constructor.split(".")[-1])
            refs.add(sm.constructor)
    return refs


def _registry_maps(source: str) -> dict[str, dict[str, str]]:
    """Module-level registry dicts mapping a string key -> class name.

    e.g. FUSION_REGISTRY = {"concat": ConcatViewFusion, "bev": BEVViewFusion}
    -> {"FUSION_REGISTRY": {"concat": "ConcatViewFusion", "bev": "BEVViewFusion"}}
    These let the user SELECT a variant (via config) and let us tell the LLM which
    concrete class is actually active.
    """
    tree = ast.parse(source)
    out: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            kv: dict[str, str] = {}
            for k, v in zip(node.value.keys, node.value.values):
                cls = _name_of(v)
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and cls:
                    kv[k.value] = cls.split(".")[-1]
            if kv:
                for tgt in node.targets:
                    tname = _name_of(tgt)
                    if tname:
                        out[tname] = kv
    return out
