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


def _factory_class_refs(source: str) -> dict[str, set[str]]:
    """Map repo-defined FUNCTION name -> set of class names it can return.

    Handles the Registry/factory pattern where a submodule is built indirectly, e.g.
        self.view_fusion = build_view_fusion(fusion_mode, ...)
    where `build_view_fusion` does `return FUSION_REGISTRY[mode](...)` and
        FUSION_REGISTRY = {"concat": ConcatViewFusion, "bev": BEVViewFusion, ...}
    Pure AST can't know which branch runs (that depends on config), so we collect ALL
    candidate classes and let the LLM pick the one matching the concrete config.

    We resolve, per function:
      - direct `return SomeClass(...)` / `return SomeClass`
      - `return REGISTRY[...](...)` where REGISTRY is a module-level dict of name->Class
      - names assigned then returned (one hop): `x = SomeClass(...); return x`
    """
    tree = ast.parse(source)

    # module-level registries: dict literals whose values are class-name references
    registries: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            cls_vals = {
                _name_of(v).split(".")[-1]
                for v in node.value.values
                if _name_of(v)
            }
            if cls_vals:
                for tgt in node.targets:
                    tname = _name_of(tgt)
                    if tname:
                        registries[tname] = cls_vals

    func_refs: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        refs: set[str] = set()
        # local one-hop var -> class assignments inside the function
        local_assign: dict[str, set[str]] = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                cname = _name_of(sub.value.func)
                if cname:
                    for t in sub.targets:
                        tn = _name_of(t)
                        if tn:
                            local_assign.setdefault(tn, set()).add(cname.split(".")[-1])

        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and sub.value is not None:
                rv = sub.value
                call = rv if isinstance(rv, ast.Call) else None
                target = call.func if call else rv
                # return REGISTRY[...](...)  -> the subscripted name is a registry
                if isinstance(target, ast.Subscript):
                    base = _name_of(target.value)
                    if base in registries:
                        refs |= registries[base]
                else:
                    nm = _name_of(target)
                    if nm:
                        short = nm.split(".")[-1]
                        if short in local_assign:
                            refs |= local_assign[short]
                        else:
                            refs.add(short)
        if refs:
            func_refs[node.name] = refs
    return func_refs


def _find_class_in_repo(class_name: str, repo_files: list[Path]) -> tuple[Path, str] | None:
    for f in repo_files:
        try:
            src = f.read_text(encoding="utf-8")
        except Exception:
            continue
        classes = _module_classes(src)
        if class_name in classes:
            return f, classes[class_name]
    return None


def _repo_python_files(root: Path, max_files: int = 400) -> list[Path]:
    files: list[Path] = []
    skip = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
                if len(files) >= max_files:
                    return files
    return files


def resolve(entry_file: str, target_class: str, config: dict[str, Any]) -> Bundle:
    """Build a code bundle for `target_class` defined in `entry_file`.

    Follows references (base classes + submodule constructors) across .py files in the same
    repository directory, transitively, ignoring obvious third-party names (nn.*, torch.*).
    """
    entry_path = Path(entry_file).resolve()
    if not entry_path.exists():
        raise FileNotFoundError(f"entry file not found: {entry_file}")

    repo_root = entry_path.parent
    repo_files = _repo_python_files(repo_root)
    # ensure the entry file is searched first
    repo_files = [entry_path] + [f for f in repo_files if f.resolve() != entry_path]

    bundle = Bundle(entry_class=target_class, entry_file=str(entry_path), config=config)

    # Repo-wide factory/registry index: function name -> candidate classes it can return.
    # Lets us follow `self.x = build_something(mode, ...)` to the concrete classes (the
    # Registry pattern), which a plain ctor-name scan would miss.
    factory_index: dict[str, set[str]] = {}
    registry_index: dict[str, dict[str, str]] = {}   # registry name -> {key: class}
    for rf in repo_files:
        try:
            text = rf.read_text(encoding="utf-8")
            factory_index.update(_factory_class_refs(text))
            registry_index.update(_registry_maps(text))
        except Exception:
            continue

    # Framework names we never try to follow (dotted access like nn.Linear, torch.*, F.*).
    # Bare names (e.g. "LayerNorm") are NOT skipped on sight: if the repo defines a class
    # of that name (a custom LayerNorm/RMSNorm/Attention), we want to collect it; only if
    # it isn't found locally do we treat it as a framework class and silently skip.
    builtin_prefixes = ("nn.", "torch.", "F.")

    to_visit = [target_class]
    visited: set[str] = set()
    seen_files: set[str] = set()

    while to_visit:
        name = to_visit.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name.startswith(builtin_prefixes):
            continue

        found = _find_class_in_repo(name, repo_files)
        if found is None:
            # not defined in the repo -> a framework/third-party class; skip quietly
            continue
        f, seg = found
        bundle.classes[name] = CollectedClass(name=name, file=str(f), source_segment=seg)
        if str(f) not in seen_files:
            seen_files.add(str(f))
            bundle.source_files.append(str(f))

        # extract facts for this class and enqueue what it references
        file_src = f.read_text(encoding="utf-8")
        file_facts = extract_classes(file_src)
        if name in file_facts:
            bundle.facts[name] = file_facts[name]
            for ref in _referenced_names(file_facts[name]):
                short = ref.split(".")[-1]
                if short not in visited and not ref.startswith(builtin_prefixes):
                    to_visit.append(short)
                # if this reference is actually a repo factory function, also enqueue
                # the concrete classes it can return (Registry/factory resolution)
                if short in factory_index:
                    for cand in factory_index[short]:
                        if cand not in visited:
                            to_visit.append(cand)

    if target_class not in bundle.classes:
        raise ValueError(
            f"target class '{target_class}' not found in {entry_file} or its same-repo imports"
        )
