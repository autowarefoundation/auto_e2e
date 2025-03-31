"""viz_ml CLI — read PyTorch source, render a left-to-right architecture diagram.

Commands:
  arch     source.py --class Net --config c.json -o net.arch.html [--save-ir net.arch.json]
             Stage 0 (resolve) -> Stage 1 (AST facts) -> Stage 3 (Claude -> arch_v1 IR)
             -> validate + render a self-contained architecture-diagram HTML.
             Use --arch <file.json> to render a pre-computed/hand-edited IR (no Claude call).
  variants source.py --class Net [--config c.json]
             List the registry/factory variants the model can select among.
  facts    source.py --class Net [--config c.json]
             Print the Stage 0/1 code bundle + AST facts (no LLM). For inspection.
  validate net.arch.json
             Validate an arch_v1 IR file against the schema + structural invariants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .resolve import resolve, load_config, bundle_to_facts_dict

_ARCH_SCHEMA = str(Path(__file__).resolve().parent.parent / "schema" / "arch_v1.schema.json")


def _eprint(*a):
    print(*a, file=sys.stderr)


def _load_ir(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_facts(args) -> int:
    cfg = load_config(args.config)
    bundle = resolve(args.source, args.target_class, cfg)
    out = {
        "entry_class": bundle.entry_class,
        "source_files": bundle.source_files,
        "config": bundle.config,
        "collected_classes": list(bundle.classes.keys()),
        "facts": bundle_to_facts_dict(bundle),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_variants(args) -> int:
    """List registry/factory variants the model can select among (and which config picks)."""
    cfg = load_config(args.config)
    bundle = resolve(args.source, args.target_class, cfg)
    if not bundle.registry_options:
        print(f"No registry/factory variants found for `{bundle.entry_class}`.")
        return 0
    by_reg: dict[str, list] = {}
    for o in bundle.registry_options:
        by_reg.setdefault(o.registry, []).append(o)
    print(f"Registry variants for `{bundle.entry_class}`:")
    for reg, opts in by_reg.items():
        print(f"\n  {reg}:")
        for o in opts:
            mark = "  ◀ ACTIVE (selected by config)" if o.active else ""
            print(f'    "{o.key}"  ->  {o.class_name}{mark}')
    if not any(o.active for o in bundle.registry_options):
        print("\n(none selected — pass a config whose value matches a key above, "
              'e.g. {"fusion_mode": "bev"})')
    return 0


def cmd_validate(args) -> int:
    from .validate import load_schema, validate_schema, validate_arch_structure
    ir = _load_ir(args.ir)
    errors = validate_schema(ir, load_schema(_ARCH_SCHEMA)) + validate_arch_structure(ir)
    errors = [e for e in errors if not e.startswith("note:")]
    if not errors:
        print("VALID (arch_v1)")
        return 0
    print(f"INVALID (arch_v1) — {len(errors)} issue(s):")
    for e in errors:
        print("  -", e)
    return 1


def cmd_arch(args) -> int:
    """Generate the left-to-right architecture diagram (paper/README-figure style)."""
    from .arch_render import render_arch_html
    from .validate import load_schema, validate_schema, validate_arch_structure

    if args.arch:
        arch = _load_ir(args.arch)
        _eprint(f"[arch] using supplied arch IR: {args.arch}")
        title = args.title
    else:
        from .extract import extract_arch, claude_available
        cfg = load_config(args.config)
        bundle = resolve(args.source, args.target_class, cfg)
        if not claude_available():
            _eprint("error: `claude` CLI not found and no --arch supplied. "
                    "Pass --arch <file.json> to render a pre-computed arch IR.")
            return 2
        _eprint(f"[stage 0/1] resolved `{bundle.entry_class}`: {list(bundle.classes.keys())}")
        if bundle.registry_options:
            act = ", ".join(sorted(bundle.active_variant_classes())) or "(none selected)"
            _eprint(f"[variants] active: {act}")
        _eprint(f"[arch] invoking claude (model={args.model or 'default'}) ...")
        arch = extract_arch(bundle, model=args.model, timeout=args.timeout)
        title = args.title or bundle.entry_class
