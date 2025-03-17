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
