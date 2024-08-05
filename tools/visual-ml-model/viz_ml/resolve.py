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
