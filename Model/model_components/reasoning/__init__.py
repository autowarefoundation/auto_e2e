"""Reasoning band — scenario classification head for AutoE2E (issue #98).

This package is opt-in (default OFF); importing it never pulls heavy
dependencies (the Qwen2-VL backend uses a lazy import inside its module).
"""

from .scenario_taxonomy import ScenarioTaxonomy, TaxonomyGroup
from .reasoning_band import ReasoningBand

__all__ = ["ScenarioTaxonomy", "TaxonomyGroup", "ReasoningBand"]
