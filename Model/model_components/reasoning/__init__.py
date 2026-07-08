"""Reasoning branch — horizon-aware, action-relevant reasoning head (issue #98).

Opt-in (default OFF). Runtime-safe: importing this package pulls NO teacher
model or client — teacher supervision is generated offline (see
``data_processing.reasoning_label_generation``) and consumed as frozen labels.
"""

from .horizon_reasoning_head import HorizonReasoningHead
from .reasoning_taxonomy import (
    DEFAULT_TAXONOMY,
    LabelMode,
    ReasoningTaxonomy,
    TaxonomyGroup,
)
from .types import HorizonReasoningPrediction

__all__ = [
    "HorizonReasoningHead",
    "HorizonReasoningPrediction",
    "ReasoningTaxonomy",
    "TaxonomyGroup",
    "LabelMode",
    "DEFAULT_TAXONOMY",
]
