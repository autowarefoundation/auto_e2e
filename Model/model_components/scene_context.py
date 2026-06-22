from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class CausalReasoningContext:
    """Structured output for causal reasoning with confidence and provenance."""
    reasoning_latent: torch.Tensor          # [B, latent_dim]
    causal_class_logits: torch.Tensor       # [B, num_classes]
    confidence: torch.Tensor                # [B] scalar probability
    provenance: str = "vlm_causal_head"     # origin of this reasoning


@dataclass
class SceneContext:
    """Typed interface for structured scene context passed to trajectory planners.
    
    Acts as a standardized bus for high-level scene understanding to avoid locking
    the planner into a single fixed head. Planners can optionally consume these
    fields (e.g. VLM outputs, HD map info, route commands).
    """
    causal_reasoning: Optional[CausalReasoningContext] = None
