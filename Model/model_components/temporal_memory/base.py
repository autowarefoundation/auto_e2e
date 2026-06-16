import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTemporalMemory(nn.Module, ABC):
    """Abstract interface for temporal memory modules.
    
    Transforms [B, T, feat] sequence histories into [B, feat] context vectors
    that are consumed downstream by the trajectory planner.
    """
    @abstractmethod
    def forward(self, visual_history, egomotion_history, **kwargs):
        """
        Args:
            visual_history: [B, T, visual_history_dim] or [B, visual_history_dim]
            egomotion_history: [B, T, egomotion_dim] or [B, egomotion_dim]
            
        Returns:
            visual_context: [B, visual_history_dim]
            egomotion_context: [B, egomotion_dim]
        """
        raise NotImplementedError
