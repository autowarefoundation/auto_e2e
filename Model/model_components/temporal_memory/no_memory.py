from .base import BaseTemporalMemory

class NoMemory(BaseTemporalMemory):
    """Baseline temporal memory: extracts the most recent timestep or passes through flat contexts."""
    def forward(self, visual_history, egomotion_history, **kwargs):
        v_ctx = visual_history[:, -1] if visual_history.ndim == 3 else visual_history
        e_ctx = egomotion_history[:, -1] if egomotion_history.ndim == 3 else egomotion_history
        return v_ctx, e_ctx
