"""Adapter registry: maps dataset names to IngestAdapter implementations."""

from __future__ import annotations

from .protocol import IngestAdapter


def get_adapter(name: str) -> IngestAdapter:
    """Instantiate an adapter by name. Lazy imports to avoid heavy deps at module level."""
    if name == "l2d":
        from .l2d import L2DAdapter
        return L2DAdapter()
    elif name == "nvidia_av":
        from .nvidia_av import NvidiaAVAdapter
        return NvidiaAVAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}. Available: l2d, nvidia_av")
