from .bev_fusion import BEVViewFusion
from .projection import (
    GEOMETRY_FTHETA,
    GEOMETRY_PINHOLE,
    GEOMETRY_PSEUDO,
    GEOMETRY_RECTIFIED_PINHOLE,
    VALID_GEOMETRY_TYPES,
    FThetaProjection,
    PinholeProjection,
    ProjectionResult,
    PseudoProjection,
)

# Re-exported public API (projection operators + geometry labels). Declared in
# __all__ so linters treat these imports as intentional re-exports, not unused.
__all__ = [
    "BEVViewFusion",
    "FUSION_REGISTRY",
    "build_view_fusion",
    "ProjectionResult",
    "PinholeProjection",
    "PseudoProjection",
    "FThetaProjection",
    "GEOMETRY_PINHOLE",
    "GEOMETRY_RECTIFIED_PINHOLE",
    "GEOMETRY_FTHETA",
    "GEOMETRY_PSEUDO",
    "VALID_GEOMETRY_TYPES",
]

FUSION_REGISTRY = {
    "bev": BEVViewFusion,
}


def build_view_fusion(fusion_mode, num_views, embed_dim=256, **kwargs):
    if fusion_mode not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion_mode '{fusion_mode}'. "
            f"Available: {list(FUSION_REGISTRY.keys())}"
        )
    return FUSION_REGISTRY[fusion_mode](
        num_views=num_views, embed_dim=embed_dim, **kwargs
    )
