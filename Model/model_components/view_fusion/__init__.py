from .bev_fusion import BEVViewFusion

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
