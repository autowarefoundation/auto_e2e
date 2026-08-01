from .residual_fusion import ResidualMapFusion
from .cross_attention_fusion import MapCrossAttentionFusion
from .deformable_cross_attention_fusion import MapDeformableCrossAttentionFusion

MAP_FUSION_REGISTRY = {
    "residual": ResidualMapFusion,
    "cross_attn": MapCrossAttentionFusion,
    "deformable": MapDeformableCrossAttentionFusion,
}


def build_map_bev_fusion(fusion_mode: str, embed_dim: int = 256, **kwargs):
    """Construct a map BEV fusion module.

    Args:
        fusion_mode: One of ``"residual"``, ``"cross_attn"``, or
            ``"deformable"``.
        embed_dim: Channel dimension shared by image and map BEV features.
    """
    if fusion_mode not in MAP_FUSION_REGISTRY:
        raise ValueError(
            f"Unknown map_fusion_mode '{fusion_mode}'. "
            f"Available: {list(MAP_FUSION_REGISTRY.keys())}"
        )
    return MAP_FUSION_REGISTRY[fusion_mode](embed_dim=embed_dim, **kwargs)
