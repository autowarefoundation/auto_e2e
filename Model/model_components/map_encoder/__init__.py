import torch.nn as nn
from .lightweight_route_encoder import LightweightRouteEncoder
from .raster_map_encoder import RasterizedMapEncoder
from .semantic_raster_encoder import SemanticRasterEncoder
from .split_navigation_encoder import SplitNavigationEncoder
from .map_bev_fusion import MAP_FUSION_REGISTRY, build_map_bev_fusion

MAP_ENCODER_REGISTRY = {
    "rasterized": RasterizedMapEncoder,
    "semantic_raster": SemanticRasterEncoder,
}


def build_map_encoder(map_type: str, **kwargs) -> nn.Module:
    """Construct a map encoder by registry name.

    Args:
        map_type: One of the keys in ``MAP_ENCODER_REGISTRY``
            (currently only ``"rasterized"``).
        **kwargs: Forwarded to the selected encoder constructor
            (``embed_dim``, ``output_h``, ``output_w``, ``in_channels``).

    """
    if map_type not in MAP_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown map_type '{map_type}'. "
            f"Available: {list(MAP_ENCODER_REGISTRY.keys())}"
        )
    return MAP_ENCODER_REGISTRY[map_type](**kwargs)


def build_split_navigation_encoder(
    map_type: str,
    *,
    map_channels: int,
    route_channels: int,
    embed_dim: int,
    output_h: int,
    output_w: int,
    route_hidden_channels: int = 64,
) -> SplitNavigationEncoder:
    map_encoder = build_map_encoder(
        map_type,
        in_channels=map_channels,
        embed_dim=embed_dim,
        output_h=output_h,
        output_w=output_w,
    )
    return SplitNavigationEncoder(
        map_encoder,
        route_channels=route_channels,
        embed_dim=embed_dim,
        output_h=output_h,
        output_w=output_w,
        route_hidden_channels=route_hidden_channels,
    )


__all__ = [
    "MAP_ENCODER_REGISTRY",
    "MAP_FUSION_REGISTRY",
    "build_map_encoder",
    "build_split_navigation_encoder",
    "build_map_bev_fusion",
    "LightweightRouteEncoder",
    "RasterizedMapEncoder",
    "SemanticRasterEncoder",
    "SplitNavigationEncoder",
]
