import torch.nn as nn
from .raster_map_encoder import RasterizedMapEncoder
from .map_bev_fusion import MAP_FUSION_REGISTRY, build_map_bev_fusion

MAP_ENCODER_REGISTRY = {
    "rasterized": RasterizedMapEncoder,
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


__all__ = [
    "MAP_ENCODER_REGISTRY",
    "MAP_FUSION_REGISTRY",
    "build_map_encoder",
    "build_map_bev_fusion",
    "RasterizedMapEncoder",
]