from .conv_next_v2_tiny import ConvNextV2Tiny
from .swin_v2_tiny import SwinV2Tiny

BACKBONE_REGISTRY = {
    "swin_tiny": ConvNextV2Tiny,
    "swin_v2_tiny": SwinV2Tiny,
}


def build_backbone(backbone):
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone]
