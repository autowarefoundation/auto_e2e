import timm

BACKBONE_REGISTRY = {
    "swin_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "swinv2_tiny_window8_256", pretrained=pretrained, features_only=True, **kwargs
    ),
    "conv_next_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "convnextv2_tiny", pretrained=pretrained, features_only=True, **kwargs
    ),
}

def build_backbone(backbone, pretrained=True, **kwargs):
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone](pretrained=pretrained, **kwargs)