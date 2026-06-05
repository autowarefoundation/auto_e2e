import timm

BACKBONE_REGISTRY = {
    "swin_v2_tiny": lambda **kwargs: timm.create_model("swinv2_tiny_window8_256", pretrained=True, features_only=True, **kwargs),
    "conv_next_v2_tiny": lambda **kwargs: timm.create_model("convnextv2_tiny", pretrained=True, features_only=True, **kwargs)
}


def build_backbone(backbone, **kwargs):
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone](**kwargs)
