import timm

BACKBONE_REGISTRY = {
    "swin_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "swinv2_tiny_window8_256", pretrained=pretrained, features_only=True, **kwargs
    ),
    "conv_next_v2_tiny": lambda pretrained=True, **kwargs: timm.create_model(
        "convnextv2_tiny", pretrained=pretrained, features_only=True, **kwargs
    ),
    "res_net_50": lambda pretrained=True, **kwargs: timm.create_model(
        "resnet50", pretrained=pretrained, features_only=True, **kwargs
    ),
}


def build_backbone(backbone, pretrained=True, **kwargs):
    """Construct a multi-scale feature backbone by registry name.

    Args:
        backbone: Registry key (see ``BACKBONE_REGISTRY`` for available names).
        pretrained: Whether to load pretrained weights.
        **kwargs: Forwarded to the underlying constructor.

    Channel discovery convention:
        The returned module is consumed by :class:`Backbone`, which discovers
        per-stage channel counts to wire up downstream fusion. The preferred
        path is for the backbone to expose a ``feature_info`` attribute (a
        sequence whose entries each provide a ``num_chs`` key — timm's standard
        contract for ``features_only=True``). When ``feature_info`` is absent,
        ``Backbone`` falls back to a one-shot dummy forward pass that sums the
        channel dimension of every returned feature map (assumed channels-first).
        Either path is supported, so custom backbones without timm metadata
        still work; ``feature_info`` is preferred only because it avoids the
        probe forward.
    """
    if backbone not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone](pretrained=pretrained, **kwargs)
