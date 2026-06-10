import torch.nn as nn
from .backbones import build_backbone

class Backbone(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", is_pretrained: bool = True):
        super().__init__()

        # Pre-trained backbone (pluggable)
        self.backbone = build_backbone(backbone, pretrained=is_pretrained)
        self.backbone_name = backbone

        if not hasattr(self.backbone, "feature_info"):
            raise ValueError(
                f"Backbone '{backbone}' does not expose feature_info; "
                "cannot infer backbone_channels dynamically."
            )
        self.backbone_channels = sum(
            info["num_chs"] for info in self.backbone.feature_info
        )

    def forward(self, image):
        features = self.backbone(image)
        for i in range(0, len(features)):
            if "swin" in self.backbone_name:
                features[i] = features[i].permute(0, 3, 1, 2)

        return features
