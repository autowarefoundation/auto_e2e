import torch.nn as nn
from .backbones import build_backbone

class Backbone(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", is_pretrained: bool = True):
        super().__init__()

        # Pre-trained backbone (pluggable)
        self.backbone = build_backbone(backbone, is_pretrained=is_pretrained)
        self.backbone_name = backbone
         
    def forward(self, image):
        features = self.backbone(image)
        backbone_channels = 0
        for i in range(0, len(features)):
            if(self.backbone_name == "swin_v2_tiny"):
                features[i] = features[i].permute(0, 3, 1, 2)

            _, C, _, _ = features[i].shape
            backbone_channels += C

        return features, backbone_channels