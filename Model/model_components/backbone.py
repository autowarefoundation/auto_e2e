import torch.nn as nn
from .backbones import build_backbone

class Backbone(nn.Module):
    def __init__(self, backbone="swin_v2_tiny"):
        super().__init__()

        # Pre-trained backbone (pluggable)
        self.backbone = build_backbone(backbone)
         
    def forward(self, image):
        features = self.backbone(image)
        return features   