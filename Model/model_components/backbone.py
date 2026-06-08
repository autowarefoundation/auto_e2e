import torch.nn as nn
from .backbones import build_backbone

class Backbone(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", is_pretrained: bool = True):
        super().__init__()

        # Pre-trained backbone (pluggable)
        self.backbone = build_backbone(backbone, pretrained=is_pretrained)
        self.backbone_name = backbone
        self.backbone_channels = 0


        if(backbone=="swin_v2_tiny" or backbone =="conv_next_v2_tiny"):
            self.backbone_channels = 1440
        
        if(backbone=="res_net_50"):
            self.backbone_channels = 3904
         
    def forward(self, image):
        features = self.backbone(image)
        for i in range(0, len(features)):
            if(self.backbone_name == "swin_v2_tiny"):
                features[i] = features[i].permute(0, 3, 1, 2)

        return features