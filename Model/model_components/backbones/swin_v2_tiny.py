import timm
import torch.nn as nn

class SwinV2Tiny(nn.Module):
    def __init__(self):
        super().__init__()

        # Load Swin V2 Tiny pre-trained on ImageNet-1k without classifier head
        self.backbone = timm.create_model('swinv2_tiny_window16_256', pretrained=True, 
                                          features_only=True)
         
    def forward(self, image):
        features = self.backbone(image)
        return features   