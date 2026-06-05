import timm
import torch.nn as nn

class SwinV2Tiny(nn.Module):
    """SwinV2 Tiny Backbone

    Pre-trained SwinV2 Tiny Backbone with window size of 16 and input image 
    size of 256 for downstream processing and fine-tuning, pre-trained on
    the ImageNet-1k dataset
    """

    def __init__(self):
        super().__init__()

        # Load Swin V2 Tiny pre-trained on ImageNet-1k without classifier head
        self.backbone = timm.create_model('swinv2_tiny_window16_256', pretrained=True, 
                                          features_only=True)
         
    def forward(self, image):
        features = self.backbone(image)
        return features   