import timm
import torch.nn as nn

class ConvNextV2Tiny(nn.Module):
    """ConvNext V2 Tiny Backbone

    Pre-trained ConvNext V2 Tiny Backbone with for downstream processing and fine-tuning
    pre-trained on the ImageNet-22k dataset and then fine-tuned on the ImageNet-1k dataset
    """
        
    def __init__(self):
        super().__init__()

        # Load ConvNextV2 Tiny pre-trained on ImageNet-22k and then fine-tuned
        # on ImageNet-1k without classifier head
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', 
                                          pretrained=True, features_only=True)
         
    def forward(self, image):
        features = self.backbone(image)
        return features   