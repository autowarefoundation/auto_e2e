import timm
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Load SwinV2 Tiny pre-trained on ImageNet-22k wihtout classifier head
        self.backbone = timm.create_model('swinv2_tiny_patch4_window8_256.ms_in22k', 
                                          pretrained=True, num_classes=0)
 
    def forward(self, image):
        features = self.backbone(image)
        return features   