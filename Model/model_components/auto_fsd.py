import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion


class AutoFSD(nn.Module):
    def __init__(self):
        super(AutoFSD, self).__init__()
        
        # Backbone feature extractor
        self.Backbone = Backbone()

        # Multi-scale feature fusion
        self.FeatureFusion = FeatureFusion()
   

    def forward(self,image):
        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)
        return fused_features