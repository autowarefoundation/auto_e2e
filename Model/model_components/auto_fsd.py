import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .driving_policy import DrivingPolicy


class AutoFSD(nn.Module):
    def __init__(self):
        super(AutoFSD, self).__init__()
        
        # Backbone feature extractor
        self.Backbone = Backbone()

        # Multi-scale feature fusion
        self.FeatureFusion = FeatureFusion()

        # Driving policy head
        self.DrivingPolicy = DrivingPolicy()
   

    def forward(self, image, ego_motion):
        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)
        driving_policy = self.DrivingPolicy(fused_features, ego_motion)
        return driving_policy