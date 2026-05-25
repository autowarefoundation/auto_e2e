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
   

    def forward(self, image, visual_history, egomotion_history):
        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)

        driving_policy, compressed_visual_feature_vector = \
            self.DrivingPolicy(fused_features, visual_history, egomotion_history)
        
        return driving_policy, compressed_visual_feature_vector