import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .driving_policy import DrivingPolicy
from .future_state import FutureState


class AutoFSD(nn.Module):
    def __init__(self):
        super(AutoFSD, self).__init__()
        
        # Backbone feature extractor
        self.Backbone = Backbone()

        # Multi-scale feature fusion
        self.FeatureFusion = FeatureFusion()

        # Driving policy prediction
        self.DrivingPolicy = DrivingPolicy()

        # Future visual state prediction
        self.FutureState = FutureState()
   

    def forward(self, image, visual_history, egomotion_history):

        features = self.Backbone(image)
        fused_features = self.FeatureFusion(features)

        driving_policy, compressed_visual_feature_vector = \
            self.DrivingPolicy(fused_features, visual_history, egomotion_history)
        
        future_visual_features = self.FutureState(fused_features)
        
        return driving_policy, compressed_visual_feature_vector, future_visual_features