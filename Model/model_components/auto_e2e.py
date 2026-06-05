import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .driving_policy import DrivingPolicy
from .future_state import FutureState


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=8, fusion_mode="concat"):
        super(AutoE2E, self).__init__()

        self.num_views = num_views

        # Backbone feature extractor
        self.Backbone = Backbone(backbone)

        # Multi-scale feature fusion with view unification
        self.FeatureFusion = FeatureFusion(num_views=num_views, fusion_mode=fusion_mode)

        # Driving policy prediction
        self.DrivingPolicy = DrivingPolicy()

        # Future visual state prediction
        self.FutureState = FutureState()

    def forward(self, x, visual_history, egomotion_history, camera_params=None):
        B, V, C, H, W = x.shape

        # Merge batch and views for backbone processing
        x = x.reshape(B * V, C, H, W)
        features = self.Backbone(x)

        # Fuse multi-scale features and unify across views
        fused_features = self.FeatureFusion(features, B, V, camera_params=camera_params)

        driving_policy, compressed_visual_feature_vector = \
            self.DrivingPolicy(fused_features, visual_history, egomotion_history)

        future_visual_features = self.FutureState(fused_features)

        return driving_policy, compressed_visual_feature_vector, future_visual_features
