import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .trajectory_planner import TrajectoryPlanner
from .future_state import FutureState


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=8, embed_dim=256,
                 fusion_mode="concat", is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896):
        super(AutoE2E, self).__init__()

        # Backbone feature extractor
        self.Backbone = Backbone(backbone=backbone, is_pretrained=is_pretrained)

        # Multi-scale feature fusion with view unification.
        # view_fusion_kwargs forwards bev_h/bev_w/pc_range/image_size to BEV fusion.
        self.FeatureFusion = FeatureFusion(
            num_views=num_views,
            backbone_channels=self.Backbone.backbone_channels,
            embed_dim=embed_dim,
            fusion_mode=fusion_mode,
            image_feature_size=image_feature_size,
            view_fusion_kwargs=view_fusion_kwargs,
        )

        # Trajectory decoder with deformable cross-attention to BEV
        self.TrajectoryPlanner = TrajectoryPlanner(
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            num_signals=num_signals,
            egomotion_dim=egomotion_dim,
            visual_history_dim=visual_history_dim,
        )

        # Future visual state prediction conditioned on planner ego_hidden
        self.FutureState = FutureState(embed_dim=embed_dim, ego_hidden_dim=embed_dim)

    def forward(self, x, visual_history, egomotion_history, camera_params=None, mode="train"):
        B, V, C, H, W = x.shape

        # Merge batch and views for backbone processing
        x = x.reshape(B * V, C, H, W)
        features = self.Backbone(x)

        # Fuse multi-scale features and unify across views
        fused_features = self.FeatureFusion(features, B, V, camera_params=camera_params)

        trajectory, ego_hidden = self.TrajectoryPlanner(
            fused_features, visual_history, egomotion_history
        )

        if mode == "train":
            future_visual_features = self.FutureState(fused_features, ego_hidden)
        else:
            future_visual_features = None

        return trajectory, ego_hidden, future_visual_features
