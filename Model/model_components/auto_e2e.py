import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .trajectory_planning import build_planner
from .future_state import FutureState


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=8, embed_dim=256,
                 fusion_mode="concat", is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896, planner_mode="gru",
                 planner_kwargs=None):
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

        # Trajectory decoder — swappable via planner_mode (gru, flow_matching).
        self.TrajectoryPlanner = build_planner(
            planner_mode,
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            num_signals=num_signals,
            egomotion_dim=egomotion_dim,
            visual_history_dim=visual_history_dim,
            **(planner_kwargs or {}),
        )

        # Future visual state prediction conditioned on planner ego_hidden
        self.FutureState = FutureState(embed_dim=embed_dim, ego_hidden_dim=embed_dim)

    def forward(self, x, visual_history, egomotion_history, camera_params=None,
                mode="train", trajectory_target=None, **kwargs):
        """Run the full autonomous-driving pipeline.

        The first return value's meaning depends on ``mode`` but is uniform
        across all planners (GRU, Flow Matching, ...):

        * ``mode="train"``: returns ``(planner_loss, ego_hidden, future)``
          where ``planner_loss`` is a SCALAR — not a trajectory. The
          planner-specific objective (imitation MSE for GRU,
          flow-matching velocity MSE for Flow Matching) is computed
          inside the planner so a training loop never has to know which
          decoder is active. ``trajectory_target`` is required.
        * any other ``mode`` (e.g. ``"infer"``): returns
          ``(trajectory, ego_hidden, None)`` where ``trajectory`` is
          ``[B, num_timesteps * num_signals]``.
        """
        B, V, C, H, W = x.shape

        # Merge batch and views for backbone processing
        x = x.reshape(B * V, C, H, W)
        features = self.Backbone(x)

        # Fuse multi-scale features and unify across views
        fused_features = self.FeatureFusion(features, B, V, camera_params=camera_params)

        if mode == "train":
            if trajectory_target is None:
                raise ValueError(
                    "AutoE2E.forward(mode='train') requires trajectory_target."
                )
            planner_loss, ego_hidden = self.TrajectoryPlanner.compute_planner_loss(
                fused_features, visual_history, egomotion_history,
                trajectory_target,
            )
            future_visual_features = self.FutureState(fused_features, ego_hidden)
            return planner_loss, ego_hidden, future_visual_features

        trajectory, ego_hidden = self.TrajectoryPlanner(
            fused_features, visual_history, egomotion_history, **kwargs,
        )
        return trajectory, ego_hidden, None
