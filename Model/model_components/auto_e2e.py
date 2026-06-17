import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .trajectory_planning import build_planner
from .future_state import FutureState
from .map_encoder import build_map_encoder, build_map_bev_fusion


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 fusion_mode="bev", is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_in_channels=3,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 planner_mode="gru", planner_kwargs=None):
        super(AutoE2E, self).__init__()

        # Camera backbone feature extractor
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

        # For BEV fusion mode the spatial size is bev_h × bev_w (potentially non-square).
        # For concat/cross_attn it is image_feature_size × image_feature_size.
        if fusion_mode == "bev":
            vfk = view_fusion_kwargs or {"bev_h": 450, "bev_w": 300}
            map_output_h = vfk["bev_h"]
            map_output_w = vfk["bev_w"]
        else:
            map_output_h = image_feature_size
            map_output_w = image_feature_size
 
        # Map encoder: encodes the BEV nav-map image into spatial map features
        self.MapEncoder = build_map_encoder(
            map_type,
            in_channels=map_in_channels,
            embed_dim=embed_dim,
            output_h=map_output_h,
            output_w=map_output_w,
        )
 
        # Map BEV fusion: combines image BEV features with map BEV features
        self.MapBEVFusion = build_map_bev_fusion(
            map_fusion_mode,
            embed_dim=embed_dim,
            **(map_fusion_kwargs or {}),
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

    def forward(self, camera_tiles, map_input, visual_history, egomotion_history,
                camera_params=None, mode="train", trajectory_target=None, **kwargs):
        """
        Run the full autonomous-driving pipeline.

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

        Args:
            camera_tiles: (B, V, 3, H, W) — V camera images (V=7 by default).
            map_input: (B, 3, H_map, W_map) — BEV nav-map image.
            visual_history: (B, visual_history_dim).
            egomotion_history: (B, egomotion_dim).
            camera_params: Optional (B, V, 3, 4) ego-to-pixel projection matrices.
            mode: "train" to produce future_visual_features; anything else skips it.

        Returns:
            trajectory: (B, num_timesteps * num_signals)
            ego_hidden: (B, embed_dim)
            future_visual_features: list of 4 × (B, embed_dim, H, W), or None
        """
        B, V, C, H, W = camera_tiles.shape

        # --- Camera branch ---
        x = camera_tiles.reshape(B * V, C, H, W)
        features = self.Backbone(x)
        image_bev = self.FeatureFusion(features, B, V, camera_params=camera_params)

        # --- Map branch ---
        map_bev = self.MapEncoder(map_input)

        # --- Fuse image BEV + map BEV ---
        fused_features = self.MapBEVFusion(image_bev, map_bev)

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
