import torch.nn as nn
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .trajectory_planning import build_planner
from .map_encoder import build_map_encoder, build_map_bev_fusion
from .temporal_memory import build_temporal_memory
from .reasoning.horizon_reasoning_head import HorizonReasoningHead


class ReactiveE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_in_channels=3,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 temporal_memory_mode="no_memory", temporal_memory_kwargs=None,
                 planner_mode="gru", planner_kwargs=None,
                 enable_reasoning=False, reasoning_mode="none",
                 reasoning_kwargs=None):
        super(ReactiveE2E, self).__init__()

        # Camera backbone feature extractor
        self.Backbone = Backbone(backbone=backbone, is_pretrained=is_pretrained)

        # Multi-scale feature fusion with view unification.
        # view_fusion_kwargs forwards bev_h/bev_w/pc_range/image_size to BEV fusion.
        self.FeatureFusion = FeatureFusion(
            num_views=num_views,
            backbone_channels=self.Backbone.backbone_channels,
            embed_dim=embed_dim,
            fusion_mode="bev",
            image_feature_size=image_feature_size,
            view_fusion_kwargs=view_fusion_kwargs,
        )

        # For BEV fusion mode the spatial size is bev_h × bev_w (potentially non-square).
        # Read each dim with a default so a PARTIAL view_fusion_kwargs (e.g. only
        # pc_range) doesn't KeyError — the `or` only fires for None/empty, and the
        # defaults must match BEVViewFusion's own (450×300).
        vfk = view_fusion_kwargs or {}
        map_output_h = vfk.get("bev_h", 450)
        map_output_w = vfk.get("bev_w", 300)

 
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

        # Temporal Memory — compresses/fuses [B, T, feat] sequence histories into contexts
        self.TemporalMemory = build_temporal_memory(
            temporal_memory_mode,
            visual_dim=visual_history_dim,
            egomotion_dim=egomotion_dim,
            **(temporal_memory_kwargs or {}),
        )

        # Reasoning branch (1 Hz, opt-in, default OFF): horizon-aware,
        # action-relevant reasoning over the effective visual history + ego
        # context produced by TemporalMemory. Runs AFTER TemporalMemory so
        # ego_ctx is available (see Design/horizon_reasoning_architecture.md).
        # Feeds the planner through a ZERO-INIT coupling (reasoning_mode), a
        # strict no-op at init so the reactive baseline is byte-identical.
        self.enable_reasoning = enable_reasoning
        self.reasoning_mode = reasoning_mode if enable_reasoning else "none"
        self.ReasoningHead = None
        if enable_reasoning:
            rkw = dict(reasoning_kwargs or {})
            rkw.setdefault("visual_history_dim", visual_history_dim)
            rkw.setdefault("ego_context_dim", egomotion_dim)
            self.ReasoningHead = HorizonReasoningHead(**rkw)

        # Trajectory decoder — swappable via planner_mode (gru, flow_matching).
        # reasoning_mode wires the zero-init reasoning coupling inside the planner.
        self.TrajectoryPlanner = build_planner(
            planner_mode,
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            num_signals=num_signals,
            egomotion_dim=egomotion_dim,
            visual_history_dim=visual_history_dim,
            reasoning_mode=self.reasoning_mode,
            **(planner_kwargs or {}),
        )

        # NOTE: future visual-state prediction now lives in the World Model
        # branch (WorldActionModel.predict_future, JEPA). The old ReactiveE2E-owned
        # FutureState module was instantiated here but NEVER called in forward — a
        # gradient-dead parameter block — so it is removed. See auto_e2e.py.

    def forward(self, camera_tiles, map_input, visual_history, egomotion_history,
                projection=None, geometry_type=None, image_transform=None,
                route_context=None, map_context=None, mode="train", **kwargs):
        """
        Run the reactive end-to-end autonomous-driving pipeline.


        Args:
            camera_tiles: (B, V, 3, H, W) — V real camera images.
            map_input: (B, 3, H_map, W_map) — BEV nav-map image.
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            projection: Optional CameraProjectionModel operator — the geometry
                ABI (Pinhole / FTheta / Pseudo). No [B,V,3,4] matrix argument.
            geometry_type: Optional explicit geometry label passed to BEV fusion.
            image_transform: Optional ImageTransform for the model-input frame.
            route_context / map_context: optional extra reasoning context.
            mode: "train" also returns the reasoning prediction (for its loss).

        Returns:
            trajectory (B, num_timesteps * num_signals), OR — when the reasoning
            branch is enabled and ``mode == "train"`` — a tuple
            ``(trajectory, reasoning_pred)`` so the training loop can compute the
            reasoning loss. ``reasoning_pred`` is a HorizonReasoningPrediction.
        """
        B, V, C, H, W = camera_tiles.shape

        # --- Camera branch ---
        x = camera_tiles.reshape(B * V, C, H, W)
        features = self.Backbone(x)
        image_bev = self.FeatureFusion(
            features, B, V,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
        )

        # --- Map branch ---
        map_bev = self.MapEncoder(map_input)

        # --- Fuse image BEV + map BEV ---
        fused_features = self.MapBEVFusion(image_bev, map_bev)

        # --- Temporal Memory ---
        visual_ctx, ego_ctx = self.TemporalMemory(visual_history, egomotion_history)

        # --- Reasoning branch (1 Hz, opt-in) ---
        # Runs on the EFFECTIVE context TemporalMemory produced, so the reasoning
        # head and the planner see the same visual/ego signal. Its latent /
        # horizon tokens feed the planner through the zero-init coupling.
        reasoning_pred = None
        reasoning_latent = None
        reasoning_horizon_tokens = None
        if self.ReasoningHead is not None:
            reasoning_pred = self.ReasoningHead(
                visual_ctx, ego_ctx,
                route_context=route_context, map_context=map_context,
            )
            reasoning_latent = reasoning_pred.reasoning_latent
            reasoning_horizon_tokens = reasoning_pred.horizon_tokens

        # --- Trajectory Prediction ---
        trajectory = self.TrajectoryPlanner(
            fused_features, visual_ctx, ego_ctx,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )

        if self.ReasoningHead is not None and mode == "train":
            return trajectory, reasoning_pred
        return trajectory
