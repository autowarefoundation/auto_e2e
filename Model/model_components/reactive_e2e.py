import torch
import torch.nn as nn
from .auxiliary_heads import (
    BEVSegmentationHead,
    RouteReconstructionHead,
)
from .backbone import Backbone
from .feature_fusion import FeatureFusion
from .fused_feature_pooling import FusedFeaturePooling
from .trajectory_planning import build_planner
from .map_encoder import (
    build_map_bev_fusion,
    build_split_navigation_encoder,
)
from .temporal_memory import build_temporal_memory
from .reasoning.horizon_reasoning_head import HorizonReasoningHead


class ReactiveE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, 
                 egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_context_channels=3,
                 route_channels=2, enable_route_conditioning=True,
                 route_encoder_hidden_channels=96,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 temporal_memory_mode="no_memory", temporal_memory_kwargs=None,
                 planner_mode="gru", planner_kwargs=None,
                 enable_reasoning=False, reasoning_mode="none",
                 reasoning_kwargs=None,
                 enable_bev_segmentation=False,
                 bev_segmentation_classes=8,
                 enable_route_reconstruction=False,
                 auxiliary_output_size=None):
        super(ReactiveE2E, self).__init__()

        # Camera backbone feature extractor
        camera_architecture = str(
            (view_fusion_kwargs or {}).get("architecture", "legacy")
        )
        self.camera_input_size: int | None = None
        self.front_camera_index: int | None = None
        self.front_camera_input_size: int | None = None
        self.camera_architecture = camera_architecture
        if camera_architecture in {"bevformer_v2_t1", "bevformer_v2_t8"}:
            camera_input_size = (view_fusion_kwargs or {}).get(
                "image_size",
                256,
            )
            if (
                not isinstance(camera_input_size, int)
                or isinstance(camera_input_size, bool)
                or camera_input_size <= 0
            ):
                raise ValueError(
                    "BEVFormer camera image_size must be a positive integer"
                )
            self.camera_input_size = camera_input_size
            front_camera_index = (view_fusion_kwargs or {}).get(
                "front_camera_index",
                0,
            )
            front_camera_input_size = (view_fusion_kwargs or {}).get(
                "front_image_size",
                1024,
            )
            if (
                not isinstance(front_camera_index, int)
                or isinstance(front_camera_index, bool)
                or not 0 <= front_camera_index < num_views
                or not isinstance(front_camera_input_size, int)
                or isinstance(front_camera_input_size, bool)
                or front_camera_input_size < camera_input_size
                or front_camera_input_size % camera_input_size
            ):
                raise ValueError(
                    "BEVFormer front camera contract is invalid"
                )
            self.front_camera_index = front_camera_index
            self.front_camera_input_size = front_camera_input_size
        self.Backbone = Backbone(
            backbone=backbone,
            is_pretrained=is_pretrained,
            input_profile=(
                "bevformer_v2"
                if camera_architecture in {
                    "bevformer_v2_t1",
                    "bevformer_v2_t8",
                }
                else "imagenet"
            ),
        )

        # Multi-scale feature fusion with view unification.
        # view_fusion_kwargs forwards bev_h/bev_w/pc_range/image_size to BEV fusion.
        self.FeatureFusion = FeatureFusion(
            num_views=num_views,
            backbone_channels=self.Backbone.feature_channels,
            embed_dim=embed_dim,
            fusion_mode="bev",
            image_feature_size=image_feature_size,
            view_fusion_kwargs=view_fusion_kwargs,
        )
        self._camera_bev_frozen = False
        self._adapt_temporal_running_stats = False

        self.planner_mode = planner_mode
        self.FusedFeaturePooling = (
            FusedFeaturePooling(embed_dim=embed_dim)
            if planner_mode == "bezier"
            else None
        )

        # For BEV fusion mode the spatial size is bev_h × bev_w (potentially non-square).
        # Read each dim with a default so a PARTIAL view_fusion_kwargs (e.g. only
        # pc_range) doesn't KeyError — the `or` only fires for None/empty, and the
        # defaults must match BEVViewFusion's own (450×300).
        vfk = view_fusion_kwargs or {}
        map_output_h = vfk.get("bev_h", 450)
        map_output_w = vfk.get("bev_w", 300)

 
        if map_context_channels <= 0 or route_channels <= 0:
            raise ValueError("navigation channel counts must be positive")
        self.map_context_channels = int(map_context_channels)
        self.route_channels = int(route_channels)
        self.enable_route_conditioning = bool(enable_route_conditioning)

        self.NavigationEncoder = build_split_navigation_encoder(
            map_type,
            map_channels=self.map_context_channels,
            route_channels=self.route_channels,
            embed_dim=embed_dim,
            output_h=map_output_h,
            output_w=map_output_w,
            route_hidden_channels=route_encoder_hidden_channels,
        )
 
        # Map BEV fusion: combines image BEV features with map BEV features
        self.MapBEVFusion = build_map_bev_fusion(
            map_fusion_mode,
            embed_dim=embed_dim,
            **(map_fusion_kwargs or {}),
        )
        self.BEVSegmentationHead = (
            BEVSegmentationHead(
                embed_dim=embed_dim,
                num_classes=bev_segmentation_classes,
                output_size=auxiliary_output_size,
            )
            if enable_bev_segmentation
            else None
        )
        self.RouteReconstructionHead = (
            RouteReconstructionHead(
                embed_dim=embed_dim,
                route_channels=route_channels,
                output_size=auxiliary_output_size,
            )
            if enable_route_reconstruction
            else None
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
            if rkw.get("route_context_dim") is not None:
                raise ValueError(
                    "route_context_dim is not supported by the initial #149 "
                    "contract; route conditioning is Reactive-only"
                )
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

    def freeze_camera_bev(
        self,
        *,
        adapt_temporal_running_stats: bool = False,
    ) -> None:
        """Freeze checkpoint weights while retaining the new front gate."""
        self._camera_bev_frozen = True
        self._adapt_temporal_running_stats = bool(
            adapt_temporal_running_stats
        )
        for module in (self.Backbone, self.FeatureFusion):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.camera_architecture in {
            "bevformer_v2_t1",
            "bevformer_v2_t8",
        }:
            front_gate = getattr(
                self.FeatureFusion.view_fusion,
                "front_residual_gate",
                None,
            )
            if not isinstance(front_gate, nn.Parameter):
                raise ValueError("BEVFormer front residual gate is missing")
            front_gate.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._camera_bev_frozen:
            self.Backbone.eval()
            self.FeatureFusion.eval()
            temporal_fusion = getattr(
                self.FeatureFusion,
                "temporal_fusion",
                None,
            )
            if temporal_fusion is not None:
                temporal_fusion.train(
                    mode and self._adapt_temporal_running_stats
                )
        return self

    def encode_camera_bev(
        self,
        camera_tiles,
        *,
        projection=None,
        geometry_type=None,
        image_transform=None,
        front_camera_tile=None,
        front_projection=None,
        front_image_transform=None,
    ):
        """Encode camera tiles without reading navigation inputs."""
        if camera_tiles.ndim != 5:
            raise ValueError(
                "camera_tiles must have shape [B,V,3,H,W]"
            )
        batch_size, num_views, channels, height, width = camera_tiles.shape
        if channels != 3:
            raise ValueError("camera_tiles must contain three RGB channels")
        if (
            self.camera_input_size is not None
            and (height, width)
            != (self.camera_input_size, self.camera_input_size)
        ):
            raise ValueError(
                "camera tile size differs from BEVFormer image_size contract"
            )
        if front_camera_tile is None and self.camera_input_size is None:
            features = self.Backbone(
                camera_tiles.reshape(
                    batch_size * num_views,
                    channels,
                    height,
                    width,
                )
            )
            return self.FeatureFusion(
                features,
                batch_size,
                num_views,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
        if front_camera_tile is None:
            if (
                front_projection is not None
                or front_image_transform is not None
            ):
                raise ValueError(
                    "front camera geometry requires a native front camera"
                )
            features = self.Backbone(camera_tiles.reshape(
                batch_size * num_views,
                channels,
                height,
                width,
            ))
            return self.FeatureFusion(
                features,
                batch_size,
                num_views,
                projection=projection,
                geometry_type=geometry_type,
                image_transform=image_transform,
            )
        if (
            self.front_camera_index is None
            or self.front_camera_input_size is None
        ):
            raise ValueError(
                "front camera input requires BEVFormer V2 fusion"
            )
        if tuple(front_camera_tile.shape) != (
            batch_size,
            3,
            self.front_camera_input_size,
            self.front_camera_input_size,
        ):
            raise ValueError(
                "front camera tile differs from BEVFormer contract"
            )
        # Every T8 frame uses the same ordinary all-view encoder. The native
        # front pass is additive and never replaces CAM_F0 in that branch.
        features = self.Backbone(camera_tiles.reshape(
            batch_size * num_views,
            channels,
            height,
            width,
        ))
        front_features = self.Backbone(front_camera_tile)
        return self.FeatureFusion(
            features,
            batch_size,
            num_views,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
            front_features=front_features,
            front_projection=front_projection,
            front_image_transform=front_image_transform,
        )

    def _encode_history_camera_bevs(
        self,
        camera_history_tiles,
        *,
        history_projections=None,
        geometry_type=None,
        image_transform=None,
    ):
        if self.camera_architecture != "bevformer_v2_t8":
            if camera_history_tiles is not None:
                raise ValueError(
                    "camera history requires BEVFormer V2 T8"
                )
            return []
        if camera_history_tiles is None:
            if history_projections is not None:
                raise ValueError(
                    "history projections require camera history tiles"
                )
            return []
        if camera_history_tiles.ndim != 6:
            raise ValueError(
                "camera_history_tiles must have shape [B,7,V,3,H,W]"
            )
        batch_size, history_count, num_views, channels, height, width = (
            camera_history_tiles.shape
        )
        if (
            history_count != 7
            or channels != 3
            or self.camera_input_size is None
            or (height, width) != (
                self.camera_input_size,
                self.camera_input_size,
            )
        ):
            raise ValueError("camera history differs from the T8 contract")
        if history_projections is not None and len(
            history_projections
        ) != history_count:
            raise ValueError("T8 requires one projection per history frame")
        backbone_training = self.Backbone.training
        fusion_training = self.FeatureFusion.training
        temporal_fusion = getattr(
            self.FeatureFusion,
            "temporal_fusion",
            None,
        )
        temporal_training = (
            temporal_fusion.training
            if temporal_fusion is not None
            else None
        )
        history_bevs = []
        self.Backbone.eval()
        self.FeatureFusion.eval()
        try:
            with torch.no_grad():
                for history_index in range(history_count):
                    tiles = camera_history_tiles[:, history_index]
                    features = self.Backbone(tiles.reshape(
                        batch_size * num_views,
                        channels,
                        height,
                        width,
                    ))
                    projection = (
                        history_projections[history_index]
                        if history_projections is not None
                        else None
                    )
                    history_bevs.append(self.FeatureFusion(
                        features,
                        batch_size,
                        num_views,
                        projection=projection,
                        geometry_type=geometry_type,
                        image_transform=image_transform,
                    ).detach())
        finally:
            self.Backbone.train(backbone_training)
            self.FeatureFusion.train(fusion_training)
            if temporal_fusion is not None:
                temporal_fusion.train(bool(temporal_training))
        return history_bevs

    def _fuse_temporal_camera_bevs(
        self,
        current_image_bev,
        history_bevs,
    ):
        if self.camera_architecture != "bevformer_v2_t8":
            if history_bevs:
                raise ValueError("non-T8 camera path received history BEVs")
            return current_image_bev
        if not history_bevs:
            history_bevs = [
                current_image_bev.detach()
                for _ in range(7)
            ]
        # Upstream BEVFormer@66b65f3 transformerV2.py:308-324 places the
        # current frame last in the ordered (-7,-6,-5,-4,-3,-2,-1,0) list.
        return self.FeatureFusion.fuse_temporal_bevs([
            *history_bevs,
            current_image_bev,
        ])

    def forward(self, camera_tiles, map_context, visual_history,
                egomotion_history, route_mask=None, map_valid=None,
                route_valid=None,
                projection=None, geometry_type=None, image_transform=None,
                camera_history_tiles=None, history_projections=None,
                front_camera_tile=None, front_projection=None,
                front_image_transform=None,
                mode="train", return_auxiliary=False,
                compute_bev_segmentation=True,
                compute_route_reconstruction=True,
                **kwargs):
        """
        Run the reactive end-to-end autonomous-driving pipeline.


        Args:
            camera_tiles: (B, V, 3, H, W) — V real camera images.
            map_context: (B, C_map, H_map, W_map) — semantic BEV map.
            route_mask: (B, C_route, H_map, W_map) — selected route.
            map_valid / route_valid: (B,) explicit sample validity.
                ``enable_route_conditioning=False`` forces the route gate off.
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            projection: Optional CameraProjectionModel operator — the geometry
                ABI (Pinhole / FTheta / Pseudo). No [B,V,3,4] matrix argument.
            geometry_type: Optional explicit geometry label passed to BEV fusion.
            image_transform: Optional ImageTransform for the model-input frame.
            front_camera_tile: Optional native-resolution front image tensor.
            front_projection: Optional one-view projection for the native front.
            camera_history_tiles: Optional seven-frame camera history ordered
                oldest to newest, shaped ``[B,7,V,3,H,W]``.
            history_projections: Optional seven projection operators mapping
                current ego coordinates into each historical image set.
            mode: "train" returns enabled auxiliary predictions.
            return_auxiliary: also return enabled auxiliary predictions during
                inference, for offline Dashboard artifact generation.

        Returns:
            trajectory (B, num_timesteps * num_signals), or
            ``(trajectory, aux_outputs)`` when auxiliary outputs were requested.
        """
        B = camera_tiles.shape[0]

        # --- Camera branch ---
        history_bevs = self._encode_history_camera_bevs(
            camera_history_tiles,
            history_projections=history_projections,
            geometry_type=geometry_type,
            image_transform=image_transform,
        )
        current_image_bev = self.encode_camera_bev(
            camera_tiles,
            projection=projection,
            geometry_type=geometry_type,
            image_transform=image_transform,
            front_camera_tile=front_camera_tile,
            front_projection=front_projection,
            front_image_transform=front_image_transform,
        )
        image_bev = self._fuse_temporal_camera_bevs(
            current_image_bev,
            history_bevs,
        )
        emit_auxiliary = mode == "train" or bool(return_auxiliary)
        aux_outputs = {}
        if (
            self.BEVSegmentationHead is not None
            and emit_auxiliary
            and compute_bev_segmentation
        ):
            aux_outputs["bev_segmentation_logits"] = (
                self.BEVSegmentationHead(image_bev)
            )

        # --- Reactive-only navigation branch ---
        if (
            map_context.ndim != 4
            or map_context.shape[0] != B
            or map_context.shape[1] != self.map_context_channels
        ):
            raise ValueError(
                "map_context must have shape "
                f"[B,{self.map_context_channels},H,W]"
            )
        if route_mask is None:
            route_mask = map_context.new_zeros(
                B,
                self.route_channels,
                map_context.shape[-2],
                map_context.shape[-1],
            )
        if (
            route_mask.ndim != 4
            or route_mask.shape[:2] != (B, self.route_channels)
            or route_mask.shape[-2:] != map_context.shape[-2:]
        ):
            raise ValueError(
                "route_mask must share map spatial dimensions and have "
                f"{self.route_channels} channels"
            )

        def validity_gate(value, valid, *, default, name):
            if valid is None:
                valid = torch.full(
                    (B,),
                    default,
                    dtype=torch.bool,
                    device=value.device,
                )
            valid = torch.as_tensor(valid, device=value.device)
            if valid.shape not in ((B,), (B, 1)):
                raise ValueError(f"{name} must have shape [B] or [B,1]")
            return value * valid.reshape(B, 1, 1, 1).to(value.dtype)

        gated_map = validity_gate(
            map_context,
            map_valid,
            default=True,
            name="map_valid",
        )
        if not self.enable_route_conditioning:
            route_valid = torch.zeros(
                B,
                dtype=torch.bool,
                device=route_mask.device,
            )
        gated_route = validity_gate(
            route_mask,
            route_valid,
            default=False,
            name="route_valid",
        )
        navigation_bev, route_contribution = self.NavigationEncoder(
            gated_map,
            gated_route,
            return_route_contribution=True,
        )

        # --- Fuse image BEV + navigation BEV ---
        fused_features = self.MapBEVFusion(image_bev, navigation_bev)
        if (
            self.RouteReconstructionHead is not None
            and emit_auxiliary
            and compute_route_reconstruction
        ):
            aux_outputs["route_reconstruction_logits"] = (
                self.RouteReconstructionHead(route_contribution)
            )

        planner_features = (
            self.FusedFeaturePooling(fused_features)
            if self.FusedFeaturePooling is not None
            else fused_features
        )

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
            )
            reasoning_latent = reasoning_pred.reasoning_latent
            reasoning_horizon_tokens = reasoning_pred.horizon_tokens

        # --- Trajectory Prediction ---
        trajectory = self.TrajectoryPlanner(
            planner_features, visual_ctx, ego_ctx,
            reasoning_latent=reasoning_latent,
            reasoning_horizon_tokens=reasoning_horizon_tokens,
            **kwargs,
        )

        if reasoning_pred is not None and mode == "train":
            aux_outputs["reasoning_pred"] = reasoning_pred
        if aux_outputs:
            return trajectory, aux_outputs
        return trajectory
