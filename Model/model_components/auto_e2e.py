import torch.nn as nn
from .reactive_e2e import ReactiveE2E
from .world_action_model import RollingHistoryBuffer, WorldActionModel


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_in_channels=3,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 temporal_memory_mode="no_memory", temporal_memory_kwargs=None,
                 planner_mode="bezier", planner_kwargs=None,
                 enable_world_model=False, world_model_kwargs=None):
        super(AutoE2E, self).__init__()

        # Reactive model which runs at 10Hz and processes multi-camera inputs
        # a rendered map image and egomotion history to predict a driving trajectory
        # to reach the near-horizon navigational goal
        self.Reactive_E2E = ReactiveE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim,
                 is_pretrained=is_pretrained,
                 image_feature_size=image_feature_size, view_fusion_kwargs=view_fusion_kwargs,
                 num_timesteps=num_timesteps, num_signals=num_signals, egomotion_dim=egomotion_dim,
                 visual_history_dim=visual_history_dim,
                 map_type=map_type, map_in_channels=map_in_channels,
                 map_fusion_mode=map_fusion_mode, map_fusion_kwargs=map_fusion_kwargs,
                 temporal_memory_mode=temporal_memory_mode, temporal_memory_kwargs=temporal_memory_kwargs,
                 planner_mode=planner_mode, planner_kwargs=planner_kwargs)

        # World Action Model (slow, ~1Hz): encodes the multi-camera history into
        # the Encoded Visual History (fed to the reactive planner) and predicts
        # future visual features (JEPA). Reuses the reactive backbone (one shared
        # backbone; the JEPA target is a frozen copy of it). Opt-in (default OFF)
        # so the reactive-only default is byte-identical.
        self.World_Action_Model_E2E = None
        self.visual_history_buffer = None
        if enable_world_model:
            wmk = dict(world_model_kwargs or {})
            history_len = wmk.pop("history_len", 4)
            wmk.setdefault("view_aggregator", "attention")
            self.World_Action_Model_E2E = WorldActionModel(
                backbone=self.Reactive_E2E.Backbone,
                frame_embed_dim=visual_history_dim // history_len,
                history_len=history_len, num_views=num_views, **wmk,
            )
            self.visual_history_buffer = RollingHistoryBuffer(history_len=history_len)

    def reset_visual_history(self):
        """Clear the World Model's rolling buffer (call between sequences)."""
        if self.visual_history_buffer is not None:
            self.visual_history_buffer = RollingHistoryBuffer(
                history_len=self.visual_history_buffer.history_len)


    def forward(self, camera_tiles, map_input, visual_history, egomotion_history,
                projection=None, geometry_type=None, image_transform=None,
                mode="train", trajectory_target=None, **kwargs):
        """
        Run the full autonomous-driving pipeline.

        Returns a single trajectory tensor ``[B, num_timesteps * num_signals]``
        (the pre-#94 3-tuple return was removed when the planner interface was
        simplified). ``mode`` and ``trajectory_target`` are threaded through for
        forward-compatibility with a future train-time planner objective but are
        currently inert in the default planner.

        Args:
            camera_tiles: (B, V, 3, H, W) — V real camera images (the nav-map is
                a separate map_input, not a camera view).
            map_input: (B, 3, H_map, W_map) — BEV nav-map image.
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            projection: Optional CameraProjectionModel operator — the geometry
                ABI (Pinhole / FTheta / Pseudo). No [B,V,3,4] matrix argument;
                construct PinholeProjection(matrix) if you have a pinhole matrix.
            geometry_type: Optional explicit geometry label ("pinhole",
                "rectified_pinhole", "ftheta", "pseudo") passed to BEV fusion.
            image_transform: Optional ImageTransform for the model-input frame.
            mode: threaded through to the planner (currently inert by default).

        Returns:
            trajectory: (B, num_timesteps * num_signals)
        """

        # World Action Model (1Hz): encode the current multi-camera frame into the
        # rolling Encoded Visual History fed to the reactive planner, and (in
        # training) predict the future feature state for the JEPA loss.
        #
        # IMPORTANT — the rolling buffer here is **inference / rollout memory only**:
        #   * pushed embeddings are DETACHED, so the buffer never carries an
        #     autograd graph across forward steps (no cross-step BPTT / graph leak),
        #   * it holds per-sequence state — call ``reset_visual_history()`` between
        #     independent sequences (it is NOT safe to share across shuffled batches).
        # Batched JEPA TRAINING does NOT use this buffer: it goes through the
        # stateless, windowed and fully-differentiable path on ``WorldActionModel``
        # (``encode_history`` -> ``aggregate_history`` -> ``predict_future`` ->
        # ``jepa_loss``) driven from ``train_il`` (see #13).
        future_state_pred = None
        if self.World_Action_Model_E2E is not None:
            wam = self.World_Action_Model_E2E
            # Encode the current 1 Hz multi-view frame; push a detached copy so the
            # planner's history is a pure rolling memory (no graph across steps).
            visual_embedding, _ = wam(camera_tiles)
            self.visual_history_buffer.push(visual_embedding.detach())
            # The planner and the future prediction read the SAME history (post-push,
            # i.e. including the current frame) so the JEPA context stays aligned
            # with what the planner actually sees.
            visual_history = wam.aggregate_history(
                self.visual_history_buffer.visual_history())
            if mode == "train":
                future_state_pred = wam.predict_future(visual_history)

        trajectory = self.Reactive_E2E(camera_tiles, map_input, visual_history, egomotion_history,
        projection=projection, geometry_type=geometry_type, image_transform=image_transform,
        mode=mode, trajectory_target=trajectory_target, **kwargs)

        # Inference: just the trajectory. Training with the World Model on: also
        # return the predicted future state (the differentiable JEPA loss itself is
        # computed in the training loop via the windowed WorldActionModel path).
        if self.World_Action_Model_E2E is not None and mode == "train":
            return trajectory, future_state_pred
        return trajectory
        
    

