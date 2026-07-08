from typing import Any, Dict, Optional

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
                 enable_world_model=False, world_model_kwargs=None,
                 enable_reasoning=False, reasoning_mode="none",
                 reasoning_kwargs: Optional[Dict[str, Any]] = None):
        super(AutoE2E, self).__init__()

        # Reactive model which runs at 10Hz and processes multi-camera inputs
        # a rendered map image and egomotion history to predict a driving trajectory
        # to reach the near-horizon navigational goal.
        #
        # The reasoning branch lives INSIDE ReactiveE2E (after TemporalMemory,
        # where ego_ctx is available) rather than as a pre-ReactiveE2E history
        # rewrite — so the head sees the effective visual/ego context and the
        # planner coupling is a first-class planner argument (#98).
        self.Reactive_E2E = ReactiveE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim,
                 is_pretrained=is_pretrained,
                 image_feature_size=image_feature_size, view_fusion_kwargs=view_fusion_kwargs,
                 num_timesteps=num_timesteps, num_signals=num_signals, egomotion_dim=egomotion_dim,
                 visual_history_dim=visual_history_dim,
                 map_type=map_type, map_in_channels=map_in_channels,
                 map_fusion_mode=map_fusion_mode, map_fusion_kwargs=map_fusion_kwargs,
                 temporal_memory_mode=temporal_memory_mode, temporal_memory_kwargs=temporal_memory_kwargs,
                 planner_mode=planner_mode, planner_kwargs=planner_kwargs,
                 enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
                 reasoning_kwargs=reasoning_kwargs)
        self.enable_reasoning = enable_reasoning

        # World Action Model (slow, ~1Hz): encodes the multi-camera history into
        # the Encoded Visual History (fed to the reactive planner) and predicts
        # future visual features (JEPA). Reuses the reactive backbone (one shared
        # backbone; the JEPA target is a frozen copy of it). Opt-in (default OFF)
        # so the reactive-only default is byte-identical.
        self.World_Action_Model_E2E: Optional[WorldActionModel] = None
        self.visual_history_buffer: Optional[RollingHistoryBuffer] = None
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

        Return contract:
            * Inference (``mode != "train"``), or train with both branches off →
              a single trajectory tensor ``[B, num_timesteps * num_signals]``.
            * Train mode with the World Model and/or the reasoning branch on →
              ``(trajectory, aux_outputs)`` where ``aux_outputs`` is a dict with
              ``"future_state_pred"`` (World Model) and/or ``"reasoning_pred"``
              (HorizonReasoningPrediction). A dict avoids a positional-tuple
              that grows with every optional branch (#98 Task 4.2).

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
            mode: "train" also returns aux branch outputs for their losses.

        Returns:
            trajectory, or (trajectory, aux_outputs) in train mode with a branch on.
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
            self.visual_history_buffer.push(visual_embedding.detach())  # type: ignore[union-attr]
            # The planner and the future prediction read the SAME history (post-push,
            # i.e. including the current frame) so the JEPA context stays aligned
            # with what the planner actually sees.
            visual_history = wam.aggregate_history(
                self.visual_history_buffer.visual_history())  # type: ignore[union-attr]
            if mode == "train":
                future_state_pred = wam.predict_future(visual_history)

        # The reasoning branch runs INSIDE ReactiveE2E (after TemporalMemory).
        # In train mode with reasoning on, ReactiveE2E returns
        # (trajectory, reasoning_pred); otherwise just the trajectory.
        reactive_out = self.Reactive_E2E(
            camera_tiles, map_input, visual_history, egomotion_history,
            projection=projection, geometry_type=geometry_type,
            image_transform=image_transform,
            mode=mode, trajectory_target=trajectory_target, **kwargs,
        )
        reasoning_pred = None
        if self.enable_reasoning and mode == "train":
            trajectory, reasoning_pred = reactive_out
        else:
            trajectory = reactive_out

        # Assemble aux outputs (dict, not a growing positional tuple).
        if mode == "train" and (
            self.World_Action_Model_E2E is not None or reasoning_pred is not None
        ):
            aux_outputs = {
                "future_state_pred": future_state_pred,
                "reasoning_pred": reasoning_pred,
            }
            return trajectory, aux_outputs
        return trajectory
        
    

