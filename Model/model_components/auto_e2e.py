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
            self.World_Action_Model_E2E = WorldActionModel(
                backbone=self.Reactive_E2E.Backbone,
                frame_embed_dim=visual_history_dim // history_len,
                history_len=history_len, **wmk,
            )
            self.visual_history_buffer = RollingHistoryBuffer(history_len=history_len)

    def reset_visual_history(self):
        """Clear the World Model's rolling buffer (call between sequences)."""
        if self.visual_history_buffer is not None:
            self.visual_history_buffer = RollingHistoryBuffer(
                history_len=self.visual_history_buffer.history_len)


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
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            camera_params: Optional (B, V, 3, 4) ego-to-pixel projection matrices.
            mode: "train" to produce future_visual_features; anything else skips it.

        Returns:
            trajectory: (B, num_timesteps * num_signals)
            ego_hidden: (B, embed_dim)
            planner_loss: Used only when mode="train" during network training, otherwise set to None
        """

        # World Action Model (1Hz): encode the current multi-camera frame, push it
        # to the circular buffer (size N=4) to form the Encoded Visual History fed
        # to the reactive planner, and (in training) predict the future feature
        # state for the JEPA loss. Realizes the wiring sketched in this file by
        # @m-zain-khawaja (24/06):
        #   visual_embedding, future_state_pred = self.World_Action_Model_E2E(inputs)
        #   visual_history.push(visual_embedding)   # circular buffer size N (N=4)
        #   return trajectory  /  trajectory, future_state_pred   (infer / train)
        future_state_pred = None
        if self.World_Action_Model_E2E is not None:
            wam = self.World_Action_Model_E2E
            # aggregate the FIFO into the visual_history (concat-FIFO default, or
            # the opt-in temporal attention-pool); same 896 interface either way.
            vh_prev = wam.aggregate_history(self.visual_history_buffer.visual_history())
            visual_embedding, future_state_pred = wam(camera_tiles, vh_prev)
            self.visual_history_buffer.push(visual_embedding)
            visual_history = wam.aggregate_history(self.visual_history_buffer.visual_history())

        trajectory = self.Reactive_E2E(camera_tiles, map_input, visual_history, egomotion_history,
        camera_params=camera_params, mode=mode, trajectory_target=trajectory_target, **kwargs)

        # Inference: just the trajectory. Training with the World Model on: also
        # return the predicted future state (the JEPA loss is computed in the
        # training loop via WorldActionModel.jepa_loss).
        if self.World_Action_Model_E2E is not None and mode == "train":
            return trajectory, future_state_pred
        return trajectory
        
    

