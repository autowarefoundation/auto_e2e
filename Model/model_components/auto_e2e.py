from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .reactive_e2e import ReactiveE2E, StatefulCameraFPNCache
from .world_action_model import RollingHistoryBuffer, WorldActionModel

# Egomotion history is 64 timesteps x [speed, acceleration, yaw_rate, curvature]
# (see data_parsing/*/egomotion.py — the layout is identical for KITScenes, L2D
# and PhysicalAI).
_EGOMOTION_SIGNALS = 4
_SPEED_INDEX = 0
_ACCELERATION_INDEX = 1

# Raw speed in m/s divided by 33, corresponding to a max speed of 74 mph.
_SPEED_SCALE = 33.0
# Raw acceleration in m/s^2 divided by 8, since that equates to a harsh
# emergency braking manoeuvre.
_ACCELERATION_SCALE = 8.0

# Matrix-style geometry kwargs are not parameters. Without an explicit check
# **kwargs would absorb them and the run would silently fall back to
# geometry_type="pseudo", a learned prior rather than real geometry, so they are
# rejected by name.
_REMOVED_GEOMETRY_KWARGS = (
    "camera_params", "camera_matrices", "calib", "calibration",
    "intrinsics", "extrinsics", "projection_matrix",
)


def normalize_egomotion(egomotion_history):
    """Scale the speed and acceleration signals of an egomotion history.

    Args:
        egomotion_history: ``[..., T * 4]`` history, last dim laid out as
            ``T`` repeats of ``[speed, acceleration, yaw_rate, curvature]``.

    Returns:
        A tensor of the same shape, dtype and device with the speed and
        acceleration columns scaled and the angular signals untouched.
    """
    width = egomotion_history.shape[-1]
    if width % _EGOMOTION_SIGNALS != 0:
        raise ValueError(
            f"egomotion history width {width} is not a multiple of "
            f"{_EGOMOTION_SIGNALS} signals per timestep"
        )

    scales = torch.ones(
        _EGOMOTION_SIGNALS,
        dtype=egomotion_history.dtype,
        device=egomotion_history.device,
    )
    scales[_SPEED_INDEX] = 1.0 / _SPEED_SCALE
    scales[_ACCELERATION_INDEX] = 1.0 / _ACCELERATION_SCALE

    timesteps = width // _EGOMOTION_SIGNALS
    scaled = egomotion_history.reshape(
        *egomotion_history.shape[:-1], timesteps, _EGOMOTION_SIGNALS
    ) * scales
    return scaled.reshape(egomotion_history.shape)


class AutoE2E(nn.Module):
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
                 planner_mode="bezier", planner_kwargs=None,
                 enable_world_model=False, world_model_kwargs=None,
                 enable_reasoning=False, reasoning_mode="none",
                 reasoning_kwargs: Optional[Dict[str, Any]] = None,
                 enable_bev_segmentation=False,
                 bev_segmentation_classes=8,
                 enable_route_reconstruction=False,
                 auxiliary_output_size=None):
        super(AutoE2E, self).__init__()

        # Reactive model which runs at 10Hz and processes multi-camera inputs,
        # a rendered map image and egomotion history to predict a driving
        # trajectory to reach the near-horizon navigational goal. The reasoning
        # branch lives inside ReactiveE2E, after TemporalMemory, so it sees the
        # effective visual/ego context.
        self.Reactive_E2E = ReactiveE2E(backbone=backbone, num_views=num_views, embed_dim=embed_dim,
                 is_pretrained=is_pretrained,
                 image_feature_size=image_feature_size, view_fusion_kwargs=view_fusion_kwargs,
                 num_timesteps=num_timesteps, num_signals=num_signals,
                 egomotion_dim=egomotion_dim,
                 visual_history_dim=visual_history_dim,
                 map_type=map_type,
                 map_context_channels=map_context_channels,
                 route_channels=route_channels,
                 enable_route_conditioning=enable_route_conditioning,
                 route_encoder_hidden_channels=route_encoder_hidden_channels,
                 map_fusion_mode=map_fusion_mode, map_fusion_kwargs=map_fusion_kwargs,
                 temporal_memory_mode=temporal_memory_mode, temporal_memory_kwargs=temporal_memory_kwargs,
                 planner_mode=planner_mode, planner_kwargs=planner_kwargs,
                 enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
                 reasoning_kwargs=reasoning_kwargs,
                 enable_bev_segmentation=enable_bev_segmentation,
                 bev_segmentation_classes=bev_segmentation_classes,
                 enable_route_reconstruction=enable_route_reconstruction,
                 auxiliary_output_size=auxiliary_output_size)
        self.enable_reasoning = enable_reasoning

        # World Action Model (slow, ~1Hz): encodes the multi-camera history into
        # the Encoded Visual History (fed to the reactive planner) and predicts
        # future visual features (JEPA). Reuses the reactive backbone; the JEPA
        # target is a frozen copy of it. Opt-in, off by default.
        self.World_Action_Model_E2E: Optional[WorldActionModel] = None
        self.visual_history_buffer: Optional[RollingHistoryBuffer] = None
        if enable_world_model:
            wmk = dict(world_model_kwargs or {})
            history_len = wmk.pop("history_len", 4)
            wmk.setdefault("view_aggregator", "attention")
            # The JEPA operates on the backbone's last feature map, whose channel
            # count is backbone-dependent (swin_v2_tiny=768, res_net_50=2048, ...).
            wmk.setdefault(
                "feature_channels",
                self.Reactive_E2E.Backbone.feature_channels[-1],
            )
            # frame_embed_dim * history_len must equal the planner's
            # visual_history_dim.
            if visual_history_dim % history_len != 0:
                raise ValueError(
                    f"history_len ({history_len}) must divide visual_history_dim "
                    f"({visual_history_dim}); got remainder "
                    f"{visual_history_dim % history_len}. Pick a divisor of "
                    f"{visual_history_dim} (e.g. 1,2,4,7,8)."
                )
            # jepa_loss indexes the packed future window, whose length follows
            # history_len.
            wmk.setdefault("num_future_steps", history_len)
            self.World_Action_Model_E2E = WorldActionModel(
                backbone=self.Reactive_E2E.Backbone,
                frame_embed_dim=visual_history_dim // history_len,
                history_len=history_len, num_views=num_views, **wmk,
            )
            self.visual_history_buffer = RollingHistoryBuffer(history_len=history_len)

    # Bound as a staticmethod rather than an instance attribute so the module
    # stays deep-copyable and picklable — a closure stored on self is neither.
    normalize_egomotion = staticmethod(normalize_egomotion)

    def reset_visual_history(self):
        """Clear the World Model's rolling buffer (call between sequences)."""
        if self.visual_history_buffer is not None:
            self.visual_history_buffer = RollingHistoryBuffer(
                history_len=self.visual_history_buffer.history_len)

    def create_stateful_camera_fpn_cache(
        self,
    ) -> StatefulCameraFPNCache:
        """Create an inference cache for one BEVFormer T8 camera stream."""
        return self.Reactive_E2E.create_stateful_camera_fpn_cache()

    def forward(self, camera_tiles, map_context, visual_history,
                egomotion_history, route_mask=None, map_valid=None,
                route_valid=None,
                projection=None, geometry_type=None, image_transform=None,
                camera_history_tiles=None, history_projections=None,
                camera_fpn_cache=None,
                camera_fpn_stream_ids=None,
                camera_fpn_timestamps_us=None,
                front_camera_tile=None, front_camera_fpn_tile=None,
                front_camera_fpn_available=None,
                front_projection=None,
                front_image_transform=None,
                mode="train", trajectory_target=None,
                history_frames=None, future_frames=None,
                return_auxiliary=False,
                precomputed_image_bev=None,
                return_image_bev=False,
                compute_bev_segmentation=True,
                compute_route_reconstruction=True,
                bev_only=False,
                return_planner_loss=False,
                planner_loss_fn=None,
                **kwargs):
        """
        Run the full autonomous-driving pipeline.

        Return contract:
            * Inference (``mode != "train"``), or train with both branches off →
              a single trajectory tensor ``[B, num_timesteps * num_signals]``.
            * Train mode with the World Model and/or the reasoning branch on →
              ``(trajectory, aux_outputs)`` where ``aux_outputs`` is a dict with
              ``"future_state_pred"`` (World Model) and/or ``"reasoning_pred"``
              (HorizonReasoningPrediction).

        Args:
            camera_tiles: (B, V, 3, H, W) — V real camera images (the nav-map is
                a separate navigation input, not a camera view).
            map_context: (B, C_map, H_map, W_map) — semantic BEV map.
            route_mask: (B, 2, H_map, W_map) — selected route corridor and
                destination marker. It enters only the Reactive branch.
            map_valid / route_valid: (B,) explicit validity gates.
                The checkpoint-level ``enable_route_conditioning`` flag can
                force the route gate off for a controlled baseline.
            visual_history: (B, T, visual_history_dim) or (B, visual_history_dim).
            egomotion_history: (B, T, egomotion_dim) or (B, egomotion_dim).
            projection: Optional CameraProjectionModel operator — the geometry
                ABI (Pinhole / FTheta / Pseudo). No [B,V,3,4] matrix argument;
                construct PinholeProjection(matrix) if you have a pinhole matrix.
            geometry_type: Optional explicit geometry label ("pinhole",
                "rectified_pinhole", "ftheta", "pseudo") passed to BEV fusion.
            image_transform: Optional ImageTransform for the model-input frame.
            front_camera_tile: Optional native-resolution front image tensor.
            front_camera_fpn_tile: Optional base-resolution Front image packed
                identically to historical Front frames for cache insertion.
            front_camera_fpn_available: Per-sample validity for the exact
                base-resolution Front companion.
            front_projection: Optional one-view projection for the native front.
            camera_history_tiles: Optional seven-frame camera history for T8.
            history_projections: Optional current-ego-aligned history geometry.
            camera_fpn_cache: Optional per-stream stateful T8 FPN cache.
            camera_fpn_stream_ids: Ordered scene or stream identity per sample.
            camera_fpn_timestamps_us: Source timestamps sampled exactly every
                500 ms. Do not synthesize continuity across dropped frames.
            precomputed_image_bev: Optional detached image BEV from an earlier
                inference pass over the same camera inputs.
            return_image_bev: Return the detached image BEV for route-only
                counterfactual inference.
            mode: "train" also returns aux branch outputs for their losses.

        Returns:
            trajectory, or (trajectory, aux_outputs) in train mode with a branch on.

        Raises:
            TypeError: if a removed geometry kwarg (e.g. ``camera_params``) is
                passed. See ``_REMOVED_GEOMETRY_KWARGS``.
        """
        removed = sorted(set(kwargs) & set(_REMOVED_GEOMETRY_KWARGS))
        if removed:
            raise TypeError(
                f"{type(self).__name__}.forward() got removed geometry argument(s) "
                f"{removed}. The [B,V,3,4] matrix argument was replaced by the "
                f"projection ABI (#77/#107):\n\n"
                f"    from model_components.view_fusion.projection import PinholeProjection\n"
                f"    model(camera_tiles, map_input, visual_history, egomotion_history,\n"
                f"          projection=PinholeProjection(camera_params),  # [B,V,3,4]\n"
                f"          geometry_type='pinhole', mode=...)\n\n"
                f"This is an error and not a warning because **kwargs would otherwise "
                f"absorb the argument silently and the model would run on "
                f"geometry_type='pseudo' — a learned spatial prior, not your calibration."
            )

        # Rebinds rather than mutating so the caller's batch is left untouched.
        egomotion_history = self.normalize_egomotion(egomotion_history)

        # World Action Model (1 Hz): produce the Encoded Visual History fed to the
        # reactive planner + reasoning branch, and (in training) the predicted
        # future feature maps for the JEPA loss. Two mutually exclusive paths:
        #
        #  A. Windowed training path (when ``history_frames`` is given):
        #     stateless and differentiable, so it is safe under shuffled batch
        #     training. It replaces the caller's visual_history so the planner is
        #     conditioned on the WM history.
        #  B. Rolling-buffer inference/rollout path: pushes a detached per-frame
        #     embedding into a per-sequence FIFO. Not safe to share across
        #     shuffled batches; call ``reset_visual_history()`` between
        #     independent sequences.
        future_state_pred = None
        if self.World_Action_Model_E2E is not None:
            wam = self.World_Action_Model_E2E
            if history_frames is not None:
                # A. Windowed, differentiable path. history_frames: [B, T, V, 3, H, W]
                # (or [B, T, 3, H, W]) oldest→newest, current frame last.
                history_concat = wam.encode_history(history_frames)
                visual_history = wam.aggregate_history(history_concat)
                # The WM yields an aggregated 2-D [B, dim] visual history. Keep
                # egomotion_history at the same rank, since a rank-branching
                # temporal memory (e.g. one_hz) needs both inputs to agree.
                if egomotion_history.ndim == 3:
                    egomotion_history = egomotion_history[:, -1]
                # JEPA trains the WM through the non-detached visual_history.
                if mode == "train":
                    future_state_pred = wam.predict_future(visual_history)
                # The planner reads a stop-gradient WM representation so the
                # trajectory loss does not add a second gradient path into the
                # WM, which is shaped by JEPA only. This matches path B.
                visual_history = visual_history.detach()
            else:
                # B. Rolling-buffer path. Encode the current 1 Hz multi-view frame;
                # push a detached copy so the planner's history is pure memory.
                visual_embedding, _ = wam(camera_tiles)
                self.visual_history_buffer.push(visual_embedding.detach())  # type: ignore[union-attr]
                visual_history = wam.aggregate_history(
                    self.visual_history_buffer.visual_history())  # type: ignore[union-attr]
                # Same rank-collapse as path A.
                if egomotion_history.ndim == 3:
                    egomotion_history = egomotion_history[:, -1]
                if mode == "train":
                    future_state_pred = wam.predict_future(visual_history)

        # ReactiveE2E returns (trajectory, aux) when it has auxiliary outputs.
        reactive_out = self.Reactive_E2E(
            camera_tiles, map_context, visual_history, egomotion_history,
            route_mask=route_mask,
            map_valid=map_valid,
            route_valid=route_valid,
            projection=projection, geometry_type=geometry_type,
            image_transform=image_transform,
            camera_history_tiles=camera_history_tiles,
            history_projections=history_projections,
            camera_fpn_cache=camera_fpn_cache,
            camera_fpn_stream_ids=camera_fpn_stream_ids,
            camera_fpn_timestamps_us=camera_fpn_timestamps_us,
            front_camera_tile=front_camera_tile,
            front_camera_fpn_tile=front_camera_fpn_tile,
            front_camera_fpn_available=front_camera_fpn_available,
            front_projection=front_projection,
            front_image_transform=front_image_transform,
            mode=mode,
            return_auxiliary=return_auxiliary,
            precomputed_image_bev=precomputed_image_bev,
            return_image_bev=return_image_bev,
            compute_bev_segmentation=compute_bev_segmentation,
            compute_route_reconstruction=compute_route_reconstruction,
            bev_only=bev_only,
            trajectory_target=trajectory_target,
            return_planner_loss=return_planner_loss,
            planner_loss_fn=planner_loss_fn,
            **kwargs,
        )
        reactive_aux = {}
        if isinstance(reactive_out, tuple):
            trajectory, reactive_aux = reactive_out
        else:
            trajectory = reactive_out

        # future_frames travels with the prediction so the training loop can
        # call jepa_loss(future_state_pred, future_frames) directly.
        if mode == "train" and self.World_Action_Model_E2E is not None:
            reactive_aux.update({
                "future_state_pred": future_state_pred,
                "future_frames": future_frames,
            })
        if reactive_aux:
            aux_outputs = reactive_aux
            return trajectory, aux_outputs
        return trajectory
