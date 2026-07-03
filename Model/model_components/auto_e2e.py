import torch.nn as nn
from .reactive_e2e import ReactiveE2E


class AutoE2E(nn.Module):
    def __init__(self, backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                 is_pretrained=True,
                 image_feature_size=8, view_fusion_kwargs=None,
                 num_timesteps=64, num_signals=2, egomotion_dim=256,
                 visual_history_dim=896,
                 map_type="rasterized", map_in_channels=3,
                 map_fusion_mode="residual", map_fusion_kwargs=None,
                 temporal_memory_mode="no_memory", temporal_memory_kwargs=None,
                 planner_mode="bezier", planner_kwargs=None):
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

        ### Placeholder for self.World_Action_Model_E2E which processes a 1Hz or tunable 
        ### stream of images and encodes it as a visual history vector which is fed as input
        ### to the the Reactive_E2E module and outputs the future feature state which is used as
        ### JEPA loss during training

        trajectory = self.Reactive_E2E(camera_tiles, map_input, visual_history, egomotion_history,
        projection=projection, geometry_type=geometry_type, image_transform=image_transform,
        mode=mode, trajectory_target=trajectory_target, **kwargs)


        return trajectory
        
    

