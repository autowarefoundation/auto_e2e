import torch
import torch.nn as nn


class FutureState(nn.Module):
    """Predict future BEV feature maps conditioned on the planner's ego_hidden.

    The legacy 14-dim compressed visual vector is replaced by the GRU hidden
    state (``ego_hidden``, 256-dim) coming from TrajectoryPlanner. ``ego_hidden``
    summarises the planner's intent over the prediction horizon, so the future
    feature predictions reflect the trajectory the model expects to drive.
    """

    def __init__(self, embed_dim=256, ego_hidden_dim=256):
        super(FutureState, self).__init__()

        self.embed_dim = embed_dim

        # Project ego_hidden to per-channel bias broadcast over BEV spatial dims
        self.ego_proj = nn.Linear(ego_hidden_dim, embed_dim)

        # Predict future visual features (4 timesteps × C channels = 4C)
        self.predict_future_1 = nn.Conv2d(embed_dim, 2*embed_dim, 3, 1, 1)
        self.predict_future_2 = nn.Conv2d(2*embed_dim, 4*embed_dim, 3, 1, 1)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features, ego_hidden):
        # fused_features: [B, C, H, W]; ego_hidden: [B, ego_hidden_dim]

        # Inject planner intent as a per-channel bias broadcast over the BEV grid
        bias = self.ego_proj(ego_hidden).view(-1, self.embed_dim, 1, 1)
        conditioned = fused_features + bias

        # Predicting 4 future visual feature vectors over a
        # 6.4s horizon equivalent to 1.6s intervals
        future_features = self.predict_future_1(conditioned)
        future_features = self.activation(future_features)
        future_features = self.predict_future_2(future_features)

        # Split into 4 future feature vectors: each [B, C, H, W]
        future_visual_features = torch.chunk(future_features, chunks=4, dim=1)

        return future_visual_features
