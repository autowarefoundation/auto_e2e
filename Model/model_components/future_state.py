import torch
import torch.nn as nn


class FutureState(nn.Module):
    """Predict future BEV feature maps conditioned on the planner's ego_hidden.

    The legacy 14-dim compressed visual vector is replaced by the GRU hidden
    state (``ego_hidden``, 256-dim) coming from the trajectory planner (GRU
    or Flow Matching). ``ego_hidden`` summarises the planner's intent over
    the prediction horizon, so the future feature predictions reflect the
    trajectory the model expects to drive.

    Optional action conditioning (``action_dim``): when set, the flattened
    trajectory ``[B, num_timesteps * num_signals]`` is projected and added as
    an extra per-channel bias (on top of ``ego_hidden``). This makes the
    world model explicitly action-conditioned, enabling counterfactual
    rollouts ("what would the scene look like if I drove trajectory A vs B?").
    With ``action_dim=None`` (default) the behaviour is identical to the
    unconditioned module.
    """

    def __init__(self, embed_dim=256, ego_hidden_dim=256, action_dim=None):
        super(FutureState, self).__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # Project ego_hidden to per-channel bias broadcast over BEV spatial dims
        self.ego_proj = nn.Linear(ego_hidden_dim, embed_dim)

        # Optional projection of the planned trajectory (action) to a second
        # per-channel bias for counterfactual, action-conditioned rollouts.
        self.action_proj = (
            nn.Linear(action_dim, embed_dim) if action_dim is not None else None
        )

        # Predict future visual features (4 timesteps × C channels = 4C)
        self.predict_future_1 = nn.Conv2d(embed_dim, 2*embed_dim, 3, 1, 1)
        # WARNING: at full 450x300 BEV resolution the 4*embed_dim output is
        # memory-intensive — roughly 450 * 300 * 4 * 256 * 4 bytes ≈ 550MB per
        # sample in fp32. Training will likely require mixed precision (bf16/fp16)
        # or spatial downsampling of fused_features before FutureState to fit on
        # commodity GPUs.
        self.predict_future_2 = nn.Conv2d(2*embed_dim, 4*embed_dim, 3, 1, 1)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features, ego_hidden, trajectory=None):
        # fused_features: [B, C, H, W]; ego_hidden: [B, ego_hidden_dim]
        # trajectory (optional): [B, action_dim] flattened planned trajectory,
        # only used when the module was built with ``action_dim`` set.

        # Inject planner intent as a per-channel bias broadcast over the BEV grid
        bias = self.ego_proj(ego_hidden).view(-1, self.embed_dim, 1, 1)
        conditioned = fused_features + bias

        if trajectory is not None:
            if self.action_proj is None:
                raise ValueError(
                    "FutureState received a trajectory but was built with "
                    "action_dim=None; pass action_dim at construction to "
                    "enable action conditioning."
                )
            action_bias = self.action_proj(trajectory).view(
                -1, self.embed_dim, 1, 1
            )
            conditioned = conditioned + action_bias

        # Predicting 4 future visual feature vectors over a
        # 6.4s horizon equivalent to 1.6s intervals
        future_features = self.predict_future_1(conditioned)
        future_features = self.activation(future_features)
        future_features = self.predict_future_2(future_features)

        # Split into 4 future feature vectors: each [B, C, H, W]
        future_visual_features = torch.chunk(future_features, chunks=4, dim=1)

        return future_visual_features
