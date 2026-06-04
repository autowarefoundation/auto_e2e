import torch
import torch.nn as nn


class FutureState(nn.Module):
    def __init__(self):
        super(FutureState, self).__init__()

        # Predict future visual features (4 timesteps × 1440 channels = 5760)
        self.predict_future_1 = nn.Conv2d(1440, 2880, 3, 1, 1)
        self.predict_future_2 = nn.Conv2d(2880, 5760, 3, 1, 1)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features):
        # fused_features: [B, 1440, 7, 7]

        # Predicting 4 future visual feature vectors over a
        # 6.4s horizon equivalent to 1.6s intervals
        future_features = self.predict_future_1(fused_features)
        future_features = self.activation(future_features)
        future_features = self.predict_future_2(future_features)

        # Split into 4 future feature vectors: each [B, 1440, 7, 7]
        future_visual_features = torch.chunk(future_features, chunks=4, dim=1)

        return future_visual_features
