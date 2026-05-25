import torch
import torch.nn as nn

class DrivingPolicy(nn.Module):
    def __init__(self):
        super(DrivingPolicy, self).__init__()

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(1440, 24, 3, 1, 1)

        # Linear layers to process fused features
        self.fc1 = nn.Linear(1176, 1176)
        self.fc2 = nn.Linear(1176, 588)
        self.fc3 = nn.Linear(588, 128)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()
 
    def forward(self, fused_features):

        # Reduce channels
        feature_map = self.reduce_channels(fused_features)
        feature_vector = torch.flatten(feature_map)

        # Multi-layer perceptron
        f1 = self.fc1(feature_vector)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        trajectory = self.fc3(f2)

        return trajectory   