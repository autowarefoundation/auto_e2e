import torch
import torch.nn as nn


class DrivingPolicy(nn.Module):
    def __init__(self, visual_feature_dim=1440 * 7 * 7, visual_history_dim=896, egomotion_dim=256):
        super(DrivingPolicy, self).__init__()

        # Dimensions for the compressed visual feature vector
        compressed_dim = 14
        visual_flat_dim = 3 * 7 * 7  # After channel reduction

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(1440, 3, 3, 1, 1)

        # Total input dimension to MLP
        mlp_input_dim = visual_flat_dim + visual_history_dim + egomotion_dim

        # Linear layers to process reduced features
        self.fc1 = nn.Linear(mlp_input_dim, mlp_input_dim)
        self.fc2 = nn.Linear(mlp_input_dim, 1164)
        self.fc3 = nn.Linear(1164, 128)

        # Visual history compression layer
        self.compress_vision = nn.Linear(visual_flat_dim, compressed_dim)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()

    def forward(self, fused_features, visual_history, egomotion_history):
        # fused_features: [B, 1440, 7, 7]

        # Reduce visual feature channels: [B, 3, 7, 7]
        feature_map = self.reduce_channels(fused_features)

        # Flatten preserving batch dimension: [B, 3*7*7]
        visual_feature_vector = torch.flatten(feature_map, start_dim=1)

        # Concatenate with visual scene history and egomotion history
        feature_vector = torch.cat((visual_feature_vector,
                                    visual_history, egomotion_history), dim=1)

        # Multi-layer perceptron
        f1 = self.fc1(feature_vector)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        # Trajectory output - 64 x (acceleration & curvature) at
        # 10Hz yielding a 6.4s future time horizon prediction
        trajectory = self.fc3(f2)

        # Compressed visual feature vector of length 14 to form visual history
        compressed_visual_feature_vector = self.compress_vision(visual_feature_vector)

        return trajectory, compressed_visual_feature_vector
