import torch
import torch.nn as nn

class DrivingPolicy(nn.Module):
    def __init__(self):
        super(DrivingPolicy, self).__init__()

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(1440, 3, 3, 1, 1)

        # Linear layers to process reduced features
        self.fc1 = nn.Linear(1432, 1432)
        self.fc2 = nn.Linear(1432, 716)
        self.fc3 = nn.Linear(716, 128)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()
 
    def forward(self, fused_features, ego_motion):

        # Reduce channels
        feature_map = self.reduce_channels(fused_features)
        feature_vector = torch.cat((torch.flatten(feature_map), ego_motion), dim=0)
        
        # Multi-layer perceptron
        f1 = self.fc1(feature_vector)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        trajectory = self.fc3(f2)

        return trajectory   