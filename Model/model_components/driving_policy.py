import torch
import torch.nn as nn

class DrivingPolicy(nn.Module):
    def __init__(self):
        super(DrivingPolicy, self).__init__()

        # 2D Conv layer to reduce channels
        self.reduce_channels = nn.Conv2d(1440, 3, 3, 1, 1)

        # Linear layers to process reduced features
        self.fc1 = nn.Linear(2328, 2328)
        self.fc2 = nn.Linear(2328, 1164)
        self.fc3 = nn.Linear(1164, 128)

        # Visual history compression layer
        self.compress_vision = nn.Linear(1176, 14)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()
 
    def forward(self, fused_features, visual_history, egomotion_history):

        # Reduce visual feature channels
        feature_map = self.reduce_channels(fused_features)

        # Flatten visual features and concatenate with
        # visual scene history and egomotion history
        visual_feature_vector = torch.flatten(feature_map)
        feature_vector = torch.cat((visual_feature_vector, 
                                    visual_history, egomotion_history), dim=0)
        
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

        # Compressed visual feature vector of length 14 to form
        # visual history
        compressed_visual_feature_vector = \
            self.compress_vision(visual_feature_vector)

        return trajectory, compressed_visual_feature_vector   