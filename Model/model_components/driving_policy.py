import torch.nn as nn

class DrivingPolicy(nn.Module):
    def __init__(self):
        super(DrivingPolicy, self).__init__()

        # Linear layers to process fused features
        self.fc1 = nn.Linear(1440, 1440)
        self.fc2 = nn.Linear(1440, 720)
        self.fc3 = nn.Linear(720, 64)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.activation = nn.GELU()
 
    def forward(self, fused_features):

        # Multi-layer perceptron
        f1 = self.fc1(fused_features)
        f1 = self.activation(f1)
        f1 = self.dropout(f1)

        f2 = self.fc2(f1)
        f2 = self.activation(f2)
        f2 = self.dropout(f2)

        trajectory = self.fc3(f2)

        return trajectory   