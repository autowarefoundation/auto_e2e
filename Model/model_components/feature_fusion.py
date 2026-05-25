import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

        # Adaptive pooling to achieve 7x7 resolution
        self.pool = nn.AdaptiveMaxPool2d(7)
 
    def forward(self, features):

        # Applying max-pooling to higher resolution
        # features to match 7x7 size of lowest resolution
        # feature map and permute the features to be in
        # Batch x Channels x Height x Width format

        f0 = self.pool(features[0].permute(0, 3, 1, 2))
        f1 = self.pool(features[1].permute(0, 3, 1, 2))
        f2 = self.pool(features[2].permute(0, 3, 1, 2))
        f3 = features[3].permute(0, 3, 1, 2)

        # Concatenate features across channels to form a
        # fused multi-scale feature map
        fused_features = torch.cat((f0, f1, f2, f3), dim=1)

        return fused_features   