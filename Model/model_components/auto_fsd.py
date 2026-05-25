from .backbone import Backbone
import torch.nn as nn

class AutoFSD(nn.Module):
    def __init__(self):
        super(AutoFSD, self).__init__()
        
        # Backbone feature extractor
        self.Backbone = Backbone()
   

    def forward(self,image):
        features = self.Backbone(image)
        return features