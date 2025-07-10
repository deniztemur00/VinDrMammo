import torch
from torch import nn
import torch.nn.functional as F
from GMIC_adaptation.config import GlobalConfig


class DensityAttention(nn.Module):
    """
    Spatial Attention Module for Breast Density Classification.
    """

    def __init__(self, in_channels: int, out_features: int = 4):
        """
        Initializes the DensityAttention module.
        :param in_channels: Number of channels in the input feature map (e.g., 256).
        :param out_features: Number of output classes for density (e.g., 4 for A-D).
        """
        super(DensityAttention, self).__init__()

        # These convolutional layers function as the Q, K projections to create
        # a spatial attention map. The 'value' is the input feature map itself.
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1, bias=False),
        )

        # Final classifier
        self.classifier = nn.Linear(in_channels, out_features)

    def forward(self, x):
        # attention_scores shape: (N, 1, H, W)
        attention_scores = self.attention_conv(x)


        attention_map = F.softmax(attention_scores.view(x.size(0), -1), dim=1)


        attention_map = attention_map.view(x.size(0), 1, x.size(2), x.size(3))

        attended_features = x * attention_map
        global_vec = attended_features.mean(dim=[2, 3])


        logits = self.classifier(global_vec)

        return logits
