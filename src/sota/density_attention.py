from torch import nn
import torch.nn.functional as F


from torch import nn
import torch
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Helper module for channel-wise attention."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Helper module for spatial attention."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_features = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_features))
        return x * attention


class DensityAttention(nn.Module):
    """
    A more sophisticated Attention Module for Breast Density Classification,
    inspired by the Convolutional Block Attention Module (CBAM).
    """

    def __init__(
        self, in_channels: int, out_features: int = 4, reduction_ratio: int = 16
    ):
        super(DensityAttention, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

        # Final classifier using both average and max pooled features
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_channels, out_features),
        )

    def forward(self, x):
        # Sequentially apply channel and then spatial attention
        x = self.channel_attention(x)
        attended_features = self.spatial_attention(x)

        # Aggregate features using both average and max pooling
        avg_pool = F.adaptive_avg_pool2d(attended_features, (1, 1)).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(attended_features, (1, 1)).view(x.size(0), -1)
        global_vec = torch.cat([avg_pool, max_pool], dim=1)

        # Classify the aggregated feature vector
        logits = self.classifier(global_vec)

        return logits


"""
class DensityAttention(nn.Module):
    
    #Spatial Attention Module for Breast Density Classification.
    

    def __init__(self, in_channels: int, out_features: int = 4):
        super(DensityAttention, self).__init__()

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
"""
