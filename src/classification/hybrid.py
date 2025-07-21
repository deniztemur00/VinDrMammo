import torch
import torch.nn as nn
import timm
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class ModelConfig:
    cnn_name: str = "resnet50"
    vit_name: str = "vit_base_patch16_224"
    num_classes: int = 5
    fusion: str = "concat"
    pretrained: bool = True


class CNN_ViT_Hybrid(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fusion = config.fusion

        self.cnn = timm.create_model(
            config.cnn_name,
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        cnn_out_dim = self.cnn.num_features

        # Load pretrained ViT backbone
        self.vit = timm.create_model(
            config.vit_name, in_chans=1, pretrained=True, num_classes=0
        )
        vit_out_dim = self.vit.num_features

        # Fusion layer
        if config.fusion == "concat":
            fusion_dim = cnn_out_dim + vit_out_dim
        elif config.fusion == "add":
            fusion_dim = min(cnn_out_dim, vit_out_dim)
            self.cnn_proj = nn.Linear(cnn_out_dim, fusion_dim)
            self.vit_proj = nn.Linear(vit_out_dim, fusion_dim)
        else:
            raise ValueError("fusion must be 'concat' or 'add'")

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes),
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)

        if self.fusion == "concat":
            fused = torch.cat([cnn_feat, vit_feat], dim=1)
        elif self.fusion == "add":
            fused = self.cnn_proj(cnn_feat) + self.vit_proj(vit_feat)

        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate CE loss without reduction to get per-example losses
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # Calculate the probability of the correct class p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Calculate the focal loss term: (1 - pt)^gamma * ce_loss
        focal_term = (1 - pt) ** self.gamma * ce_loss

        # Apply the specified reduction
        if self.reduction == "mean":
            return focal_term.mean()
        elif self.reduction == "sum":
            return focal_term.sum()
        else:  # 'none'
            return focal_term
