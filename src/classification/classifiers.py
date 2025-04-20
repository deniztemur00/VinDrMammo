import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


from transformers import ViTModel, ViTConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BaseClassifierConfig:
    """Base configuration for classifiers."""

    num_birads_classes: int = 5  # Example: BI-RADS 1-5
    num_density_classes: int = 4  # Example: Density A-D
    input_channels: int = 3
    image_size: Tuple[int, int] = (224, 224)  # Default size for many models


@dataclass
class EfficientNetClassifierConfig(BaseClassifierConfig):
    """Configuration for the EfficientNet-based classifier."""

    model_name: str = "efficientnet_b3"  # Example, can be b0-b7
    pretrained: bool = True
    trainable_backbone_layers: int = 3  # Number of blocks from the end to unfreeze


@dataclass
class ResNetClassifierConfig(BaseClassifierConfig):
    """Configuration for the ResNet-based classifier."""

    pretrained: bool = True
    trainable_backbone_layers: int = 3  # 0 to 5


@dataclass
class ViTClassifierConfig(BaseClassifierConfig):
    """Configuration for the Vision Transformer-based classifier."""

    model_name: str = "google/vit-base-patch16-224-in21k"
    pretrained: bool = True


class BIRADSHead(nn.Module):
    """BI-RADS classification head."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DensityHead(nn.Module):
    """Breast density classification head."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for mammography."""

    def __init__(self, config: EfficientNetClassifierConfig):
        super().__init__()
        self.config = config

        weights = EfficientNet_B3_Weights.DEFAULT if config.pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        num_features = self.backbone.classifier[1].in_features

        self.birads_head = BIRADSHead(num_features, config.num_birads_classes)
        self.density_head = DensityHead(num_features, config.num_density_classes)

        # --- Layer Freezing Logic (Optional but Recommended for Pretrained) ---
        if config.pretrained:
            # Freeze all parameters initially
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the classifier head first (always trainable)
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

            # Unfreeze blocks from the end based on config.trainable_backbone_layers
            # EfficientNet blocks are in backbone.features
            num_blocks = len(self.backbone.features)
            for i in range(num_blocks - config.trainable_backbone_layers, num_blocks):
                if i >= 0:  # Ensure index is valid
                    for param in self.backbone.features[i].parameters():
                        param.requires_grad = True
        # --- End Freezing Logic ---

        # Replace the final classifier layer
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)  # Get features from the modified backbone
        birads_logits = self.birads_head(features)  # BI-RADS logits
        density_logits = self.density_head(features)  # Density logits

        outputs = {
            "birads_logits": birads_logits,
            "density_logits": density_logits,
        }
        return outputs


class ResNetClassifier(nn.Module):
    """ResNet-101 based classifier for mammography."""

    def __init__(self, config: ResNetClassifierConfig):
        super().__init__()
        self.config = config

        weights = ResNet101_Weights.DEFAULT if config.pretrained else None
        self.backbone = resnet101(weights=weights)

        # Freeze layers if needed
        if config.pretrained:
            # Freeze all layers first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze layers based on trainable_backbone_layers
            if config.trainable_backbone_layers > 0:  # Unfreeze layer4
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            if config.trainable_backbone_layers > 1:  # Unfreeze layer3
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
            # ... continue for layer2, layer1, conv1 if needed

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classifier

        self.birads_head = nn.Linear(num_features, config.num_birads_classes)
        self.density_head = nn.Linear(num_features, config.num_density_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        birads_logits = self.birads_head(features)
        density_logits = self.density_head(features)
        outputs = {
            "birads_logits": birads_logits,
            "density_logits": density_logits,
        }
        return outputs


class ViTClassifier(nn.Module):
    """Vision Transformer based classifier for mammography."""

    def __init__(self, config: ViTClassifierConfig):
        super().__init__()
        self.config = config

        vit_config = ViTConfig.from_pretrained(config.model_name)
        self.vit = ViTModel.from_pretrained(
            config.model_name, config=vit_config, add_pooling_layer=False
        )

        if not config.pretrained:
            self.vit.init_weights()  # Initialize from scratch if not pretrained

        num_features = vit_config.hidden_size

        self.birads_head = nn.Linear(num_features, config.num_birads_classes)
        self.density_head = nn.Linear(num_features, config.num_density_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vit(pixel_values=x)
        # Use the [CLS] token representation for classification
        cls_token_features = outputs.last_hidden_state[:, 0]

        birads_logits = self.birads_head(cls_token_features)
        density_logits = self.density_head(cls_token_features)
        outputs = {
            "birads_logits": birads_logits,
            "density_logits": density_logits,
        }
        return outputs
