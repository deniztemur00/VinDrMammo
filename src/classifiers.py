import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from transformers import ViTModel, ViTConfig
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class BaseClassifierConfig:
    """Base configuration for classifiers."""
    num_birads_classes: int = 5  # Example: BI-RADS 1-5
    num_density_classes: int = 4 # Example: Density A-D
    input_channels: int = 3
    image_size: Tuple[int, int] = (224, 224) # Default size for many models

@dataclass
class BasicCNNConfig(BaseClassifierConfig):
    """Configuration for the Basic CNN classifier."""
    base_filters: int = 16
    fc_units: int = 128

@dataclass
class ResNetClassifierConfig(BaseClassifierConfig):
    """Configuration for the ResNet-based classifier."""
    pretrained: bool = True
    trainable_backbone_layers: int = 3 # 0 to 5

@dataclass
class ViTClassifierConfig(BaseClassifierConfig):
    """Configuration for the Vision Transformer-based classifier."""
    model_name: str = "google/vit-base-patch16-224-in21k"
    pretrained: bool = True


class BasicCNN(nn.Module):
    """A simple CNN for mammography classification."""
    def __init__(self, config: BasicCNNConfig):
        super().__init__()
        self.config = config

        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.input_channels, config.base_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.base_filters, config.base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.base_filters * 2, config.base_filters * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
            nn.Flatten()
        )

        # Calculate flattened features size dynamically (example placeholder)
        # This needs adjustment based on actual input size and conv layers
        # For image_size (224, 224) and 3 maxpools: 224 / (2*2*2) = 28
        # flattened_size = config.base_filters * 4 * ( (config.image_size[0] // 8)**2 ) # Rough estimate
        flattened_size = config.base_filters * 4 # After AdaptiveAvgPool2d

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, config.fc_units),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.birads_head = nn.Linear(config.fc_units, config.num_birads_classes)
        self.density_head = nn.Linear(config.fc_units, config.num_density_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_layers(x)
        shared_features = self.fc_layers(features)
        birads_logits = self.birads_head(shared_features)
        density_logits = self.density_head(shared_features)
        return birads_logits, density_logits


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
            if config.trainable_backbone_layers > 0: # Unfreeze layer4
                 for param in self.backbone.layer4.parameters():
                      param.requires_grad = True
            if config.trainable_backbone_layers > 1: # Unfreeze layer3
                 for param in self.backbone.layer3.parameters():
                      param.requires_grad = True
            # ... continue for layer2, layer1, conv1 if needed

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove original classifier

        self.birads_head = nn.Linear(num_features, config.num_birads_classes)
        self.density_head = nn.Linear(num_features, config.num_density_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        birads_logits = self.birads_head(features)
        density_logits = self.density_head(features)
        return birads_logits, density_logits


class ViTClassifier(nn.Module):
    """Vision Transformer based classifier for mammography."""
    def __init__(self, config: ViTClassifierConfig):
        super().__init__()
        self.config = config

        vit_config = ViTConfig.from_pretrained(config.model_name)
        self.vit = ViTModel.from_pretrained(config.model_name, config=vit_config, add_pooling_layer=False)

        if not config.pretrained:
             self.vit.init_weights() # Initialize from scratch if not pretrained

        num_features = vit_config.hidden_size

        self.birads_head = nn.Linear(num_features, config.num_birads_classes)
        self.density_head = nn.Linear(num_features, config.num_density_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vit(pixel_values=x)
        # Use the [CLS] token representation for classification
        cls_token_features = outputs.last_hidden_state[:, 0]

        birads_logits = self.birads_head(cls_token_features)
        density_logits = self.density_head(cls_token_features)
        return birads_logits, density_logits


