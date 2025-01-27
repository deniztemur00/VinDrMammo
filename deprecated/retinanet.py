from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torch import nn
import torch
from dataclasses import dataclass

from typing import Optional, Tuple
from collections import OrderedDict


@dataclass
class RetinaNetConfig:
    backbone: str = "resnet101"
    trainable_backbone_layers: int = 5
    num_classes: int = 35
    num_birads_classes: int = 5
    num_density_classes: int = 4
    detections_per_img: int = 1
    top_k_candidates: int = 1
    nms_thresh = 0.3
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class CustomRetinaNet(nn.Module):
    def __init__(self, config: RetinaNetConfig):
        super().__init__()

        self.backbone = resnet_fpn_backbone(
            backbone_name=config.backbone,
            weights=ResNet101_Weights,
            trainable_layers=config.trainable_backbone_layers,
        )

        self.model = RetinaNet(
            backbone=self.backbone,
            num_classes=config.num_classes,
            detections_per_img=config.detections_per_img,
            top_k_candidates=config.top_k_candidates,
            nms_thresh=config.nms_thresh,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )

        self.birads_classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, config.num_birads_classes),
        )

        self.density_classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, config.num_density_classes),
        )

        self.birads_loss = nn.CrossEntropyLoss()
        self.density_loss = nn.CrossEntropyLoss()

        self.backbone_features: Optional[OrderedDict] = None
        self.hook_handles = []

        self.model.backbone.register_forward_hook(self._get_backbone_features)

    def _get_backbone_features(self, module, input, output):
        self.backbone_features = output["3"]  # before pooling shape: 256,25,25

    def get_features(self):
        if self.backbone_features is None:
            raise ValueError("No features available.")
        return self.backbone_features

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            losses = self.model(images, targets)

            
            birads_logits = self.birads_classifier(self.backbone_features)
            density_logits = self.density_classifier(self.backbone_features)

            density_targets = torch.stack([t["density"] for t in targets]).flatten()
            birads_targets = torch.stack([t["birads"] for t in targets]).flatten()

            birads_loss = self.birads_loss(birads_logits, birads_targets) * 0.5
            density_loss = self.density_loss(density_logits, density_targets) * 0.3

            losses.update({"birads_loss": birads_loss, "density_loss": density_loss})

            return losses
        else:
            detections = self.model(images)
            birads_logits = self.birads_classifier(self.backbone_features)
            density_logits = self.density_classifier(self.backbone_features)

            return detections, birads_logits, density_logits
