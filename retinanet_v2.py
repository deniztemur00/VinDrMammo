from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch import nn
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


from matplotlib import pyplot as plt


@dataclass
class RetinaNetConfig:
    backbone: str = "resnet101"
    trainable_backbone_layers: int = 5
    num_classes: int = 11  # 10 findings + "Other"
    num_birads_classes: int = 5  # BI-RADS 1-5
    num_density_classes: int = 4  # Density A-D
    detections_per_img: int = 10
    top_k_candidates: int = 10
    nms_thresh: float = 0.3
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    anchor_sizes = (
        (16, 32, 64),
        (32, 64, 128),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    aux_loss_weight: float = 0.2  # Weight for BI-RADS/density losses


class CustomRetinaNet(nn.Module):
    def __init__(self, config: RetinaNetConfig):
        super().__init__()
        self.config = config

        # Backbone with FPN
        self.backbone = resnet_fpn_backbone(
            backbone_name=config.backbone,
            weights=ResNet101_Weights.IMAGENET1K_V1,
            trainable_layers=config.trainable_backbone_layers,
        )

        anchor_generator = AnchorGenerator(
            sizes=config.anchor_sizes,
            aspect_ratios=config.aspect_ratios,
        )

        # Main RetinaNet model
        self.detector = RetinaNet(
            backbone=self.backbone,
            num_classes=config.num_classes,
            detections_per_img=config.detections_per_img,
            top_k_candidates=config.top_k_candidates,
            anchor_generator=anchor_generator,
            nms_thresh=config.nms_thresh,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )

        # Multi-task heads
        self.birads_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_birads_classes),
        )

        self.density_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_density_classes),
        )

        # Loss functions
        self.birads_loss_fn = nn.CrossEntropyLoss()
        self.density_loss_fn = nn.CrossEntropyLoss()

        # Feature hooks
        self.feature_maps = None
        self.detector.backbone.register_forward_hook(self._get_backbone_features)

    def _get_backbone_features(self, module, input, output):
        self.feature_maps = output  # before pooling shape: 256,25,25

    def get_features(self):
        if self.feature_maps is None:
            raise ValueError("No features available.")
        return self.feature_maps

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):

        self.feature_maps = None
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training")

            # Forward through detector (triggers feature hook)
            detector_losses = self.detector(images, targets)
            features = self.feature_maps["3"]

            # Verify feature map dimensions
            if features is None:
                raise RuntimeError("Feature maps not captured from backbone")

        else:
            # Inference mode
            with torch.no_grad():
                detections = self.detector(images)
                features = self.feature_maps["3"]

        # Multi-task predictions
        birads_logits = self.birads_head(features)
        density_logits = self.density_head(features)

        if self.training:
            # Calculate auxiliary losses
            birads_targets = torch.cat([t["birads"] for t in targets])
            density_targets = torch.cat([t["density"] for t in targets])

            birads_loss = self.birads_loss_fn(birads_logits, birads_targets)
            density_loss = self.density_loss_fn(density_logits, density_targets)

            # Combine losses
            total_loss = sum(detector_losses.values()) + self.config.aux_loss_weight * (
                birads_loss + density_loss
            )

            return {
                "classification": detector_losses["classification"],
                "box_reg": detector_losses["bbox_regression"],
                "birads_loss": birads_loss,
                "density_loss": density_loss,
                "total_loss": total_loss,
            }
        else:
            return {
                "detections": detections,
                "birads_logits": birads_logits,
                "density_logits": density_logits,
                "features": features,
            }
