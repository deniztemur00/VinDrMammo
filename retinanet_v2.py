from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torch import nn
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


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
        (
            (16, 32, 64),
            (32, 64, 128),
            (64, 128, 256),
            (128, 256, 512),
            (256, 512, 1024),
        ),
    )
    aspect_ratios = ((0.7, 1.0, 1.5),) * 5
    aux_loss_weight: float = 0.5  # Weight for BI-RADS/density losses


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
        self._register_hooks()

    def _register_hooks(self):
        def feature_hook(module, input, output):
            self.feature_maps = output["3"]  # P3 features from FPN

        self.backbone.register_forward_hook(feature_hook)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        # Main detection forward pass
        if self.training and targets is not None:
            detector_losses = self.detector(images, targets)
        else:
            detector_losses = None

        # Get shared features from backbone
        _ = self.backbone(images.tensors if hasattr(images, "tensors") else images)
        features = self.feature_maps

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
                "total_loss": total_loss,
                "detection_loss": sum(detector_losses.values()),
                "birads_loss": birads_loss,
                "density_loss": density_loss,
            }
        else:
            # Return predictions during inference
            detections = self.detector(images) if detector_losses is None else None
            return {
                "detections": detections,
                "birads_preds": torch.softmax(birads_logits, dim=1),
                "density_preds": torch.softmax(density_logits, dim=1),
            }
