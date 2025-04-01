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
    num_birads_classes: int = (
        5  # BI-RADS 1-5num_density_classes: int = 4  # Density A-D
    )
    detections_per_img: int = 10
    top_k_candidates: int = 100
    nms_thresh: float = 0.5
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # anchor_sizes = (
    #    (16, 32, 64),
    #    (32, 64, 128),
    #    (64, 128, 256),
    #    (128, 256, 512),
    #    (256, 512, 1024),
    # )
    anchor_sizes = (
        (32.4, 64.9, 106.9),
        (64.9, 106.9, 168.3),
        (106.9, 168.3, 269.6),
        (168.3, 269.6, 400.0),
        (269.6, 400.0, 600.0),
    )
    aspect_ratios = ((0.67, 1.09, 1.57),) * 5
    # aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    birads_loss_weight = 0.7  # Weight for BI-RADS/density losses
    density_loss_weight = 0.3


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.sigmoid(attn)
        return x * attn


class BIRADSHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.AdaptiveMaxPool2d(1),  # [B, 64, 1, 1]
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DensityHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Hardswish(inplace=True),
            nn.AdaptiveMaxPool2d(1),  # [B, 64, 1, 1]
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomRetinaNet(nn.Module):
    def __init__(self, config: RetinaNetConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.spatial_attention = SpatialAttention(in_channels=256)

        self.birads_head = BIRADSHead(num_classes=config.num_birads_classes)
        self.density_head = DensityHead(num_classes=config.num_birads_classes)
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

    def visualize_attention(self, image: torch.Tensor):
        # Forward pass
        with torch.no_grad():
            self.eval()
            _ = self([image.to(self.device)])
            features = self.feature_maps["pool"]
            attn = self.spatial_attention(features)

        # Plot
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.cpu().permute(1, 2, 0))
        ax[1].imshow(attn[0].mean(dim=0).cpu(), cmap="hot")
        plt.show()

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
            features = self.feature_maps["pool"]
            features = self.spatial_attention(features)

            # Verify feature map dimensions
            if features is None:
                raise RuntimeError("Feature maps not captured from backbone")

        else:
            # Inference mode
            with torch.no_grad():
                detections = self.detector(images)
                features = self.feature_maps["pool"]
                features = self.spatial_attention(features)

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
            total_loss = (
                sum(detector_losses.values())
                + self.config.birads_loss_weight * birads_loss
                + self.config.density_loss_weight * density_loss
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
