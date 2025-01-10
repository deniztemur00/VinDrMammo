import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataclasses import dataclass
from typing import Tuple
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models import ResNet50_Weights

from torchvision.models.detection.image_list import ImageList


@dataclass
class FasterRCNNConfig:
    num_classes: int = 35
    num_birads_classes: int = 5
    num_density_classes: int = 4
    backbone_name: str = "resnet50"  # resnet101
    pretrained_backbone: bool = True
    min_size: int = 800
    max_size: int = 1333
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    anchor_sizes: Tuple[Tuple[int]] = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0),) * 5
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5


class CustomFasterRCNN(nn.Module):
    def __init__(self, config: FasterRCNNConfig):
        super(CustomFasterRCNN, self).__init__()
        # Define the anchor generator with custom sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=config.anchor_sizes, aspect_ratios=config.aspect_ratios
        )

        # Load the backbone
        backbone = resnet_fpn_backbone(
            config.backbone_name, config.pretrained_backbone, weights=ResNet50_Weights
        )

        # Create the Faster R-CNN model with the specified configuration
        self.model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            min_size=config.min_size,
            max_size=config.max_size,
            image_mean=config.image_mean,
            image_std=config.image_std,
            rpn_anchor_generator=anchor_generator,
            rpn_fg_iou_thresh=config.rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=config.rpn_bg_iou_thresh,
            box_fg_iou_thresh=config.box_fg_iou_thresh,
            box_bg_iou_thresh=config.box_bg_iou_thresh,
        )
        self.features = None
        self.model.backbone.register_forward_hook(self.get_features_hook)
        backbone_out_channels = backbone.out_channels

        self.birads_classifier = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config.num_birads_classes),
        )

        self.density_classifier = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, config.num_density_classes),
        )

    def get_features_hook(self, module, input, output):
        self.features = output

    ### BIRADS AND DENSITY CLASSIFICATION
    def forward(self, images, targets=None):
        # return self.model(images, targets)

        if self.training and targets:

            loss_dict = self.model(images, targets)

            feature_map = self.features["0"]
            global_features = (
                nn.AdaptiveAvgPool2d((1, 1))(feature_map).squeeze(-1).squeeze(-1)
            )
            
            birads_logits = self.birads_classifier(global_features)
            density_logits = self.density_classifier(global_features)

            # Collect targets from the list of dictionaries and ensure they are 1D
            birads_targets = (
                torch.stack([t["birads"] for t in targets]).flatten().long()
            )
            density_targets = (
                torch.stack([t["density"] for t in targets]).flatten().long()
            )

            # Compute classification losses
            birads_loss = nn.CrossEntropyLoss()(birads_logits, birads_targets)
            density_loss = nn.CrossEntropyLoss()(density_logits, density_targets)
            # Add classification losses to the total loss dictionary
            loss_dict["birads_loss"] = birads_loss
            loss_dict["density_loss"] = density_loss

            return loss_dict
        else:
            # During inference, perform object detection and classification
            detections = self.model(images)

            # Extract global features
            if self.features is None or "0" not in self.features:
                raise ValueError("Features not collected properly from backbone.")

            feature_map = self.features["0"]
            global_features = (
                nn.AdaptiveAvgPool2d((1, 1))(feature_map).squeeze(-1).squeeze(-1)
            )

            # Classify global features
            birads_logits = self.birads_classifier(global_features)
            density_logits = self.density_classifier(global_features)

            # Compute probabilities
            birads_probs = torch.softmax(birads_logits, dim=-1)
            density_probs = torch.softmax(density_logits, dim=-1)

            return detections, birads_probs, density_probs
