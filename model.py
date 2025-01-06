import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataclasses import dataclass
from typing import Tuple
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator


@dataclass
class FasterRCNNConfig:
    num_classes: int = 35
    backbone_name: str = "resnet50" # resnet101
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
        backbone = resnet_fpn_backbone(config.backbone_name, config.pretrained_backbone)

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

    ### BIRADS AND DENSITY CLASSIFICATION
    def forward(self, images, targets=None):
        return self.model(images, targets)


"""
class CustomFasterRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load the pretrained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Replace the classifier with a new one for your num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Optional: Freeze backbone layers
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, images, targets=None):
        # During training, targets should be passed
        # During inference, targets should be None
        return self.model(images, targets)

"""
