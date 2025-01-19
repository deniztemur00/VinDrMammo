from torchvision.models.detection import retinanet_resnet50_fpn
from torch import nn
import torch
from dataclasses import dataclass
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

@dataclass
class RetinaNetConfig:
    num_classes: int = 35
    num_birads_classes: int = 5
    num_density_classes: int = 4
    pretrained: bool = True


class CustomRetinaNet(nn.Module):
    def __init__(self, config: RetinaNetConfig):
        super().__init__()

        self.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT,
                                            num_classes=config.num_classes,
                                            trainable_backbone_layers=5,
                                            )

    def forward(self, x):
        return self.model(x)