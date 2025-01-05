import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class FasterRCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 3
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
