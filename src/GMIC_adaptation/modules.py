import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import conv3x3

from GMIC_adaptation.config import GMICConfig


from GMIC_adaptation import tools
from GMIC_adaptation.global_net import AbstractMILUnit, ResNetV1, BasicBlockV1


class TopTPercentAggregationFunction(AbstractMILUnit):
    """
    An aggregator that uses the SM to compute the y_global.
    Use the sum of topK value
    """

    def __init__(self, config: GMICConfig, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(config, parent_module)
        self.percent_t = config.percent_t
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W * H * self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)


class RetrieveROIModule(AbstractMILUnit):
    """
    A Regional Proposal Network instance that computes the locations of the crops
    Greedy select crops with largest sums
    """

    def __init__(self, config: GMICConfig, parent_module):
        super(RetrieveROIModule, self).__init__(config, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = config.K
        self.crop_shape = config.crop_shape
        self.gpu_number = None if config.device_type != "gpu" else config.gpu_number

    def forward(self, x_original, cam_size, h_small):
        """
        Function that use the low-res image to determine the position of the high-res crops
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w)
        :param h_small: N, C, h_h, w_h pytorch tensor
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve parameters
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()
        print(
            f"h_h={h_h}, h={h}, w_h={w_h}, w={w}"
        )  # Remove this after tuning for you dataset
        
        # make sure that the size of h_small == size of cam_size
        # assert h_h == h, "h_h!=h" # will use different size for my dataset
        # assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # combine channels
        max_vals = (
            current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        )
        min_vals = (
            current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        )
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = tools.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = tools.generate_mask_uplft(
                current_images, crop_shape_adjusted, max_pos, self.gpu_number
            )
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


class LocalNetwork(AbstractMILUnit):
    """
    The local network that takes a crop and computes its hidden representation
    Use ResNet
    """

    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        self.parent_module.dn_resnet = ResNetV1(self.config, BasicBlockV1)

    def forward(self, x_crop):
        """
        Function that takes in a single crop and return the hidden representation
        :param x_crop: (N,C,h,w)
        :return:
        """
        # forward propagte using ResNet
        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1, -1))
        # global average pooling
        res = res.mean(dim=2).mean(dim=2)
        return res


class AttentionModule(AbstractMILUnit):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """

    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        # The gated attention mechanism
        self.parent_module.mil_attn_V = nn.Linear(
            self.config.local_hidden_dim, 128, bias=False
        )
        self.parent_module.mil_attn_U = nn.Linear(
            self.config.local_hidden_dim, 128, bias=False
        )
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)
        # classifier
        self.parent_module.classifier_linear = nn.Linear(
            self.config.local_hidden_dim, self.config.num_classes, bias=False
        )

    def forward(self, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(
            self.parent_module.mil_attn_U(h_crops_reshape)
        ) * torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))
        attn_score = self.parent_module.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        y_crops = torch.sigmoid(self.parent_module.classifier_linear(z_weighted_avg))
        return z_weighted_avg, attn, y_crops
