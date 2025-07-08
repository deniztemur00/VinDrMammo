import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3
from GMIC_adaptation.config import GMICConfig


class BasicBlockV2(nn.Module):
    """
    Basic Residual Block of ResNet V2
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class BasicBlockV1(nn.Module):
    """
    Basic Residual Block of ResNet V1
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """

    def __init__(self, config: GMICConfig, block_fn):
        super(ResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.num_filters,
            kernel_size=config.first_layer_kernel_size,
            stride=config.first_layer_conv_stride,
            padding=config.first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=config.first_pool_size,
            stride=config.first_pool_stride,
            padding=config.first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = config.num_filters
        self.inplanes = config.num_filters
        for i, (num_blocks, stride) in enumerate(
            zip(config.blocks_per_layer_list, config.block_strides_list)
        ):
            self.layer_list.append(
                self._make_layer(
                    block=block_fn,
                    planes=current_num_filters,
                    blocks=num_blocks,
                    stride=stride,
                )
            )
            current_num_filters *= config.growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // config.growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = config.num_filters
        self.growth_factor = config.growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
        )

        layers_ = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


class ResNetV1(nn.Module):
    """
    Class that represents a ResNet with classifier sequence removed
    """

    def __init__(self, config: GMICConfig, block):

        self.inplanes = config.initial_filters_v1
        self.num_layers = len(config.layers_v1)
        super(ResNetV1, self).__init__()

        # initial sequence
        # the first sequence only has 1 input channel which is different from original ResNet
        self.conv1 = nn.Conv2d(
            config.input_channels_v1,
            config.initial_filters_v1,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.initial_filters_v1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual sequence
        for i in range(self.num_layers):
            num_filters = config.initial_filters_v1 * pow(2, i)
            num_stride = 1 if i == 0 else 2
            setattr(
                self,
                "layer{0}".format(i + 1),
                self._make_layer(
                    block, num_filters, config.layers_v1[i], stride=num_stride
                ),
            )
        self.num_filter_last_seq = config.initial_filters_v1 * pow(
            2, self.num_layers - 1
        )

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # first sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual sequences
        for i in range(self.num_layers):
            x = getattr(self, "layer{0}".format(i + 1))(x)
        return x


class DownsampleNetworkResNet18V1(ResNetV1):
    """
    Downsampling using ResNet V1
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution
    """

    def __init__(self, config: GMICConfig):
        super(DownsampleNetworkResNet18V1, self).__init__(
            config=config, block=BasicBlockV1
        )

    def forward(self, x):
        last_feature_map = super(DownsampleNetworkResNet18V1, self).forward(x)
        return last_feature_map


class AbstractMILUnit:
    """
    An abstract class that represents an MIL unit module
    """

    def __init__(self, config: GMICConfig, parent_module):
        self.config = config
        self.parent_module: nn.Module = parent_module


class PostProcessingStandard(nn.Module):
    """
    Unit in Global Network that takes in x_out and produce saliency maps
    """

    def __init__(self, config: GMICConfig):
        super(PostProcessingStandard, self).__init__()
        # map all filters to output classes
        self.gn_conv_last = nn.Conv2d(
            config.post_processing_dim, config.num_classes, (1, 1), bias=False
        )

    def forward(self, x_out):
        out = self.gn_conv_last(x_out)
        return torch.sigmoid(out)


class GlobalNetwork(AbstractMILUnit):
    """
    Implementation of Global Network using ResNet-22
    """

    def __init__(self, config: GMICConfig, parent_module):
        super(GlobalNetwork, self).__init__(config=config, parent_module=parent_module)
        # downsampling-branch
        if config.use_v1_global:
            self.downsampling_branch = DownsampleNetworkResNet18V1(config)
        else:
            self.downsampling_branch = ResNetV2(config=config, block_fn=BasicBlockV2)
        # post-processing
        self.postprocess_module = PostProcessingStandard(config)

    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.postprocess_module

    def forward(self, x):
        # retrieve results from downsampling network at all 4 levels
        last_feature_map = self.downsampling_branch.forward(x)
        # feed into postprocessing network
        cam = self.postprocess_module.forward(last_feature_map)
        print(f"cam shape: {cam.shape}, last_feature_map shape: {last_feature_map.shape}")
        return last_feature_map, cam
