from dataclasses import dataclass


from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GMICConfig:
    """
    Configuration for the GMIC model and its components.
    """

    # General
    num_classes: int = 2
    input_channels: int = 1

    # ResNetV2 & GlobalNetwork
    use_v1_global: bool = False
    num_filters: int = 16
    first_layer_kernel_size: Tuple[int, int] = (7, 7)
    first_layer_conv_stride: int = 2
    first_layer_padding: int = 3
    first_pool_size: int = 3
    first_pool_stride: int = 2
    first_pool_padding: int = 0
    blocks_per_layer_list: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    block_strides_list: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2])
    growth_factor: int = 2

    # ResNetV1 & DownsampleNetworkResNet18V1
    initial_filters_v1: int = 64
    layers_v1: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    input_channels_v1: int = 3

    # PostProcessingStandard
    post_processing_dim: int = 512

    # TopTPercentAggregationFunction
    percent_t: float = 0.1

    # RetrieveROIModule
    K: int = 5  # num_crops_per_class
    crop_shape: Tuple[int, int] = (224, 224)
    device_type: str = "gpu"
    gpu_number: int = 0

    # LocalNetwork
    # Uses ResNetV1, parameters are above

    # AttentionModule
    local_hidden_dim: int = 512  # output of LocalNetwork's ResNet

    device_type = None
    gpu_number = None
    max_crop_noise: Tuple[int, int] = (100, 100)
    max_crop_size_noise: int = 100
    image_path = None
    segmentation_path = None
    output_path = None
    cam_size: Tuple[int, int] = (14, 11) # 46x30 for 2944x1920
    K: int = 6
    crop_shape: Tuple[int, int] = (256, 256)
    post_processing_dim: int = 256
    num_classes: int = 2
    use_v1_global: bool = False
