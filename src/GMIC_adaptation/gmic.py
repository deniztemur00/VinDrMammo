import torch
import torch.nn as nn
import numpy as np
import tools
import modules as m
from global_net import GlobalNetwork
from config import GMICConfig


class GMIC(nn.Module):
    def __init__(self, config: GMICConfig):
        super(GMIC, self).__init__()

        # save config
        self.config = config
        self.cam_size = config.cam_size

        # construct networks
        # global network
        self.global_network = GlobalNetwork(config=config, parent_module=self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = m.TopTPercentAggregationFunction(self.config, self)

        # detection module
        self.retrieve_roi_crops = m.RetrieveROIModule(self.config, self)

        # detection network
        self.local_network = m.LocalNetwork(self.config, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.config, self)
        self.attention_module.add_layers()

        # fusion branch
        self.fusion_dnn = nn.Linear(
            config.post_processing_dim + 512, config.num_classes
        )

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate(
            [top_k_interpolate_x, top_k_interpolate_y], axis=-1
        )
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.config.crop_shape

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.config.device_type == "gpu":
            device = torch.device("cuda:{}".format(self.config.gpu_number))
            output = output.cuda().to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(
                    x_original_pytorch[i, 0, :, :],
                    self.config.crop_shape,
                    crop_positions[i, j, :],
                    output[i, j, :, :],
                    method=crop_method,
                )
        return output

    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(
            x_original, self.cam_size, self.saliency_map
        )

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(
            small_x_locations, self.cam_size, x_original
        )

        # patch retriever
        crops_variable = self._retrieve_crop(
            x_original, self.patch_locations, self.retrieve_roi_crops.crop_method
        )
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(
            batch_size, num_crops, -1
        )

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        return self.y_fusion

## SUCCESFUL EXECUTION ON LOCAL MACHINE
def main():
    # Example usage
    config = GMICConfig()
    model = GMIC(config=config)
    x_original = torch.randn(2, 1, 890, 650)  # Example input
    output = model(x_original)
    print(output.shape)
    print(output)


if __name__ == "__main__":
    main()
