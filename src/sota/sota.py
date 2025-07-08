import torch
from torch import nn
from retinanet import model
from GMIC_adaptation.config import GMICConfig
import GMIC_adaptation.modules as m


class SOTA(nn.Module):
    def __init__(self, config: GMICConfig):
        super(SOTA, self).__init__()
        self.config = config

        # Initialize the detection network
        self.detection_net = model.resnet18(num_classes=3, pretrained=True)

        self.local_network = m.LocalNetwork(self.config, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.config, self)
        self.attention_module.add_layers()

        # fusion branch
        self.fusion_dnn = nn.Linear(
            515, config.num_classes  ## Chaning this according to my dim
        )

    def forward(self, images, targets=None):
        """
        :param images: Input images (N, C, H, W) PyTorch tensor
        :param targets: Optional targets for training
        :return: Final predictions from the fusion module
        """
        # Step 1: Run the detection network (global module)
        if self.training and targets is not None:
            detection_loss, detections = self.detection_net([images, targets])
        else:
            detections = self.detection_net(images)

        # Step 2: Process detections (bounding boxes)
        all_patches = []
        for i, detection in enumerate(
            detections
        ):  # Iterate over detections for each image
            class_scores, class_indices, bboxes = detection
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                patch = images[i, :, y1:y2, x1:x2]  # Crop region
                patch = torch.nn.functional.interpolate(
                    patch.unsqueeze(0), size=self.config.crop_shape, mode="bilinear"
                ).squeeze(0)
                all_patches.append(patch)

        if not all_patches:
            print("No bounding boxes found")
            return None

        patches_tensor = torch.stack(all_patches)  # Combine all patches into a batch

        # Step 3: Local network: Compute hidden representations for patches
        h_crops = self.local_network.forward(patches_tensor)
        h_crops = h_crops.view(
            len(detections), -1, h_crops.size(-1)
        )  # Reshape for MIL module

        # Step 4: MIL module: Compute attention-weighted features
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        print(
            f"z shape: {z.shape}, patch_attns shape: {self.patch_attns.shape}, y_local shape: {self.y_local.shape}"
        )
        # Step 5: Fusion branch: Combine global and local features
        global_vec = torch.mean(images, dim=[2, 3])
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))

        if self.training and targets is not None:
            return detection_loss, self.y_fusion
        else:
            return self.y_fusion
