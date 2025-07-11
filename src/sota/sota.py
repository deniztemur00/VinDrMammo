import torch
from torch import nn
from retinanet import model
from GMIC_adaptation.config import GlobalConfig
import GMIC_adaptation.modules as m
from density_attention import DensityAttention


class SOTA(nn.Module):
    def __init__(self, config: GlobalConfig):
        super(SOTA, self).__init__()
        self.config = config

        # Initialize the detection network
        self.detection_net = model.resnet18(num_classes=3, pretrained=True)

        self.local_network = m.LocalNetwork(self.config, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.config, self)
        self.attention_module.add_layers()

        self.density_net = DensityAttention(
            in_channels=self.config.post_processing_dim, out_features=config.n_density
        )

        # fusion branch
        self.fusion_dnn = nn.Linear(
            768,
            config.n_birads,  ## Chaning this according to my dim
            # change 768 to the correct dimension later
        )

    def forward(self, images, targets=None):

        # Step 1: Run the detection network (global module)
        if self.training and targets is not None:
            detection_loss, detections, features = self.detection_net([images, targets])
        else:
            detections, features = self.detection_net(images)

        # Step 2: Process detections (bounding boxes)
        all_patches = []
        for i, detection in enumerate(detections):
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
        # print(f"h_crops shape: {h_crops.shape}")
        # Step 4: MIL module: Compute attention-weighted features
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)
        # print(
        #    f"z shape: {z.shape}, patch_attns shape: {self.patch_attns.shape}, y_local shape: {self.y_local.shape}"
        # )

        # backbone features shape: (batch_size,256,h,w)
        # hxw = (64, 64) , (32,32), (16,16), (8,8), (4,4)
        last_feature_map = features[-1]

        density_logits = self.density_net(last_feature_map)
        g1, _ = torch.max(last_feature_map, dim=2)
        global_vec, _ = torch.max(g1, dim=2)

        concat_vec = torch.cat([global_vec, z], dim=1)
        # print(f"concat_vec shape: {concat_vec.shape}")
        self.y_fusion_birads = self.fusion_dnn(concat_vec)

        if self.training:
            loss_dict = {
                "birads_logits": self.y_fusion_birads,
                "density_logits": density_logits,
                "finding_loss": detection_loss[0],
                "reg_loss": detection_loss[1],
            }
            return loss_dict
        else:
            inference_results = {
                "detections": detections,
                "birads_logits": self.y_fusion_birads,
                "density_logits": density_logits,
            }
            return inference_results
