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
        self.detection_net = model.resnet18(
            num_classes=config.n_findings, pretrained=True
        )
        self.freeze_detection_backbone()
        self.local_network = m.LocalNetwork(self.config, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = m.AttentionModule(self.config, self)
        self.attention_module.add_layers()
        self.feature_dropout = nn.Dropout2d(p=0.2)

        # Density Classification branch
        self.density_net = DensityAttention(
            in_channels=self.config.post_processing_dim, out_features=config.n_density
        )

        # fusion branch
        self.dropout = nn.Dropout(p=0.2)
        self.bn_fusion = nn.BatchNorm1d(768)
        self.fusion_dnn = nn.Linear(
            768,
            config.n_birads,  ## Chaning this according to my dim
            # change 768 to the correct dimension later
        )

    def freeze_detection_backbone(self):
        for name, parameter in self.detection_net.named_parameters():
            if not (
                "fpn" in name
                or "regressionModel" in name
                or "classificationModel" in name
            ):
                parameter.requires_grad = False

    def forward(self, images, targets=None):

        # Step 1: Run the detection network (global module)
        if self.training and targets is not None:
            detection_loss, detections, features = self.detection_net([images, targets])
        else:
            detections, features = self.detection_net(images)

        # Step 2: Process detections (bounding boxes)
        all_patches = []
        patches_per_image = []  # Keep track of how many patches belong to each image
        for i, detection_result in enumerate(detections):
            bboxes = detection_result[2]
            num_patches = 0
            for bbox in bboxes:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                if y2 > y1 and x2 > x1:  # Ensure patch has a non-zero area
                    patch = images[i, :, y1:y2, x1:x2]
                    patch = torch.nn.functional.interpolate(
                        patch.unsqueeze(0), size=self.config.crop_shape, mode="bilinear"
                    ).squeeze(0)
                    all_patches.append(patch)
                    num_patches += 1
            patches_per_image.append(num_patches)

        if not all_patches:
            print("No bounding boxes found")
            return None

        else:
            # last_feature_map = features[-1]
            # feature map sizes :
            last_feature_map = self.feature_dropout(features[-1])
            density_logits = self.density_net(last_feature_map)
            patches_tensor = torch.stack(
                all_patches
            )  # Combine all patches into a batch

            # Step 3: Local network: Compute hidden representations for patches
            h_crops = self.local_network.forward(patches_tensor)
            h_crops_grouped = torch.split(h_crops, patches_per_image, dim=0)

            # Pad the sequences so they all have the same length (the max number of patches in the batch).
            h_crops_padded = nn.utils.rnn.pad_sequence(
                h_crops_grouped, batch_first=True
            )

            # Now the MIL module can process the padded batch.
            z, self.patch_attns, self.y_local = self.attention_module.forward(
                h_crops_padded
            )

            # Fuse global and local features for final BI-RADS prediction.
            g1, _ = torch.max(last_feature_map, dim=3)
            global_vec, _ = torch.max(g1, dim=2)

            concat_vec = torch.cat([global_vec, z], dim=1)
            concat_vec = self.dropout(concat_vec)
            concat_vec = self.bn_fusion(concat_vec)
            birads_logits = self.fusion_dnn(concat_vec)

        if self.training:
            loss_dict = {
                "birads_logits": birads_logits,
                "density_logits": density_logits,
                "finding_loss": detection_loss[0],
                "reg_loss": detection_loss[1],
            }
            return loss_dict
        else:
            inference_results = {
                "detections": detections,
                "birads_logits": nn.functional.softmax(birads_logits, dim=1),
                "density_logits": nn.functional.softmax(density_logits, dim=1),
            }
            return inference_results
