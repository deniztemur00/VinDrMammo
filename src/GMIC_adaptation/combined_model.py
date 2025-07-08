import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import List, Dict, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.sota.config import GMICConfig
import modules as m
from detection.retinanet_v2 import CustomRetinaNet, RetinaNetConfig


class CombinedDetectorGMIC(nn.Module):
    def __init__(self, detection_config: RetinaNetConfig, gmic_config: GMICConfig):
        super().__init__()
        self.detection_net = CustomRetinaNet(detection_config)
        self.gmic_config = gmic_config

        # Re-use GMIC's local network and attention module
        self.local_network = m.LocalNetwork(gmic_config, self)
        self.local_network.add_layers()
        self.attention_module = m.AttentionModule(gmic_config, self)
        self.attention_module.add_layers()

        fusion_input_dim = 256 + gmic_config.local_hidden_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, gmic_config.num_classes),
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):

        if self.training and targets is not None:
            
            detector_output = self.detection_net(images, targets)
            global_features = self.detection_net.feature_maps["pool"]
            
            boxes_list = [t["boxes"] for t in targets]
        else:
            
            detector_output = self.detection_net(images)
            global_features = detector_output["features"]
            boxes_list = [d["boxes"] for d in detector_output["detections"]]

        
        all_patches = []
        for i, image in enumerate(images):
            boxes = boxes_list[i]
            if boxes.shape[0] > 0:
                for box in boxes:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    print("box coordinates:", x1, y1, x2, y2)
                    patch = TF.crop(image, y1, x1, y2 - y1, x2 - x1)
                    x_crop, y_crop = self.gmic_config.crop_shape
                    patch = TF.resize(
                        patch, [x_crop, y_crop]
                    )  # Resize to the expected crop size
                    all_patches.append(patch)

        if not all_patches:
           
            print("no boxes found")
            return detector_output

        patches_tensor = torch.stack(all_patches)

        
        h_crops = self.local_network.forward(patches_tensor)
        
        h_crops = h_crops.unsqueeze(0)  # pseudo-batch for the attention module

        
        z, _, _ = self.attention_module.forward(h_crops)

        global_vec = global_features.mean(dim=[2, 3])  # Global Average Pooling

        # Concatenate and classify
        fused_features = torch.cat([global_vec, z], dim=1)
        final_logits = self.fusion_classifier(fused_features)

        # 6. Combine outputs
        if self.training:
            detector_output["fusion_logits"] = final_logits
            return detector_output
        else:
            detector_output["fusion_logits"] = final_logits
            return detector_output


def main():
    detection_config = RetinaNetConfig(num_classes=5)
    gmic_config = GMICConfig(num_classes=5, local_hidden_dim=512, crop_shape=(224, 224))

    combined_model = CombinedDetectorGMIC(detection_config, gmic_config)
    combined_model.eval()
    # Dummy input
    images = [torch.randn(3, 512, 512) for _ in range(2)]  # Two images of size 512x512
    targets = None  # No targets for inference
    with torch.no_grad():
        output = combined_model(images, targets)
    print(output["fusion_logits"]) 


if __name__ == "__main__":
    main()