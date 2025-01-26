import pandas as pd
import numpy as np
from utils.visualize import convert_dicom_to_png
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from typing import Dict, List, Tuple
import ast

# Simplified class mapping based on Table 4
FINDING_CATEGORIES = [
    "Mass",
    "Suspicious Calcification",
    "Asymmetry",
    "Focal Asymmetry",
    "Global Asymmetry",
    "Architectural Distortion",
    "Skin Thickening",
    "Skin Retraction",
    "Nipple Retraction",
    "Suspicious Lymph Node",
    "Other"
]
CAT2IDX = {cat: idx for idx, cat in enumerate(FINDING_CATEGORIES)}


class MammographyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str,
        inter_name: str,
        img_size: Tuple[int, int] = (800, 800),
    ) -> None:
        # Group all annotations per image
        self.image_groups = df.groupby("image_id")
        self.image_ids = list(self.image_groups.groups.keys())

        self.zip_path = zip_path
        self.inter_name = inter_name
        self.img_size = img_size

        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> torch.Tensor:
        # Get first row for image metadata
        sample_row = self.image_groups.get_group(image_id).iloc[0]

        # Load DICOM (replace this with your actual DICOM loading logic)
        dicom_path = f"{self.inter_name}/{sample_row.study_id}/{image_id}.dicom"
        png_img = convert_dicom_to_png(dicom_path)

        # Convert to 3-channel and normalize
        img = np.stack([png_img] * 3, axis=-1).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return self.transform(img)

    def _get_annotations(self, image_id: str) -> Dict[str, torch.Tensor]:
        group = self.image_groups.get_group(image_id)
        original_width = group.iloc[0].width
        original_height = group.iloc[0].height

        # Scale factors
        w_scale = self.img_size[1] / original_width  # width in torch is dim 1
        h_scale = self.img_size[0] / original_height  # height is dim 0

        boxes = []
        labels = []

        for _, row in group.iterrows():
            if pd.isna(row["xmin"]):
                continue  # Skip invalid annotations

            # Scale coordinates
            xmin = row["xmin"] * w_scale
            ymin = row["ymin"] * h_scale
            xmax = row["xmax"] * w_scale
            ymax = row["ymax"] * h_scale

            # Handle multi-label findings (take first category)
            finding = ast.literal_eval(row["mapping_category"])[0]
            if finding not in CAT2IDX:
                continue  # Skip undefined categories

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CAT2IDX[finding])

        if len(boxes) == 0:  # No findings
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        target = self._get_annotations(image_id)
        return image, target


def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
