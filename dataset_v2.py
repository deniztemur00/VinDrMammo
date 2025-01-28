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
    "Other",
]
CAT2IDX = {cat: idx for idx, cat in enumerate(FINDING_CATEGORIES)}

IDX2CAT = {idx: cat for cat, idx in CAT2IDX.items()}


class MammographyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str,
        inter_name: str,
        img_size: Tuple[int, int] = (800, 800),
    ) -> None:
        # Use individual rows instead of grouping
        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.inter_name = inter_name
        self.img_size = img_size

        # Create mappings
        self.birads_idx = {
            v: k for k, v in enumerate(sorted(df.breast_birads.unique()))
        }
        self.density_idx = {
            v: k for k, v in enumerate(sorted(df.breast_density.unique()))
        }
        self.cat2idx = CAT2IDX
        self.idx2cat = IDX2CAT

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(img_size),
                # transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(num_output_channels=3),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        row = self.df.iloc[idx]

        # Load image
        image = self._load_image(row)

        # Get annotations
        target = self._get_annotations(row)

        return image, target

    def _load_image(self, row) -> torch.Tensor:
        # Your DICOM loading logic here
        dicom_path = f"{self.inter_name}/{row.study_id}/{row.image_id}.dicom"
        png_img = convert_dicom_to_png(dicom_path)

        # Convert to 3-channel and normalize
        img = np.stack([png_img] * 3, axis=-1).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return self.transform(img)

    def _get_annotations(self, row) -> Dict[str, torch.Tensor]:
        # Scale coordinates
        w_scale = self.img_size[1] / row.width
        h_scale = self.img_size[0] / row.height

        boxes = []
        labels = []

        finding = row.mapped_category.strip()
        if finding != "No Finding" and not pd.isna(row.xmin):
            xmin = row.xmin * w_scale
            ymin = row.ymin * h_scale
            xmax = row.xmax * w_scale
            ymax = row.ymax * h_scale
            boxes.append([xmin, ymin, xmax, ymax])

            # Now we know it's a valid finding, so we can add the label
            labels.append(self.cat2idx.get(finding, len(self.cat2idx) - 1))

        return {
            "boxes": (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": (
                torch.tensor(labels, dtype=torch.int64)
                if labels
                else torch.zeros((0,), dtype=torch.int64)
            ),
            "birads": torch.tensor(
                [self.birads_idx[row.breast_birads]], dtype=torch.long
            ),
            "density": torch.tensor(
                [self.density_idx[row.breast_density]], dtype=torch.long
            ),
        }


def collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(
            {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "birads": target["birads"],
                "density": target["density"],
            }
        )

    # Stack images into tensor [B, C, H, W]
    images = torch.stack(images, dim=0)

    return images, targets


def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def create_categories(df: pd.DataFrame) -> tuple[List[str], Dict[str, int]]:
    all_categories = []
    for cats_str in df.mapped_category.value_counts().index:
        cats = ast.literal_eval(cats_str)
        if len(cats) > 1:
            all_categories.append("-".join(cats))
        else:
            all_categories.append(cats[0])
    cat2idx = {cat: idx for idx, cat in enumerate(all_categories)}
    return all_categories, cat2idx
