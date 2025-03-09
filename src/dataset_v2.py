import pandas as pd
import numpy as np
from utils.dicom2png import convert_dicom_to_png, Image
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
        png_converted: bool = False,
    ) -> None:

        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.inter_name = inter_name
        self.img_size = img_size
        self.png_converted = png_converted

        self.birads_idx = {
            v: k for k, v in enumerate(sorted(df.breast_birads.unique()))
        }
        self.density_idx = {
            v: k for k, v in enumerate(sorted(df.breast_density.unique()))
        }
        self.cat2idx = CAT2IDX
        self.idx2cat = IDX2CAT
        self.findings = FINDING_CATEGORIES

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

        if self.png_converted:
            image = self._load_image_png(row)
            target = self._get_annotation_agg(row)
        else:
            image = self._load_image_dicom(row)
            target = self._get_annotations(row)

        return image, target

    def _load_image_png(self, row) -> torch.Tensor:

        png_path = f"{self.inter_name}/{row.study_id}/{row.image_id}.png"
        img = np.array(Image.open(png_path))

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        img = np.stack([img] * 3, axis=-1)

        return self.transform(img)

    def _load_image_dicom(self, row) -> torch.Tensor:

        dicom_path = f"{self.inter_name}/{row.study_id}/{row.image_id}.dicom"
        png_img = convert_dicom_to_png(dicom_path)

        img = np.stack([png_img] * 3, axis=-1).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return self.transform(img)

    def _get_annotation_agg(self, row) -> Dict[str, torch.Tensor]:
        """
        Process aggregated annotations for images with multiple findings.
        Handles entries where xmin, ymin, xmax, ymax, and mapped_category are already in list format.
        Also handles "No Findings" cases where these lists may be empty.
        """
        h_scale = self.img_size[0] / row.height
        w_scale = self.img_size[1] / row.width

        boxes = []
        labels = []

        xmins = ast.literal_eval(row.xmin)
        ymins = ast.literal_eval(row.ymin)
        xmaxs = ast.literal_eval(row.xmax)
        ymaxs = ast.literal_eval(row.ymax)
        categories = ast.literal_eval(row.mapped_category)

        # Ensure all lists have the same length
        assert (
            len(xmins) == len(ymins) == len(xmaxs) == len(ymaxs) == len(categories)
        ), "Length mismatch in annotations"

        for xmin, ymin, xmax, ymax, category in zip(
            xmins, ymins, xmaxs, ymaxs, categories
        ):
            # Check if this is a valid finding (not "No Finding")
            if category != "No Finding":
                # Scale coordinates to match image size
                xmin_scaled = float(xmin) * w_scale
                ymin_scaled = float(ymin) * h_scale
                xmax_scaled = float(xmax) * w_scale
                ymax_scaled = float(ymax) * h_scale

                boxes.append([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled])
                labels.append(self.cat2idx.get(category, len(self.cat2idx) - 1))

        # Branch for "No Findings" or empty lists
        # This branch is left empty as requested, but will be triggered when:
        # - The lists are empty
        # - Or only contain "No Finding" entries
        # In both cases, boxes and labels will remain empty

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

    def _get_annotations(self, row) -> Dict[str, torch.Tensor]:
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
