import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import ast
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # Import cv2 for border_mode constant

FINDING_CATEGORIES_TOP3 = [
    "Mass",
    "Suspicious Calcification",
    "Focal Asymmetry",
]


class SOTADataset(Dataset):
    """
    Dataset for classifying breast BI-RADS and density using PNG images.
    Applies augmentations only to rows marked as duplicated.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,  # Directory containing the PNG images
        img_size: Tuple[int, int] = (224, 224),  # 1024x512
        augment_duplicated: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment_duplicated = augment_duplicated
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Create label mappings
        self.birads_map = {
            label: idx for idx, label in enumerate(sorted(df["breast_birads"].unique()))
        }
        self.density_map = {
            label: idx
            for idx, label in enumerate(sorted(df["breast_density"].unique()))
        }
        self.cat2idx = {cat: idx for idx, cat in enumerate(FINDING_CATEGORIES_TOP3)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}

        self.num_birads_classes = len(self.birads_map)
        self.num_density_classes = len(self.density_map)

        self.bbox_params = A.BboxParams(
            format="pascal_voc",  # [xmin, ymin, xmax, ymax]
            label_fields=["class_labels"],
            min_area=1,
            min_visibility=0,
            clip=True,  # Clip boxes to stay within image boundaries
        )

        # Basic transforms (applied to all images)
        self.base_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            seed=42,
            strict=True,
            bbox_params=self.bbox_params,
        )

        # Augmentations using Albumentations
        self.augment_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                # --- Augmentations Start ---
                A.HorizontalFlip(p=0.7),  # Adjusted p=0.5
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                # A.ShiftScaleRotate(
                #    shift_limit=0.05,
                #    scale_limit=0.01,
                #    rotate_limit=10,
                #    p=0.5,
                # ),
                ## --- Augmentations End ---
                A.Normalize(
                    mean=self.mean, std=self.std, max_pixel_value=255
                ),  # Adjust mean/std if single channel
                ToTensorV2(),
            ],
            seed=42,
            strict=True,
            bbox_params=self.bbox_params,
        )

    def unnormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Unnormalize the tensor to convert it back to an image. For plotting purposes.
        Args:
            tensor (torch.Tensor): The input tensor to unnormalize.
        Returns:
            np.ndarray: The unnormalized image as a NumPy array.
        """
        # Convert tensor to numpy array and move to CPU
        tensor = tensor.clone().detach().cpu()
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)

        tensor.mul_(std).add_(mean)
        tensor = torch.clamp(tensor, 0, 1)
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        return img_np
        # Unnormalize the image

    def __len__(self) -> int:
        return len(self.df)

    def _process_image(self, row: pd.Series) -> torch.Tensor:
        img_path = f"{self.image_dir}/{row.study_id}/{row.image_id}.png"
        # Load image as NumPy array (H, W, C)
        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

        # if not apply_augmentation:
        transformed = self.base_transform(image=image)["image"]
        # else:
        #    transformed = self.augment_transform(image=image)["image"]
        # print("Augmentation applied")  # Optional debug print

        image_tensor = transformed.to(self.device)

        # print(
        #    f"Image shape: {image_tensor.shape} device {image_tensor.device}"
        # )  # Optional debug print
        # print(image_tensor.shape) # Optional debug print

        # Move to device in the training loop or collate_fn, not here
        # image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _preprocess_targets(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        # Get labels
        birads_label = self.birads_map[row.breast_birads]
        density_label = self.density_map[row.breast_density]
        w_scale = self.img_size[1] / row.cropped_width
        h_scale = self.img_size[0] / row.cropped_height

        # Parse coordinate strings
        xmin = ast.literal_eval(row.cropped_xmin)
        ymin = ast.literal_eval(row.cropped_ymin)
        xmax = ast.literal_eval(row.cropped_xmax)
        ymax = ast.literal_eval(row.cropped_ymax)
        finding = str(row.mapped_category)

        detections = []

        # Handle single category case (either as string or from a list with one item)
        if isinstance(finding, list):
            finding = finding[0]

        # Handle coordinates (either as values or from lists with one item)
        if isinstance(xmin, list):
            xmin = xmin[0]
            ymin = ymin[0]
            xmax = xmax[0]
            ymax = ymax[0]

        # Scale coordinates to target size
        xmin_scaled = float(xmin) * w_scale
        ymin_scaled = float(ymin) * h_scale
        xmax_scaled = float(xmax) * w_scale
        ymax_scaled = float(ymax) * h_scale

        label = self.cat2idx.get(
            finding, len(self.cat2idx) - 1
        )  # Default to last index if not foun
        detections.append([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled, label])

        targets = {
            "birads": torch.tensor(birads_label, dtype=torch.uint8).to(self.device),
            "density": torch.tensor(density_label, dtype=torch.uint8).to(self.device),
            "detections": torch.tensor(detections, dtype=torch.float32).to(
                self.device
            ),  # Shape: (num_boxes, 5)
        }

        return targets

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]

        # 1. Load Image
        img_path = f"{self.image_dir}/{row.study_id}/{row.image_id}.png"
        image = np.array(Image.open(img_path).convert("RGB"))

        # 2. Prepare Bounding Boxes and Labels for Albumentations
        bboxes = []
        class_labels = []
        finding = str(row.mapped_category)

        
        # ast.literal_eval can be slow, use direct access if format is consistent
        xmin_list = ast.literal_eval(row.cropped_xmin)
        ymin_list = ast.literal_eval(row.cropped_ymin)
        xmax_list = ast.literal_eval(row.cropped_xmax)
        ymax_list = ast.literal_eval(row.cropped_ymax)

        for i in range(len(xmin_list)):
            bboxes.append([xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]])
            label_idx = self.cat2idx.get(finding, len(self.cat2idx) - 1)
            class_labels.append(label_idx)

        # 3. Apply Transformations
        apply_augmentation = (
            row.get("split") == "training"
            and self.augment_duplicated
            and row.get("is_oversampled", False)
        )

        transform = (
            self.augment_transform if apply_augmentation else self.base_transform
        )

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        image_tensor = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]

        # 4. Format Targets
        detections = []
        for bbox, label in zip(transformed_bboxes, transformed_labels):
            detections.append([*bbox, label])

        targets = {
            "birads": torch.tensor(
                self.birads_map[row.breast_birads], dtype=torch.long
            ).to(self.device),
            "density": torch.tensor(
                self.density_map[row.breast_density], dtype=torch.long
            ).to(self.device),
            "detections": torch.tensor(detections, dtype=torch.float32).to(self.device),
        }
        image_tensor = image_tensor.to(self.device)
        return image_tensor, targets

    def __getitem__no_augment__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        apply_augmentation = (
            row.get("split") == "training"
            and self.augment_duplicated
            and row.get("is_oversampled", False)
        )

        image_tensor = self._process_image(row)

        target_tensors = self._preprocess_targets(row)

        # print(row)

        return image_tensor, target_tensors


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)

    birads = torch.stack([t["birads"] for t in targets])
    density = torch.stack([t["density"] for t in targets])
    detections = [t["detections"] for t in targets]  # keep as list of tensors

    batch_targets = {
        "birads": birads,
        "density": density,
        "detections": detections,
    }
    return images, batch_targets
