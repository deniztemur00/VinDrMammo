import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # Import cv2 for border_mode constant


class ClassificationDataset(Dataset):
    """
    Dataset for classifying breast BI-RADS and density using PNG images.
    Applies augmentations only to rows marked as duplicated.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,  # Directory containing the PNG images
        img_size: Tuple[int, int] = (224, 224),
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
        self.num_birads_classes = len(self.birads_map)
        self.num_density_classes = len(self.density_map)

        # Basic transforms (applied to all images)
        self.base_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

        # Augmentations using Albumentations
        self.augment_transform = A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                # --- Augmentations Start ---
                A.HorizontalFlip(p=0.5),  # Adjusted p=0.5
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # --- Augmentations End ---
                A.Normalize(
                    mean=self.mean, std=self.std, max_pixel_value=255
                ),  # Adjust mean/std if single channel
                ToTensorV2(),
            ],
            seed=42,
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

        apply_augmentation = (
            row.get("split") == "training"
            and self.augment_duplicated
            and row.get("is_duplicated", False)
        )

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

        targets = {
            "birads": torch.tensor(birads_label, dtype=torch.uint8).to(self.device),
            "density": torch.tensor(density_label, dtype=torch.uint8).to(self.device),
        }

        return targets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]

        image_tensor = self._process_image(row)

        target_tensors = self._preprocess_targets(row)

        # print(row)

        return image_tensor, target_tensors


def collate_fn(batch):
    """
    Custom collate function to handle the dictionary of targets.
    """
    images = []
    targets_birads = []
    targets_density = []

    for img, target in batch:
        images.append(img)
        targets_birads.append(target["birads"])
        targets_density.append(target["density"])

    images = torch.stack(images, dim=0)
    targets = {
        "birads": torch.stack(targets_birads, dim=0),
        "density": torch.stack(targets_density, dim=0),
    }

    return images, targets
