import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # Import cv2 for border_mode constant


class MammoDataset(Dataset):
    def __init__(
        self,
        df,
        preprocessing_cfg=None,
        split="train",
        transform=None,
    ):
        self.df = df
        self.img_size = (224,224)  # Default image size for ConvNeXt models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = (
            "/home/deniztemur/Dataset/vindrmammo_findings_dataset_cropped/"
        )
        self.birads_map = self.birads_map = {
            label: idx for idx, label in enumerate(sorted(df["breast_birads"].unique()))
        }
        self.cfg = preprocessing_cfg or self._get_default_preprocessing_cfg()
        self.transform = transform or self._get_default_transform()
        self.split = split
        self.augment = (
            self._get_train_augmentation()
            if split == "train"
            else self._get_test_augmentation()
        )

    def _get_test_augmentation(self):
        return A.Compose(
            [
                A.Resize(288, 288),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def _get_train_augmentation(self):
        return A.Compose(
            [
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=0.10,
                    p=0.7,
                    border_mode=0,
                ),
                # Intensity augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                # A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                # Robustness
                A.GridDistortion(p=0.1),
                # Resize and ToTensor
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def _get_default_preprocessing_cfg(self):
        return {
            "denoise": "gaussian",  # Options: 'gaussian', 'median', None
            "contrast": "clahe",  # Options: 'clahe', 'hist_eq', None
            "sharpen": True,  # Boolean
        }

    def _get_default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def apply_preprocessing(self, image):
        # Convert PIL to numpy
        img = np.array(image)

        # Denoising
        if self.cfg.get("denoise") == "gaussian":
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif self.cfg.get("denoise") == "median":
            img = cv2.medianBlur(img, 5)

        # Contrast Enhancement
        if self.cfg.get("contrast") == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        elif self.cfg.get("contrast") == "hist_eq":
            img = cv2.equalizeHist(img)

        # Sharpening
        if self.cfg.get("sharpen", False):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

        return Image.fromarray(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(
            self.dataset_path, row["study_id"], row["image_id"] + ".png"
        )
        label = self.birads_map.get(row["breast_birads"], -1)

        img = Image.open(img_path).convert("L")  # Mammograms are grayscale
        image = np.array(img)
        #image = np.stack([img, img, img], axis=-1)
        # image = self.apply_preprocessing(image)

        # image = self.transform(image)
        #image = np.array(image)

        image = self.augment(image=image)["image"]

        label = torch.tensor(label, dtype=torch.long)

        # image = image.to(self.device)
        # label = label.to(self.device)

        return image, label


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
