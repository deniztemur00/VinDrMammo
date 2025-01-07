import pandas as pd
import numpy as np
from utils.visualize import convert_dicom_to_png
import zipfile
import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import ast
from typing import List, Dict, Tuple

# global_df, local_df = split.process_data()
# split.show_df(split.count_box_birads(local_df[local_df.fold == "training"]))
# split.show_df(split.count_box_birads(local_df[local_df.fold == "test"]))
# local_df.to_csv('metadata/stratified_local.csv', index=False)
# global_df.to_csv('metadata/stratified_global.csv', index=False)
# print(local_df.head())
# print(global_df.head())
############################################


def create_categories(df: pd.DataFrame) -> tuple[List[str], Dict[str, int]]:
    all_categories = []
    for cats_str in df.finding_categories.value_counts().index:
        cats = ast.literal_eval(cats_str)
        if len(cats) > 1:
            all_categories.append("-".join(cats))
        else:
            all_categories.append(cats[0])
    cat2idx = {cat: idx for idx, cat in enumerate(all_categories)}
    return all_categories, cat2idx


def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

class MammographyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str,
        inter_name: str,
    ) -> None:
        self.df: pd.DataFrame = df
        self.zip_path: str = zip_path
        self.inter_name: str = inter_name

        self.size = (1130, 880)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=self.size),  # mean / 3
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                #),
                # model will normalize internally
            ]
        )
        self.categories: List[str] = None
        self.cat2idx: Dict[str, int] = None

    def set_categories(self, categories: List[str], cat2idx: Dict[str, int]) -> None:
        self.categories = categories
        self.cat2idx = cat2idx

    def _scale_bbox(self, idx: int) -> torch.Tensor:
        row: pd.Series = self.df.iloc[idx]
        original_size = (row["width"], row["height"])
        scale_x = self.size[0] / original_size[0]
        scale_y = self.size[1] / original_size[1]
        boxes: torch.Tensor = torch.tensor(
            [
                [
                    float(row["xmin"]) * scale_x,
                    float(row["ymin"]) * scale_y,
                    float(row["xmax"]) * scale_x,
                    float(row["ymax"]) * scale_y,
                ]
            ],
            dtype=torch.float32,
        )
        return boxes

    def _scale_img(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.stack([img, img, img], axis=-1)
        # img = np.stack([img] * 3)
        return img

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row: pd.Series = self.df.iloc[idx]
        study_id: str = row["study_id"]
        image_id: str = row["image_id"]

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            dicom_path: str = f"{self.inter_name}/{study_id}/{image_id}.dicom"
            with zf.open(dicom_path) as dicom_file:
                dicom_bytes: io.BytesIO = io.BytesIO(dicom_file.read())
                png_img = convert_dicom_to_png(dicom_bytes)

                png_img = self._scale_img(png_img)
                img = self.transform(png_img)

        target = {
            "boxes": torch.ones((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        if not pd.isna(row["xmin"]):
            boxes = self._scale_bbox(idx)
        else:
            boxes = torch.ones((1, 4), dtype=torch.float32)

        labels = torch.tensor(
            [self.cat2idx["-".join(ast.literal_eval(row["finding_categories"]))]],
            dtype=torch.int64,
        )

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["size"] = (row["height"], row["width"])
        target["category"] = row["finding_categories"]  # for debugging

        return img, target
