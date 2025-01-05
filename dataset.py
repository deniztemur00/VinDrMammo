import pandas as pd
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
        self.transform: transforms.Compose = (
            transforms.ToTensor(),
            transforms.Resize(size=(880, 1130)),  # mean / 3
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        )
        self.categories: List[str] = None
        self.cat2idx: Dict[str, int] = None

    def set_categories(self, categories: List[str], cat2idx: Dict[str, int]) -> None:
        self.categories = categories
        self.cat2idx = cat2idx

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
                img: torch.Tensor = self.transform(png_img)

        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

        if not pd.isna(row["xmin"]):

            boxes: torch.Tensor = torch.tensor(
                [
                    [
                        float(row["xmin"]),
                        float(row["ymin"]),
                        float(row["xmax"]),
                        float(row["ymax"]),
                    ]
                ],
                dtype=torch.float32,
            )

            categories: List[str] = row["finding_categories"]
            if isinstance(categories, str):
                categories = ast.literal_eval(categories)
            labels: torch.Tensor = torch.tensor(
                [self.cat2idx[categories[0]]], dtype=torch.int64
            )

            target["boxes"] = boxes
            target["labels"] = labels

        return img, target
