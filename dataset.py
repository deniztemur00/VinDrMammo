import pandas as pd
from utils.visualize import convert_dicom_to_png
import zipfile
from matplotlib import pyplot as plt
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# global_df, local_df = split.process_data()
# split.show_df(split.count_box_birads(local_df[local_df.fold == "training"]))
# split.show_df(split.count_box_birads(local_df[local_df.fold == "test"]))
# local_df.to_csv('metadata/stratified_local.csv', index=False)
# global_df.to_csv('metadata/stratified_global.csv', index=False)
# print(local_df.head())
# print(global_df.head())
############################################




from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import zipfile
import io
from utils.visualize import convert_dicom_to_png

class MammographyDataset(Dataset):
    def __init__(self, df, zip_path, inter_name, transform=None):
        self.df = df
        self.zip_path = zip_path
        self.inter_name = inter_name
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1920, 2444))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_id = self.df.iloc[idx]["study_id"]
        image_id = self.df.iloc[idx]["image_id"]
        
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            dicom_path = f"{self.inter_name}/{study_id}/{image_id}.dicom"
            with zf.open(dicom_path) as dicom_file:
                dicom_bytes = io.BytesIO(dicom_file.read())
                png_img = convert_dicom_to_png(dicom_bytes)
                img = self.transform(png_img)
                return img


