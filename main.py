from dataset import MammographyDataset,DataLoader
import pandas as pd


df = pd.read_csv("metadata/stratified_local.csv")
ZIP_PATH = "N:\\IDM Downloads\\Compressions\\vindr-mammo.zip"
INTER_NAME = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"



# Read the stratified data
global_df = pd.read_csv("metadata/stratified_local.csv")

# Split into train and test based on fold
train_df = global_df[global_df['fold'] == 'training']
test_df = global_df[global_df['fold'] == 'test']



train_dataset = MammographyDataset(train_df, ZIP_PATH, INTER_NAME)
test_dataset = MammographyDataset(test_df, ZIP_PATH, INTER_NAME)

# Create dataloaders if needed
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


for i, batch in enumerate(train_loader):
    print(i, batch.shape)
    if i == 10:
        break

for i, batch in enumerate(test_loader):
    print(i, batch.shape)
    if i == 10:
        break