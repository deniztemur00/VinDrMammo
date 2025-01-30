import argparse
import pandas as pd
from inference import MammographyInference
from dataset_v2 import MammographyDataset


INTER_NAME = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
CSV_PATH = "metadata\\final_only_findings.csv"
def main():
    parser = argparse.ArgumentParser(description="Run inference on a DICOM image.")
    #parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    #parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with metadata.")
    #parser.add_argument("--inter_name", type=str, required=True, help="Intermediate name for DICOM path.")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to the ZIP file containing DICOM images.")
    parser.add_argument("--idx", type=int, default=8, help="Index of the test sample to run inference on.")
    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(CSV_PATH)
    test_df = df[df['fold'] == 'test']
    test_dataset = MammographyDataset(test_df, args.zip_path, args.inter_name)

    # Initialize the inferencer
    inferencer = MammographyInference(model_path=args.model_path)

    # Get the DICOM path and the image/target
    row = test_df.iloc[args.idx]
    dicom_path = f"{args.inter_name}/{row.study_id}/{row.image_id}.dicom"
    img, target = test_dataset[args.idx]

    # Run inference
    inference_result = inferencer.predict(dicom_path=dicom_path, top_k=1, visualize=True)

    # Visualize ground truth
    inferencer.visualize_gt(img, target)

if __name__ == "__main__":
    main()