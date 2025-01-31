import argparse
import pandas as pd
from inference import MammographyInference
from dataset_v2 import MammographyDataset


INTER_NAME = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
CSV_PATH = "metadata\\final_only_findings.csv"
MODEL_PATH = "models\\resnet101_retinanet_final_only_findings_best_model.pth"
def main():
    parser = argparse.ArgumentParser(description="Run inference on a DICOM image.")
    #parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    #parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with metadata.")
    #parser.add_argument("--inter_name", type=str, required=True, help="Intermediate name for DICOM path.")
    parser.add_argument("--dicom_path", type=str, required=True, help="Path to the dicom image.")
    #parser.add_argument("--idx", type=int, default=-1, help="Index of the test sample to run inference on.")
    args = parser.parse_args()

    

    # Initialize the inferencer
    inferencer = MammographyInference(model_path=MODEL_PATH)

    

    # Run inference
    inference_result = inferencer.predict(dicom_path=args.dicom_path, top_k=1, visualize=True)

    # Visualize ground truth
    #inferencer.visualize_gt(img, target)

if __name__ == "__main__":
    main()