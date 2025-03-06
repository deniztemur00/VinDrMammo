from pathlib import Path
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Union, List
import argparse


@dataclass
class YOLOConfig:
    """YOLO prediction configuration."""

    task: str = "detect"
    mode: str = "predict"
    model: Optional[str] = None
    source: Optional[str] = None
    device: Optional[str] = None
    conf: float = 0.00001
    iou: float = 0.7
    imgsz: Union[int, List[int]] = 992
    half: bool = False
    max_det: int = 100
    stream_buffer: bool = False
    visualize: bool = False
    augment: bool = False
    agnostic_nms: bool = False
    classes: Optional[Union[int, List[int]]] = None
    retina_masks: bool = False
    embed: Optional[str] = None
    save: bool = False
    optimize: bool = True
    verbose: bool = False

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class YoloCropper:
    def __init__(self, config: YOLOConfig, model_path: Path, output_dir: Path):
        """
        Initialize the YOLO cropper with configuration.

        Args:
            config: YOLOConfig dataclass with YOLO settings
            model_path: Path to the YOLO model file
            output_dir: Directory to save cropped images and results
        """
        self.config = config
        self.config.model = str(model_path)
        self.output_dir = Path(output_dir)
        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.config.model).to(self.config.device)
        self.results_dict = {}

        print(f"{self.config.device=}", f"{self.config.model=}")

    def crop_image(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Extract ROI from image using bounding box coordinates."""
        return image[y1:y2, x1:x2]

    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save the cropped image to the specified path."""
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def process_folder(self, folder_path: Path) -> None:
        """Process all images in a folder."""
        image_output_dir = self.output_dir / folder_path.name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Update source in config for current folder
        self.config.source = str(folder_path)
        # Run prediction
        results = self.model.predict(**self.config.to_dict())

        for result in results:
            image_height, image_width = map(int, result.orig_shape)
            boxes = result.boxes
            save_path = image_output_dir / Path(result.path).name

            if boxes:
                max_conf_box = max(boxes, key=lambda box: box.conf.item())
                if max_conf_box.conf.item() < 0.001:
                    x1, y1, x2, y2 = 0, 0, image_width, image_height
                else:
                    x1, y1, x2, y2 = map(int, max_conf_box.xyxy[0])

                cropped_img = self.crop_image(result.orig_img, x1, y1, x2, y2)
                self.save_image(cropped_img, save_path)

                self.results_dict[result.path] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "img_width": image_width,
                    "img_height": image_height,
                }

    def save_results(self, output_path: Path) -> None:
        """Save all cropping results to CSV file."""
        with open(output_path, "w") as file:
            file.write("image_path,x1,y1,x2,y2,img_width,img_height\n")
            for image_path, data in self.results_dict.items():
                x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
                image_width, image_height = data["img_width"], data["img_height"]
                file.write(
                    f"{image_path},{x1},{y1},{x2},{y2},{image_width},{image_height}\n"
                )

    def process_all(self, source_dir: Path) -> None:
        """Process all images in the source directory or its subfolders."""
        # First check if there are any directories
        image_folder_paths = [p for p in source_dir.glob("*/") if p.is_dir()]
        if image_folder_paths:
            # Process each subfolder
            for image_folder_path in tqdm(
                image_folder_paths, desc="Processing Image Folders"
            ):
                self.process_folder(image_folder_path)
        else:
            # If no subfolders found, process the main directory itself
            print(f"No subfolders found in {source_dir}, processing main directory")
            self.process_folder(source_dir)

        # Save results after processing all folders
        self.save_results(self.output_dir / "roi_crop_results.csv")


def main(
    model_path: Path,
    source_dir: Path,
    output_dir: Path,
):
    # Create configuration
    config = YOLOConfig()
    cropper = YoloCropper(config, model_path, output_dir)

    # Process all folders
    
    cropper.process_all(source_dir)


if __name__ == "__main__":

    # Base paths using relative paths as you specified
    default_model_path = Path("..") / "models" / "yolo_crop.pt"
    default_source_dir = Path("..") / "data_output"
    default_output_dir = Path("..") / "data_output_cropped"

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Crop mammography images using YOLO")
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model_path),
        help=f"Path to YOLO model (default: {default_model_path})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(default_source_dir),
        help=f"Source directory with PNG images (default: {default_source_dir})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output_dir),
        help=f"Output directory for cropped images (default: {default_output_dir})",
    )

    args = parser.parse_args()
    
    # Use the paths from arguments or defaults
    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "model_path": model_path,
        "source_dir": source_dir,
        "output_dir": output_dir,
    }

    main(**kwargs)
