import torch

import torch
from torchvision import transforms
import numpy as np
from utils.visualize import convert_dicom_to_png
from fasterrcnn import CustomFasterRCNN, FasterRCNNConfig
from matplotlib import pyplot as plt
import torch.nn.functional as F
from typing import Dict


class MammographyInference:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model configuration
        self.config = FasterRCNNConfig()
        self.model = CustomFasterRCNN(self.config)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.size = (1130, 880)  # Same as dataset
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=self.size),
            ]
        )

    def preprocess_image(self, dicom_path: str) -> torch.Tensor:
        """Preprocess DICOM image following dataset preprocessing"""
        # Convert DICOM to PNG using existing utility
        png_img = convert_dicom_to_png(dicom_path)
        if png_img is None:
            raise ValueError(f"Could not convert DICOM file: {dicom_path}")

        img = png_img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.stack([img, img, img], axis=-1)

        img = self.transform(img)
        return img.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        dicom_path: str,
        confidence_threshold: float = 0.5,
        top_k: int = 1,
        visualize: str = False,
    ) -> Dict:
        """
        Perform inference on a single DICOM image

        Args:
            dicom_path: Path to DICOM file
            confidence_threshold: Threshold for detection confidence

        Returns:
            Dictionary containing:
                - boxes: Detection boxes [x1, y1, x2, y2]
                - labels: Detection class labels
                - scores: Detection confidence scores
                - birads: BIRADS classification probabilities
                - density: Density classification probabilities
        """
        img = self.preprocess_image(dicom_path)
        img = img.to(self.device)

        detections, birads_probs, density_probs = self.model(img)

        if len(detections[0]["boxes"]) > 0:
            _, top_k_indices = torch.topk(detections[0]["scores"], k=top_k)

            top_boxes = detections[0]["boxes"][top_k_indices]
            top_scores = detections[0]["scores"][top_k_indices]
            top_labels = detections[0]["labels"][top_k_indices]

        else:
            top_boxes = torch.empty(0, 4)
            top_scores = torch.empty(0)
            top_labels = torch.empty(0)

        birads_probs = F.softmax(birads_probs, dim=-1).cpu()
        density_probs = F.softmax(density_probs, dim=-1).cpu()

        if visualize:
            self._visualize(
                img[0].cpu().numpy().transpose(1, 2, 0),
                top_boxes,
                top_labels,
                birads_probs,
                density_probs,
            )

        return {
            "boxes": top_boxes.numpy(),
            "labels": top_labels.numpy(),
            "scores": top_scores.numpy(),
            "birads": birads_probs.numpy(),
            "density": density_probs.numpy(),
        }

    def _visualize(self, img: np.ndarray, boxes, labels, birads, density):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        for box, label, birad, dens in zip(boxes, labels, birads, density):
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                )
            )
            ax.text(x1, y1, f"{label} {birad} {dens}", color="red", fontsize=12)
        plt.show()
        return fig


def main():
    # Example usage
    model_path = "models/best_model.pth"
    dicom_path = "path/to/your/image.dicom"

    # Initialize inference
    inferencer = MammographyInference(model_path)

    # Run prediction
    results = inferencer.predict(dicom_path)

    # Print results
    print("Detections:")
    print(f"Found {len(results['boxes'])} lesions")
    print(f"BIRADS probabilities: {results['birads']}")
    print(f"Density probabilities: {results['density']}")

    for box, label, score in zip(
        results["boxes"], results["labels"], results["scores"]
    ):
        print(f"Label: {label}, Score: {score:.3f}, Box: {box}")


if __name__ == "__main__":
    main()
