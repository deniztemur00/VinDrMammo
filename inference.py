import torch

from torchvision import transforms
import numpy as np
from utils.visualize import convert_dicom_to_png

# from fasterrcnn import CustomFasterRCNN, FasterRCNNConfig
from retinanet import CustomRetinaNet, RetinaNetConfig
from matplotlib import pyplot as plt
import torch.nn.functional as F
from typing import Dict
from dataset import create_categories, pd


class MammographyInference:
    def __init__(self, model_path: str, df_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model configuration
        self.config = RetinaNetConfig()
        self.model = CustomRetinaNet(self.config)

        self.birad_categories = [
            "BIRADS-1",
            "BIRADS-2",
            "BIRADS-3",
            "BIRADS-4",
            "BIRADS-5",
        ]
        self.density_categories = ["Density-A", "Density-B", "Density-C", "Density-D"]
        self.df = pd.read_csv(df_path)
        self.all_categories, self.cat2idx = create_categories(self.df)

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
        with torch.no_grad():
            detections, birads_logits, density_logits = self.model(img)

        if len(detections[0]["boxes"]) > 0:
            _, top_k_indices = torch.topk(detections[0]["scores"], k=top_k)

            top_boxes = detections[0]["boxes"][top_k_indices]
            top_scores = detections[0]["scores"][top_k_indices]
            top_labels = detections[0]["labels"][top_k_indices]

        else:
            top_boxes = torch.empty(0, 4)
            top_scores = torch.empty(0)
            top_labels = torch.empty(0)

        birads_probs = F.softmax(birads_logits, dim=-1).cpu()
        density_probs = F.softmax(density_logits, dim=-1).cpu()

        birads_confidence, birads_indice = torch.max(birads_probs, -1)
        density_confidence, density_indice = torch.max(density_probs, -1)
        birads = self.birad_categories[birads_indice]
        density = self.density_categories[density_indice]
        if visualize:
            self._visualize(
                img[0].cpu().numpy().transpose(1, 2, 0),
                top_boxes,
                top_scores,
                top_labels,
            )

        top_labels = top_labels.cpu().numpy()[0]
        top_category = self.all_categories[top_labels]
        inference_dict = {
            "boxes": top_boxes.numpy()[0],
            "finding_category": top_category,
            "scores": top_scores.numpy()[0],
            "birads": birads,
            "birads_confidence": birads_confidence[0],
            "density": density,
            "density_confidence": density_confidence[0],
        }
        print(inference_dict)

        return inference_dict

    def _visualize(self, img: np.ndarray, boxes, scores, labels):

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        for box, score, label in zip(boxes, scores, labels):
            category = self.all_categories[label.item()]
            score = score.item()
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                )
            )
            ax.text(x1, y1, f"{category}:{score:.4f}", color="red", fontsize=12)
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
