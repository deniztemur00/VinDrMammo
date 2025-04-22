import torch
from torchvision import transforms
import numpy as np
from utils.dicom2png import convert_dicom_to_png
from retinanet_v2 import CustomRetinaNet, RetinaNetConfig
from matplotlib import pyplot as plt
import torch.nn.functional as F
from typing import Dict
from PIL import Image


class MammographyInference:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model configuration
        self.config = RetinaNetConfig()
        self.model = CustomRetinaNet(self.config)

        self.finding_categories = [
            "Mass",
            "Suspicious Calcification",
            "Asymmetry",
            "Focal Asymmetry",
            "Global Asymmetry",
            "Architectural Distortion",
            "Skin Thickening",
            "Skin Retraction",
            "Nipple Retraction",
            "Suspicious Lymph Node",
            "Other",
        ]

        self.finding_categories_top3 = [
            "Mass",
            "Suspicious Calcification",
            "Focal Asymmetry",
        ]
        self.finding_birads = [
            # "BIRADS-1",
            "BIRADS-2",
            "BIRADS-3",
            "BIRADS-4",
            "BIRADS-5",
        ]
        self.birad_categories = [
            "BIRADS-1",
            "BIRADS-2",
            "BIRADS-3",
            "BIRADS-4",
            "BIRADS-5",
        ]
        self.density_categories = ["Density-A", "Density-B", "Density-C", "Density-D"]

        # Load model weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}. Device: {self.device}")
        # Image preprocessing
        self.size = (800, 800)  # Same as dataset
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.size),
                # transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(num_output_channels=3),
            ]
        )

    def preprocess_image_dicom(self, dicom_path: str) -> torch.Tensor:
        """Preprocess DICOM image following dataset preprocessing"""
        # Convert DICOM to PNG using existing utility
        png_img = convert_dicom_to_png(dicom_path)
        if png_img is None:
            raise ValueError(f"Could not convert DICOM file: {dicom_path}")

        img = np.stack([png_img] * 3, axis=-1).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = self.transform(img)
        img = img.to(self.device)
        return img.unsqueeze(0)

    def preprocess_image_png(self, png_path: str) -> torch.Tensor:

        img = np.array(Image.open(png_path))
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
        img = np.stack([img] * 3, axis=-1)
        img = self.transform(img)
        img = img.to(self.device)
        return img.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        img_path: str,
        top_k: int = 3,
        visualize: bool = False,
        conf_threshold: float = 0.2,
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
        img = self.preprocess_image_png(img_path)
        with torch.no_grad():
            outputs = self.model(img)

        detections = outputs["detections"]
        birads_logits = outputs["birads_logits"]
        density_logits = outputs["density_logits"]

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
        birads = self.finding_birads[birads_indice]
        density = self.density_categories[density_indice]
        birads_confidence = birads_confidence.cpu().numpy()[0]
        density_confidence = density_confidence.cpu().numpy()[0]
        if visualize:
            self._visualize(
                img[0].cpu().numpy().transpose(1, 2, 0),
                top_boxes.cpu().numpy(),
                top_scores.cpu().numpy(),
                top_labels.cpu().numpy(),
                birads,
                birads_confidence,
                density,
                density_confidence,
            )

        top_labels = top_labels.cpu().numpy()[0]
        top_category = self.finding_categories_top3[top_labels]
        inference_dict = {
            "boxes": top_boxes.cpu().numpy()[0],
            "finding_category": top_category,
            "scores": top_scores.cpu().numpy()[0],
            "birads": birads,
            "birads_confidence": birads_confidence,
            "density": density,
            "density_confidence": density_confidence,
        }
        # print(inference_dict)

        return inference_dict

    def _visualize(
        self,
        img: np.ndarray,
        boxes,
        scores,
        labels,
        birads,
        birads_confidence,
        density,
        density_confidence,
    ):

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        botttom_text = f"{birads}: {birads_confidence:.4f}\n"  # {density}: {density_confidence:.4f}
        for box, score, label in zip(boxes, scores, labels):
            category = self.finding_categories_top3[label.item()]
            score = score.item()
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
                )
            )
            ax.text(
                0.40,
                0.05,
                botttom_text,
                transform=ax.transAxes,
                color="red",
                fontsize=12,
            )

            ax.text(x1, y1, f"{category}:{score:.4f}", color="red", fontsize=12)
        plt.show()
        return fig

    def visualize_gt(self, img: torch.tensor, target: dict):
        img = img.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        birads = self.birad_categories[target["birads"].item()]
        density = self.density_categories[target["density"].item()]
        category = self.finding_categories_top3[target["labels"].item()]

        bottom_text = f"{birads}\n{density}"
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="green", linewidth=2
                )
            )
            ax.text(
                0.40,
                0.05,
                bottom_text,
                transform=ax.transAxes,
                color="green",
                fontsize=12,
            )
            ax.text(x1, y1, f"{category}", color="green", fontsize=12)
        plt.show()

        result_dict = {
            "boxes": target["boxes"].numpy(),
            "finding_category": category,
            "birads": birads,
            "density": density,
        }
        return result_dict
