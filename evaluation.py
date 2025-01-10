import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import cv2

import torch
from torchvision.ops import box_iou
import numpy as np
from typing import List, Dict, Tuple


def get_best_match(
    pred_box: torch.Tensor,
    pred_label: torch.Tensor,
    pred_score: torch.Tensor,
    true_box: torch.Tensor,
    true_label: torch.Tensor,
) -> Dict[str, float]:
    """Returns best matching prediction based on score and IoU"""
    if len(pred_box) == 0:
        return {"best_iou": 0.0, "best_score": 0.0, "matched": False}

    # Sort by confidence scores
    scores, indices = torch.sort(pred_score, descending=True)
    pred_box = pred_box[indices]
    pred_label = pred_label[indices]
    print(scores, indices)
    print("asdlkfjasşdklfj")

    # Filter by matching label
    mask = pred_label == true_label
    if not mask.any():
        return {"best_iou": 0.0, "best_score": 0.0, "matched": False}

    pred_box = pred_box[mask]
    pred_score = scores[mask]

    # Get IoUs for remaining predictions
    ious = torch.tensor([box_iou(pb, true_box) for pb in pred_box])

    if len(ious) > 0:
        best_idx = torch.argmax(ious)
        return {
            "best_iou": float(ious[best_idx]),
            "best_score": float(pred_score[best_idx]),
            "matched": True,
        }

    return {"best_iou": 0.0, "best_score": 0.0, "matched": False}


def evaluate_classification(
    pred_probs: torch.Tensor, true_labels: torch.Tensor, task: str = "birads"
) -> Dict[str, float]:
    """Evaluate classification metrics"""
    num_classes = 5 if task == "birads" else 4
    pred_labels = pred_probs.argmax(dim=1).cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=range(num_classes),
        ),
        "recall": recall_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=range(num_classes),
        ),
        "f1": f1_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=range(num_classes),
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, pred_labels, labels=range(num_classes)
        ),
    }

    return metrics


def visualize_detections(
    image, pred_boxes, pred_labels, true_boxes, true_labels, class_names
):
    """
    Draws bounding boxes and labels on the image for both predicted and true annotations.

    Args:
        image (np.array): Image array in RGB format.
        pred_boxes (np.array): Predicted bounding boxes in [x_min, y_min, x_max, y_max] format.
        pred_labels (np.array): Predicted labels.
        true_boxes (np.array): True bounding boxes in [x_min, y_min, x_max, y_max] format.
        true_labels (np.array): True labels.
        class_names (list): List of class names corresponding to labels.
    """
    image = image.copy()

    # Draw true boxes
    for box, label in zip(true_boxes, true_labels):
        color = (0, 255, 0)  # Green for true
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
        )
        cv2.putText(
            image,
            f"True: {class_names[label]}",
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    # Draw predicted boxes
    for box, label in zip(pred_boxes, pred_labels):
        color = (0, 0, 255)  # Red for predicted
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
        )
        cv2.putText(
            image,
            f"Pred: {class_names[label]}",
            (int(box[0]), int(box[1]) - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    plt.imshow(image)
    plt.show()


def plot_confusion_matrix(
    conf_matrix, classes, normalize=False, title="Confusion matrix"
):
    """
    Plots a confusion matrix using seaborn.

    Args:
        conf_matrix (np.array): Confusion matrix.
        classes (list): List of class labels.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
    """
    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
