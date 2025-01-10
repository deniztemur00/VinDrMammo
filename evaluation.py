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
import numpy as np
from typing import List, Dict, Tuple


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


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
    print(scores,indices)
    # Filter by matching label
    mask = pred_label == true_label
    if not mask.any():
        return {"best_iou": 0.0, "best_score": 0.0, "matched": False}

    pred_box = pred_box[mask]
    pred_score = scores[mask]

    # Get IoUs for remaining predictions
    ious = torch.tensor([compute_iou(pb, true_box) for pb in pred_box])

    if len(ious) > 0:
        best_idx = torch.argmax(ious)
        return {
            "best_iou": float(ious[best_idx]),
            "best_score": float(pred_score[best_idx]),
            "matched": True,
        }

    return {"best_iou": 0.0, "best_score": 0.0, "matched": False}


def evaluate_single_detection(
    pred_box, pred_label, true_box, true_label, iou_thresh=0.5
):
    """Evaluate single detection against ground truth"""
    if len(pred_box) == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    iou = compute_iou(pred_box[0], true_box[0])
    correct = (iou >= iou_thresh) and (pred_label[0] == true_label[0])

    return {
        "precision": float(correct),
        "recall": float(correct),
        "f1": float(correct),
        "iou": float(iou),
    }


def evaluate_classification(pred_probs: torch.Tensor, true_labels: torch.Tensor):
    """Evaluate classification metrics using sklearn"""
    pred_labels = pred_probs.argmax(dim=1).cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="macro"),
        "recall": recall_score(true_labels, pred_labels, average="macro"),
        "f1": f1_score(true_labels, pred_labels, average="macro"),
        "confusion_matrix": confusion_matrix(true_labels, pred_labels),
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
