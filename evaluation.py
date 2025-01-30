import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

import cv2
import torch
from torchvision.ops import box_iou
from typing import Dict


FINDING_CATEGORIES = [
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
ALL_BIRADS = [
    "BI-RADS 1",
    "BI-RADS 2",
    "BI-RADS 3",
    "BI-RADS 4",
    "BI-RADS 5",
]

FINDING_BIRADS = [
    #   'BI-RADS 1',
    "BI-RADS 2",
    "BI-RADS 3",
    "BI-RADS 4",
    "BI-RADS 5",
]

DENSITY = [
    "DENSITY A",
    "DENSITY B",
    "DENSITY C",
    "DENSITY D",
]


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

    scores, indices = torch.sort(pred_score, descending=True)
    pred_box = pred_box[indices]
    pred_label = pred_label[indices]

    mask = pred_label == true_label
    if not mask.any():
        return {"best_iou": 0.0, "best_score": 0.0, "matched": False}

    pred_box = pred_box[mask]
    pred_score = scores[mask]

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
    pred_labels: np.ndarray, true_labels: np.ndarray, task: str = "birads"
) -> Dict[str, float]:
    """Evaluate classification metrics"""
    num_classes = 5 if task == "birads" else 4

    metrics = {
        # "accuracy": accuracy_score(true_labels, pred_labels),
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


def generate_list(result_dict, task: str = "detection") -> tuple[list, list]:
    labels, preds = [], []

    if task == "detection":
        for det, tgt in zip(result_dict["detections"], result_dict["targets"]):
            tgt_label = tgt["labels"].item()
            det_pred = det["labels"][0].item()

            preds.append(det_pred)
            labels.append(tgt_label)

        return labels, preds

    if task == "birads":
        for det, tgt in zip(result_dict["birad_results"], result_dict["birad_targets"]):
            preds.append(det)
            labels.append(tgt.item())

        return labels, preds

    if task == "density":
        for det, tgt in zip(
            result_dict["density_results"], result_dict["density_targets"]
        ):
            preds.append(det)
            labels.append(tgt.item())

        return labels, preds


def plot_confusion_matrix(
    labels: list,
    preds: list,
    target_names: list,
    normalize=False,
    title="Confusion Matrix",
):

    conf_matrix = confusion_matrix(labels, preds)
    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="Blues",
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def top_5_detections(result_dict):
    all_true_binary = []
    all_pred_top5_binary = []

    # Initialize storage for ROC AUC (class-wise scores and labels)
    class_scores = {i: [] for i in range(len(FINDING_CATEGORIES))}
    class_true = {i: [] for i in range(len(FINDING_CATEGORIES))}

    for idx in range(len(result_dict["detections"])):
        dets = result_dict["detections"][idx]
        tgts = result_dict["targets"][idx]

        # Extract true labels for the image (convert to CPU and numpy)
        true_labels = tgts["labels"].cpu().numpy().flatten()
        unique_true_labels = np.unique(true_labels)

        # Extract detections (scores and labels)
        det_scores = dets["scores"].cpu().numpy().flatten()
        det_labels = dets["labels"].cpu().numpy().flatten()

        # --- Classification Report (Top 5) ---
        # Sort detections by score (descending) and select top 5
        sorted_indices = np.argsort(-det_scores)
        top5_indices = sorted_indices[:5]
        top5_labels = det_labels[top5_indices]
        predicted_labels = np.unique(top5_labels)  # Deduplicate

        # Create binary vectors for true and predicted
        true_binary = np.zeros(len(FINDING_CATEGORIES), dtype=int)
        pred_binary = np.zeros(len(FINDING_CATEGORIES), dtype=int)
        for lbl in unique_true_labels:
            if lbl < len(FINDING_CATEGORIES):
                true_binary[lbl] = 1
        for lbl in predicted_labels:
            if lbl < len(FINDING_CATEGORIES):
                pred_binary[lbl] = 1
        all_true_binary.append(true_binary)
        all_pred_top5_binary.append(pred_binary)

    all_true_binary = np.array(all_true_binary)
    all_pred_top5_binary = np.array(all_pred_top5_binary)

    return all_true_binary, all_pred_top5_binary


def roc_auc(result_dict):

    # Initialize storage for ROC AUC (class-wise scores and labels)
    class_scores = {i: [] for i in range(len(FINDING_CATEGORIES))}
    class_true = {i: [] for i in range(len(FINDING_CATEGORIES))}

    for idx in range(len(result_dict["detections"])):
        dets = result_dict["detections"][idx]
        tgts = result_dict["targets"][idx]

        true_labels = tgts["labels"].cpu().numpy().flatten()

        # Extract detections (scores and labels)
        det_scores = dets["scores"].cpu().numpy().flatten()
        det_labels = dets["labels"].cpu().numpy().flatten()
        # --- ROC AUC Preparation ---
        # For each class, store max detection score and its true presence
        for class_idx in range(len(FINDING_CATEGORIES)):
            class_mask = det_labels == class_idx
            if np.any(class_mask):
                max_score = np.max(det_scores[class_mask])
            else:
                max_score = 0.0  # No detection for this class
            class_scores[class_idx].append(max_score)
            class_true[class_idx].append(1 if class_idx in true_labels else 0)

    return class_scores, class_true


def visualize_roc_auc(class_scores, class_true):
    plt.figure(figsize=(10, 8))
    for class_idx, class_name in enumerate(FINDING_CATEGORIES):
        scores = np.array(class_scores[class_idx])
        true = np.array(class_true[class_idx])

        # Skip if no positive samples
        if np.sum(true) == 0:
            continue

        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(true, scores)
        roc_auc = auc(fpr, tpr)

        # Plot curve
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class")
    plt.legend(loc="lower right")
    plt.show()
