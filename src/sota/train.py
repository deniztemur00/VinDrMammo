import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    auc,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
from itertools import cycle
from scipy.special import softmax


@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 0.07
    model_dir: str = "models/"
    plot_dir: str = "plots/"
    name: str = "SOTA_v1.0"
    birads_loss_weight: float = 0.8
    density_loss_weight: float = 0.2
    focal_loss_gamma: float = 2.0
    ##Calculated weights (inversely proportional):
    birads_class_weights = {
        "BI-RADS 1": 0.418775,
        "BI-RADS 2": 0.711345,
        "BI-RADS 3": 2.072876,
        "BI-RADS 4": 1.977734,
        "BI-RADS 5": 4.581900,
    }

    density_class_weights = {
        "DENSITY A": 40.322581,
        "DENSITY B": 1.556663,
        "DENSITY C": 0.377872,
        "DENSITY D": 1.456876,
    }


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate CE loss without reduction to get per-example losses
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # Calculate the probability of the correct class p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Calculate the focal loss term: (1 - pt)^gamma * ce_loss
        focal_term = (1 - pt) ** self.gamma * ce_loss

        # Apply the specified reduction
        if self.reduction == "mean":
            return focal_term.mean()
        elif self.reduction == "sum":
            return focal_term.sum()
        else:  # 'none'
            return focal_term


class ClassificationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainerConfig,
    ):
        self.model = model
        torch.set_float32_matmul_precision(
            "high"
        )  # # Set precision for matmul operations
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = config.epochs
        self.model_dir = config.model_dir
        self.plot_dir = config.plot_dir
        self.name = config.name if config.name else "classification_model"
        self.birads_loss_weight = config.birads_loss_weight
        self.density_loss_weight = config.density_loss_weight

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",  # Options: "reduce-overhead", "max-autotune"
            )
            print("Model compiled with torch.compile()")
        else:
            print("Warning: PyTorch version <2.0 - torch.compile unavailable")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Loss functions
        self.birads_class_weights = torch.tensor(
            list(config.birads_class_weights.values()), device=self.device
        )

        self.density_class_weights = torch.tensor(
            list(config.density_class_weights.values()), device=self.device
        )

        # self.birads_loss_fn = FocalLoss(gamma=config.focal_loss_gamma)
        # self.density_loss_fn = FocalLoss(gamma=config.focal_loss_gamma)

        self.birads_loss_fn = nn.CrossEntropyLoss(self.birads_class_weights)
        self.density_loss_fn = nn.CrossEntropyLoss(weight=self.density_class_weights)

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )
        # Mean Average Precision metric
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )
        print(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):_}"
        )

        print(f"Device: {self.device}, Model: {self.model.__class__.__name__}")
        print(
            f"Optimizer: {self.optimizer.__class__.__name__}, Scheduler: {self.scheduler.__class__.__name__}"
        )
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"BIRADS class weights: {self.birads_class_weights}")
        print(f"DENSITY class weights: {self.density_class_weights}")

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.birads_recall_scores = []
        self.density_recall_scores = []
        self.birads_precision_scores = []
        self.density_precision_scores = []
        self.detection_map_scores = []
        self.current_epoch = 0

        # Store final validation results
        self.final_birads_preds = []
        self.final_birads_targets = []
        self.final_birads_logits = []
        self.final_density_preds = []
        self.final_density_targets = []
        self.final_density_logits = []
        self.birads_labels = list(config.birads_class_weights.keys())
        self.density_labels = list(config.density_class_weights.keys())

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def train(self):
        # best_val_loss = float("inf")
        best_birads_f1 = 0.0
        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.model.train()
                epoch_loss = 0.0
                epoch_birads_loss = 0.0  # Accumulate birads loss
                epoch_density_loss = 0.0  # Accumulate density loss
                epoch_detection_loss = 0.0  # Accumulate detection loss
                batch_count = 0
                train_pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.epochs} - Training",
                    leave=True,
                )

                for images, targets in train_pbar:
                    self.optimizer.zero_grad()

                    outputs = self.model(images, targets)

                    loss, loss_dict = self._calculate_loss(outputs, targets)
                    finding_loss = outputs["finding_loss"].item()
                    reg_loss = outputs["reg_loss"].item()

                    loss.backward()
                    finding_loss.backward()
                    reg_loss.backward()

                    # Optional: Gradient clipping
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    epoch_loss += loss.item() + finding_loss + reg_loss
                    epoch_birads_loss += loss_dict["birads_loss"].item()
                    epoch_density_loss += loss_dict["density_loss"].item()
                    epoch_detection_loss += finding_loss + reg_loss
                    batch_count += 1

                    # Update progress bar postfix
                    lr = self.scheduler.get_last_lr()[0]  # Get current LR
                    train_pbar.set_postfix(
                        {
                            "avg_birads_loss": f"{epoch_birads_loss / batch_count:.4f}",
                            "avg_density_loss": f"{epoch_density_loss / batch_count:.4f}",
                            "avg_detection_loss": f"{epoch_detection_loss / batch_count:.4f}",
                            "avg_total_loss": f"{epoch_loss / batch_count:.4f}",
                            "LR": f"{lr:.6f}",
                        }
                    )

                # Step the scheduler after each epoch
                self.scheduler.step()

                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.train_losses.append(avg_epoch_loss)
                self.save_loss_plot()

                # Validation
                if self.val_loader:
                    val_loss, metrics = self.validate()
                    self.val_losses.append(val_loss)
                    self.birads_precision_scores.append(metrics["birads_precision"])
                    self.density_precision_scores.append(metrics["density_precision"])
                    self.birads_recall_scores.append(metrics["birads_recall"])
                    self.density_recall_scores.append(metrics["density_recall"])
                    self.save_metrics_plots()  # Save plots each epoch
                    self.save_detection_results()  # Save detection results each epoch

                    # print(
                    #    f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {val_loss:.4f}, "
                    #    f"BiRADS F1: {metrics['birads_f1']:.4f}, Density F1: {metrics['density_f1']:.4f}"
                    # )
                    current_birads_f1 = metrics["birads_f1"]
                    if current_birads_f1 > best_birads_f1:
                        print(
                            f"BiRADS F1 score improved from {best_birads_f1:.4f} to {current_birads_f1:.4f}. Saving best model..."
                        )
                        best_birads_f1 = current_birads_f1
                        self.save(
                            os.path.join(self.model_dir, f"{self.name}_best_model.pth")
                        )
                        self.save_final_metrics_report()  # save only best metrics report

                self.save(os.path.join(self.model_dir, f"{self.name}_last_epoch.pth"))
            self.save_loss_plot()

            print("Training finished.")
            # Save final loss plots

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current model state...")
            self.save(
                os.path.join(self.model_dir, f"{self.name}_interrupted_model.pth")
            )
            self.save_loss_plot()
            self.save_metrics_plots()
            print("Model saved.")

        # might return history

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.map_metric.reset()

        total_val_loss = 0.0
        val_birads_loss = 0.0
        val_density_loss = 0.0
        val_batch_count = 0

        all_birads_preds = []
        all_birads_targets = []
        all_birads_logits = []

        all_density_preds = []
        all_density_targets = []
        all_density_logits = []

        all_map_scores = []

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs} - Validation",
            leave=True,
        )

        for images, targets in val_pbar:

            outputs = self.model(images)

            loss, loss_dict = self._calculate_loss(outputs, targets)
            total_val_loss += loss.item()
            val_birads_loss += loss_dict["birads_loss"].item()
            val_density_loss += loss_dict["density_loss"].item()
            val_batch_count += 1

            # Get predictions
            birads_preds = torch.argmax(outputs["birads_logits"], dim=1)
            density_preds = torch.argmax(outputs["density_logits"], dim=1)

            # Store predictions,targets,and logits for metric calculation
            all_birads_logits.append(outputs["birads_logits"].cpu().numpy())
            all_density_logits.append(outputs["density_logits"].cpu().numpy())
            all_birads_preds.extend(birads_preds.cpu().numpy())
            all_birads_targets.extend(targets["birads"].cpu().numpy())
            all_density_preds.extend(density_preds.cpu().numpy())
            all_density_targets.extend(targets["density"].cpu().numpy())

            # detection results
            all_preds, all_targets = self._prepare_map(outputs, targets)
            self.map_metric.update(all_preds, all_targets)
            mAP = self.map_metric.compute()["map"]  # only bbox mAP
            all_map_scores.append(mAP)

            birads_f1 = f1_score(
                all_birads_targets, all_birads_preds, average="macro", zero_division=0
            )
            density_f1 = f1_score(
                all_density_targets, all_density_preds, average="macro", zero_division=0
            )

            # Update progress bar postfix (optional)
            val_pbar.set_postfix(
                {
                    "birads_f1": f"{birads_f1:.4f}",  # F1 calculated on accumulated preds
                    "density_f1": f"{density_f1:.4f}",  # F1 calculated on accumulated preds
                    "running_mAP": f"{mAP:.4f}",
                    "avg_birads_loss": f"{val_birads_loss / val_batch_count:.4f}",
                    "avg_density_loss": f"{val_density_loss/ val_batch_count:.4f}",
                    "avg_total_loss": f"{total_val_loss / val_batch_count:.4f}",  # Use total_val_loss here as it's already accumulated
                }
            )

        avg_val_loss = total_val_loss / len(self.val_loader)

        # Or using sklearn directly:
        birads_precision = precision_score(
            all_birads_targets, all_birads_preds, average="macro", zero_division=0
        )
        density_precision = precision_score(
            all_density_targets, all_density_preds, average="macro", zero_division=0
        )
        birads_recall = recall_score(
            all_birads_targets, all_birads_preds, average="macro", zero_division=0
        )
        density_recall = recall_score(
            all_density_targets, all_density_preds, average="macro", zero_division=0
        )

        birads_f1 = f1_score(
            all_birads_targets, all_birads_preds, average="macro", zero_division=0
        )

        density_f1 = f1_score(
            all_density_targets, all_density_preds, average="macro", zero_division=0
        )

        final_detection_results = self.map_metric.compute()

        # Save final predictions and targets for later analysis
        self.final_birads_preds = all_birads_preds
        self.final_birads_targets = all_birads_targets
        self.final_birads_logits = all_birads_logits
        self.final_density_preds = all_density_preds
        self.final_density_targets = all_density_targets
        self.final_density_logits = all_density_logits
        self.detection_map_scores = final_detection_results
        metrics = {
            "birads_precision": birads_precision,  # Get F1 or default to 0
            "density_precision": density_precision,
            "birads_recall": birads_recall,
            "density_recall": density_recall,
            "birads_f1": birads_f1,
            "density_f1": density_f1,
            "detection_results": final_detection_results,
            # Add other metrics from evaluate_classification if needed
        }

        return avg_val_loss, metrics

    def _calculate_loss(self, outputs, targets):
        """Calculates the combined weighted loss."""
        birads_logits = outputs["birads_logits"]
        density_logits = outputs["density_logits"]

        birads_target = targets["birads"]
        density_target = targets["density"]

        birads_loss = self.birads_loss_fn(birads_logits, birads_target)
        density_loss = self.density_loss_fn(density_logits, density_target)

        total_loss = (
            self.birads_loss_weight * birads_loss
            + self.density_loss_weight * density_loss
        )

        return total_loss, {
            "birads_loss": birads_loss,
            "density_loss": density_loss,
            "total_loss": total_loss,
        }

    def _prepare_map(self, outputs, dummy_annotations):
        """
        Processes model outputs and ground truth annotations for a full batch
        to prepare them for the mAP metric.
        """
        all_preds = []
        all_targets = []

        batch_size = len(outputs["detections"])

        for i in range(batch_size):

            cls_scores, cls_indices, bboxes = outputs["detections"][i]

            # Move tensors to CPU
            cls_scores = cls_scores.cpu()
            cls_indices = cls_indices.cpu()
            bboxes = bboxes.cpu()

            # Filter out invalid boxes where x1 > x2 or y1 > y2
            valid_indices = (bboxes[:, 0] < bboxes[:, 2]) & (
                bboxes[:, 1] < bboxes[:, 3]
            )

            all_preds.append(
                {
                    "boxes": bboxes[valid_indices],
                    "labels": cls_indices[valid_indices],
                    "scores": cls_scores[valid_indices],
                }
            )

            annotation = dummy_annotations[i].cpu()

            all_targets.append(
                {
                    "boxes": annotation[:, :4],
                    "labels": annotation[:, 4].long(),
                }
            )
            print(f"Processed image {i + 1}/{batch_size}:")

        return all_preds, all_targets

    def save(self, filename):
        """Saves the model state dictionary."""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Loads the model state dictionary."""
        print(f"Loading model from {filename}")
        self.model.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)

    def save_detection_plot(self):
        fig, ax = self.map_metric.plot()

        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # ensure the legend is not cut off
        plot_path = os.path.join(self.plot_dir, f"{self.name}_mAP_plot.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close()

    def save_loss_plot(self):
        """Saves plots for training and validation losses."""
        epochs = list(
            range(1, self.current_epoch + 2)
        )  # +1 for current epoch, +1 for range end

        plt.figure(figsize=(12, 5))

        # Plot Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(
            epochs[: len(self.train_losses)],
            self.train_losses,
            "b-o",
            label="Training Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()

        # Plot Validation Loss (if available)
        if self.val_losses:
            plt.subplot(1, 2, 2)
            plt.plot(
                epochs[: len(self.val_losses)],
                self.val_losses,
                "r-o",
                label="Validation Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f"{self.name}_losses.png")
        plt.savefig(plot_path)
        # print(f"Loss plots saved to {plot_path}")
        plt.close()

    def save_final_metrics_report(self):
        if not self.val_loader or not self.final_birads_targets:
            print("Skipping final metrics report: No validation results available.")
            return

        # Helper function to plot ROC curves
        def plot_roc_curve(y_true, y_score, labels, task_name, plot_dir, model_name):
            n_classes = len(labels)
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_true_bin.ravel(), y_score.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            lw = 2  # line width

            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.3f})',
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.3f})',
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            colors = cycle(
                ["aqua", "darkorange", "cornflowerblue", "green", "red", "purple"]
            )
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=lw,
                    label=f"ROC curve of class {labels[i]} (area = {roc_auc[i]:0.3f})",
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{task_name} Multi-class ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            roc_plot_path = os.path.join(
                plot_dir, f"{model_name}_{task_name.lower()}_roc_curve.png"
            )
            plt.savefig(roc_plot_path)
            plt.close()
            return roc_auc  # Return AUCs, maybe add macro AUC to report

        # --- BI-RADS Metrics ---
        try:
            birads_report = classification_report(
                self.final_birads_targets,
                self.final_birads_preds,
                target_names=self.birads_labels,
                digits=4,
                zero_division=0,
            )
            # print(birads_report) # Keep printing optional

            # Confusion Matrix (code unchanged)
            cm_birads = confusion_matrix(
                self.final_birads_targets,
                self.final_birads_preds,
                labels=range(len(self.birads_labels)),
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_birads,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.birads_labels,
                yticklabels=self.birads_labels,
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("BI-RADS Confusion Matrix")
            plt.tight_layout()
            cm_plot_path = os.path.join(
                self.plot_dir, f"{self.name}_birads_confusion_matrix.png"
            )
            plt.savefig(cm_plot_path)
            plt.close()

            # ROC Curve
            birads_auc = None
            if self.final_birads_logits:
                # Concatenate list of batch logits into a single array
                all_birads_logits_np = np.concatenate(self.final_birads_logits, axis=0)
                # Calculate probabilities using softmax
                birads_probs = softmax(all_birads_logits_np, axis=1)
                # Plot ROC
                birads_auc = plot_roc_curve(
                    self.final_birads_targets,
                    birads_probs,
                    self.birads_labels,
                    "BI-RADS",
                    self.plot_dir,
                    self.name,
                )

            # Save report to text file (append AUC if calculated)
            report_path = os.path.join(
                self.plot_dir, f"{self.name}_birads_classification_report.txt"
            )
            with open(report_path, "w") as f:
                f.write("BI-RADS Classification Report\n")
                f.write("=" * 30 + "\n")
                f.write(birads_report)
                if birads_auc:
                    f.write("\n\n--- ROC AUC Scores ---\n")
                    f.write(f"Macro-average AUC: {birads_auc['macro']:.4f}\n")
                    f.write(f"Micro-average AUC: {birads_auc['micro']:.4f}\n")
                    for i, label in enumerate(self.birads_labels):
                        f.write(f"AUC for class {label}: {birads_auc[i]:.4f}\n")

        except Exception as e:
            print(f"Error generating BI-RADS metrics: {e}")
            import traceback

            traceback.print_exc()  # Print stack trace for debugging ROC issues

        # --- Density Metrics ---
        try:

            density_report = classification_report(
                self.final_density_targets,
                self.final_density_preds,
                target_names=self.density_labels,
                digits=4,
                zero_division=0,
            )

            # Confusion Matrix (code unchanged)
            cm_density = confusion_matrix(
                self.final_density_targets,
                self.final_density_preds,
                labels=range(len(self.density_labels)),
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_density,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=self.density_labels,
                yticklabels=self.density_labels,
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Density Confusion Matrix")
            plt.tight_layout()
            cm_plot_path = os.path.join(
                self.plot_dir, f"{self.name}_density_confusion_matrix.png"
            )
            plt.savefig(cm_plot_path)
            plt.close()

            # ROC Curve
            density_auc = None
            if self.final_density_logits:
                # Concatenate list of batch logits into a single array
                all_density_logits_np = np.concatenate(
                    self.final_density_logits, axis=0
                )
                # Calculate probabilities using softmax
                density_probs = softmax(all_density_logits_np, axis=1)
                # Plot ROC
                density_auc = plot_roc_curve(
                    self.final_density_targets,
                    density_probs,
                    self.density_labels,
                    "Density",
                    self.plot_dir,
                    self.name,
                )

            # Save report to text file (append AUC if calculated)
            report_path = os.path.join(
                self.plot_dir, f"{self.name}_density_classification_report.txt"
            )
            with open(report_path, "w") as f:
                f.write("Density Classification Report\n")
                f.write("=" * 30 + "\n")
                f.write(density_report)
                if density_auc:
                    f.write("\n\n--- ROC AUC Scores ---\n")
                    f.write(f"Macro-average AUC: {density_auc['macro']:.4f}\n")
                    f.write(f"Micro-average AUC: {density_auc['micro']:.4f}\n")
                    for i, label in enumerate(self.density_labels):
                        f.write(f"AUC for class {label}: {density_auc[i]:.4f}\n")

        except Exception as e:
            print(f"Error generating Density metrics: {e}")
            import traceback

            traceback.print_exc()

    def save_metrics_plots(self):
        """Saves plots for validation metrics (Precision and Recall)."""
        # Check if validation happened and scores exist using one of the lists
        if not self.val_loader or not self.birads_recall_scores:
            print("Skipping metrics plot generation: No validation data or scores.")
            return

        epochs = list(
            range(1, self.current_epoch + 2)
        )  # +1 for current epoch, +1 for range end

        num_val_epochs = len(self.val_losses)  # Use val_losses as the reference length
        epochs = epochs[:num_val_epochs]
        birads_prec = self.birads_precision_scores[:num_val_epochs]
        birads_rec = self.birads_recall_scores[:num_val_epochs]
        density_prec = self.density_precision_scores[:num_val_epochs]
        density_rec = self.density_recall_scores[:num_val_epochs]

        if not epochs:
            print("Skipping metrics plot generation: No epochs with validation data.")
            return

        # --- Plot 1: BiRADS Precision and Recall ---
        plt.figure(figsize=(8, 5))  # Create a new figure for BiRADS
        plt.plot(
            epochs,
            birads_prec,
            "b-o",
            label="BiRADS Precision",
        )
        plt.plot(
            epochs,
            birads_rec,
            "r-^",  # Different marker for recall
            label="BiRADS Recall",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score (Macro)")
        plt.title("BiRADS Classification - Precision & Recall")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # Keep y-axis between 0 and 1
        plt.tight_layout()
        birads_plot_path = os.path.join(
            self.plot_dir, f"{self.name}_birads_precision_recall.png"
        )
        plt.savefig(birads_plot_path)
        # print(f"BiRADS Precision/Recall plot saved to {birads_plot_path}")
        plt.close()

        # --- Plot 2: Density Precision and Recall ---
        plt.figure(figsize=(8, 5))  # Create a new figure for Density
        plt.plot(
            epochs,
            density_prec,
            "g-o",
            label="Density Precision",
        )
        plt.plot(
            epochs,
            density_rec,
            "m-^",  # Different marker for recall
            label="Density Recall",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Score (Macro)")
        plt.title("Breast Density Classification - Precision & Recall")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # Keep y-axis between 0 and 1
        plt.tight_layout()
        density_plot_path = os.path.join(
            self.plot_dir, f"{self.name}_density_precision_recall.png"
        )
        plt.savefig(density_plot_path)
        # print(f"Density Precision/Recall plot saved to {density_plot_path}")
        plt.close()  # Close the figure
