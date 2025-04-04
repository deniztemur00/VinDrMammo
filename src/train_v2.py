import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation import evaluate_classification
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms, box_iou


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 10,
        save_dir: str = "models/",
        name: str = None,
    ):
        self.model = model
        self.param_groups = [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.detector.head.parameters(), "lr": 1e-3},
            # {"params": model.detector.parameters(), "lr": 1e-4},  # Detection head
            {"params": model.birads_head.parameters(), "lr": 5e-4},  # Auxiliary heads
            {"params": model.density_head.parameters(), "lr": 2e-4},  # Auxiliary heads
        ]
        self.optimizer = torch.optim.AdamW(self.param_groups, weight_decay=0.01)
        self.box_loss = nn.SmoothL1Loss()

        self.map_metric = MeanAveragePrecision(
            iou_type="bbox",
            class_metrics=True,
        )
        self.birads_loss = nn.CrossEntropyLoss()
        self.density_loss = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-6,
        )
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)

        self.name = name if name else ""
        self.train_losses = []
        self.val_losses = []

    def train(self):
        best_loss = float("inf")
        try:
            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0
                train_loader = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.epochs} - Training",
                )
                for images, targets in train_loader:
                    images = [img.to(self.device) for img in images]

                    targets = [
                        {
                            k: v.to(self.device)
                            for k, v in t.items()
                            if isinstance(v, torch.Tensor)
                        }
                        for t in targets
                    ]

                    loss_dict = self.model(images, targets)
                    total_loss = loss_dict["total_loss"]

                    self.optimizer.zero_grad()
                    total_loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)

                    self.optimizer.step()
                    # self.scheduler.step()

                    current_loss = loss_dict["total_loss"].item()

                    epoch_loss += current_loss
                    lr = self.scheduler.get_last_lr()[0]

                    # Update progress bar
                    train_loader.set_postfix(
                        {
                            "cls_loss": f"{loss_dict['classification'].item():.4f}",
                            "bbox_loss": f"{loss_dict['box_reg'].item():.4f}",
                            "birads_loss": f"{loss_dict['birads_loss'].item():.4f}",
                            "density_loss": f"{loss_dict['density_loss'].item():.4f}",
                            "avg_curr_loss": f"{current_loss:.4f}",
                            "LR": f"{lr:.5f}",
                        }
                    )
                self.scheduler.step()
                epoch_loss /= len(self.train_loader)
                self.train_losses.append(epoch_loss)

                # Validation
                if self.val_loader:
                    val_loss, _ = self.validate()
                    if val_loss < best_loss:
                        print(
                            f"Validation loss improved from {best_loss:.4f} to {val_loss:.4f}"
                        )
                        print("Saving model...")
                        best_loss = val_loss
                        self.save(
                            os.path.join(self.save_dir, f"{self.name}_best_model.pth")
                        )
                else:
                    self.save(
                        os.path.join(self.save_dir, f"{self.name}_epoch_{epoch}.pth")
                    )

            self.save_loss_plot()
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            self.save(os.path.join(self.save_dir, f"{self.name}_interrupted_model.pth"))
            self.save_loss_plot()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        birad_preds = []
        birad_targets = []
        density_preds = []
        density_targets = []
        detections_list = []
        targets_list = []

        val_loader = tqdm(self.val_loader, desc="Validation")
        for images, targets in val_loader:
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device)
                    for k, v in t.items()
                    if isinstance(v, torch.Tensor)
                }
                for t in targets
            ]
            # Forward pass
            outputs = self.model(images)
            detections = outputs["detections"]

            detections = self.keep_detections(detections)
            birads_logits = outputs["birads_logits"]
            density_logits = outputs["density_logits"]

            # Calculate losses
            loss, loss_dict = self.eval_loss_multi(
                detections, birads_logits, density_logits, targets
            )
            val_loss += loss.item()

            # Metrics
            birad_preds.extend(torch.argmax(birads_logits, dim=1).cpu().numpy())
            birad_targets.extend(
                torch.stack([t["birads"] for t in targets]).cpu().numpy()
            )

            density_preds.extend(torch.argmax(density_logits, dim=1).cpu().numpy())
            density_targets.extend(
                torch.stack([t["density"] for t in targets]).cpu().numpy()
            )

            # Update detection metrics

            detections = self.sanitize_detections(detections)

            self.map_metric.update(detections, targets)
            map_results = self.map_metric.compute()

            detections_list.extend(detections)
            targets_list.extend(targets)

            birad_results = evaluate_classification(birad_preds, birad_targets)
            density_results = evaluate_classification(
                density_preds, density_targets, task="density"
            )

            # Update progress bar
            val_loader.set_postfix(
                {
                    "mAP": f"{map_results['map'].item():.4f}",
                    "birads_f1": f"{birad_results['f1']:.4f}",
                    "density_f1": f"{density_results['f1']:.4f}",
                    "det_loss": f"{loss_dict['detection_loss'].item():.4f}",
                    "birads_loss": f"{loss_dict['birads_loss'].item():.4f}",
                    "density_loss": f"{loss_dict['density_loss'].item():.4f}",
                    "avg_curr_loss": f"{loss.item():.4f}",
                }
            )

        # Calculate final metrics
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        # Classification metrics

        # Detection metrics
        map_results = self.map_metric.compute()
        print("\n", "*" * 50, "\n")

        result_dict = {
            "val_loss": val_loss,
            "birad_results": birad_preds,
            "birad_targets": birad_targets,
            "density_results": density_preds,
            "density_targets": density_targets,
            "detections": detections_list,
            "targets": targets_list,
            "map_results": map_results,
        }
        self.result_dict = result_dict

        return val_loss, result_dict

    def keep_detections(self, detections):
        for det in detections:
            if len(det["boxes"]) == 0:
                continue

            # Apply NMS (critical for mAP with multiple predictions)
            keep = nms(
                boxes=det["boxes"],
                scores=det["scores"],
                iou_threshold=0.5,  # Match your RetinaNet config
            )
            for key in ["boxes", "scores", "labels"]:
                det[key] = det[key][keep]

            # Filter by confidence (optional but recommended)
            conf_mask = det["scores"] > 0.05  # Adjust threshold
            for key in ["boxes", "scores", "labels"]:
                det[key] = det[key][conf_mask]

        return detections

    def sanitize_detections(self, detections):
        for det in detections:
            if len(det["boxes"]) > 0:
                valid = (det["boxes"][:, 2] > det["boxes"][:, 0]) & (
                    det["boxes"][:, 3] > det["boxes"][:, 1]
                )
                for key in ["boxes", "scores", "labels"]:
                    det[key] = det[key][valid]
        return detections

    def eval_loss_multi(self, detections, birads_logits, density_logits, targets):
        # Detection loss (Smooth L1 on boxes)
        detection_loss = torch.tensor(0.0, device=self.device)
        total_boxes = 0

        for det, tgt in zip(detections, targets):
            if len(det["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                continue

            # Calculate IoU matrix between predictions and targets
            ious = box_iou(det["boxes"], tgt["boxes"])

            if ious.numel() > 0:
                # For each ground truth box, find best matching prediction
                best_match_idx = ious.argmax(dim=0)

                # Calculate loss only for the best-matching prediction for each GT box
                matched_boxes = det["boxes"][best_match_idx]
                box_loss = self.box_loss(matched_boxes, tgt["boxes"])

                # Add to total loss, weighted by number of boxes
                detection_loss += box_loss * len(tgt["boxes"])
                total_boxes += len(tgt["boxes"])

        # Classification losses
        birads_targets = torch.stack([t["birads"] for t in targets]).long().flatten()
        density_targets = torch.stack([t["density"] for t in targets]).long().flatten()

        birads_loss = self.birads_loss(birads_logits, birads_targets)
        density_loss = self.density_loss(density_logits, density_targets)

        # Normalize detection loss by total number of ground truth boxes
        if total_boxes > 0:
            detection_loss /= total_boxes

        # Combine losses
        total_loss = detection_loss + (birads_loss * 0.5)  # + density_loss * 0.3)

        return total_loss, {
            "detection_loss": detection_loss,
            "birads_loss": birads_loss,
            "density_loss": density_loss,
        }

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

    def save_loss_plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f"{self.name}_losses.png"))
        plt.close()
