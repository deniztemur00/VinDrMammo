import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation import evaluate_classification
from torchmetrics.detection import MeanAveragePrecision


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 10,
        lr: float = 1e-3,
        save_dir: str = "models/",
        name: str = None,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.box_loss = nn.SmoothL1Loss(
            beta=1.0 / 9.0
        )  # beta=1/9 as per the original paper

        self.map_metric = MeanAveragePrecision(class_metrics=True)
        self.detection_cls_loss = nn.CrossEntropyLoss()
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

        # self.scheduler = OneCycleLR(
        #    self.optimizer,
        #    max_lr=4e-3,
        #    epochs=self.epochs,
        #    steps_per_epoch=len(self.train_loader),
        # )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(train_loader),
            eta_min=1e-6,
        )

        self.name = name if name else ""
        self.train_losses = []
        self.val_losses = []

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

    def train(self):
        best_loss = float("inf")
        torch.autograd.set_detect_anomaly(True)
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
                    ]  # Process each target dict

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    self.optimizer.zero_grad()
                    losses.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)

                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

                    self.optimizer.step()
                    self.scheduler.step()

                    current_loss = losses.item()
                    epoch_loss += current_loss
                    lr = self.scheduler.get_last_lr()[0]

                    train_loader.set_postfix(
                        {
                            "cls_loss": f"{loss_dict['classification']:.4f}",
                            "box_loss": f"{loss_dict['bbox_regression']:.4f}",
                            # "objectness_loss": f"{loss_dict['loss_objectness']:.4f}",
                            # "rpn_box_loss": f"{loss_dict['loss_rpn_box_reg']:.4f}",
                            "birads_loss": f"{loss_dict['birads_loss']:.4f}",
                            "density_loss": f"{loss_dict['density_loss']:.4f}",
                            "LR": f"{lr:.5f}",
                        }
                    )

                epoch_loss /= len(self.train_loader)
                self.train_losses.append(epoch_loss)

                if self.val_loader:
                    val_loss = self.validate()
                    if val_loss < best_loss:
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
            print("Training interrupted. Saving the model...")
            self.save(os.path.join(self.save_dir, "interrupted_model.pth"))
            self.save_loss_plot()
            return

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        val_loss = 0.0
        birad_preds = []
        birad_targets = []
        density_preds = []
        density_targets = []
        batch_maps = []

        with torch.no_grad():
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

                detections, birads_logits, density_logits = self.model(images)

                birad_preds.extend(birads_logits.argmax(dim=-1).cpu().numpy())
                birad_targets.extend(
                    torch.stack([t["birads"] for t in targets]).flatten().cpu().numpy()
                )

                density_preds.extend(density_logits.argmax(dim=-1).cpu().numpy())
                density_targets.extend(
                    torch.stack([t["density"] for t in targets]).flatten().cpu().numpy()
                )

                birad_results = evaluate_classification(birad_preds, birad_targets)
                density_results = evaluate_classification(
                    density_preds, density_targets, task="density"
                )

                loss, loss_dict = self.eval_loss(
                    detections, birads_logits, density_logits, targets
                )

                birad_f1 = birad_results["f1"]
                density_f1 = density_results["f1"]

                detections = self.sanitize_detections(detections)
                self.map_metric.update(
                    preds=detections,
                    target=targets,
                )
                map_results = self.map_metric.compute()

                current_map = map_results["map"].item()
                current_map_50 = map_results["map_50"].item()

                batch_maps.append(current_map)
                current_loss = loss.item()
                val_loss += current_loss

                val_loader.set_postfix(
                    {
                        "total_loss": f"{current_loss:.4f}",
                        "bbox_l1": f"{loss_dict['detection_loss']:.4f}",
                        "mAP": f"{current_map:.4f}",
                        "mAP@50": f"{current_map_50:.4f}",
                        "birads_loss": f"{loss_dict['birads_loss']:.4f}",
                        "density_loss": f"{loss_dict['density_loss']:.4f}",
                        "birads_f1": f"{birad_f1:.4f}",
                        "density_f1": f"{density_f1:.4f}",
                    }
                )

        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)

        self.model.train()
        self.birad_preds = birad_preds
        self.birad_targets = birad_targets
        self.density_preds = density_preds
        self.density_targets = density_targets
        self.batch_maps = batch_maps

        return val_loss

    def sanitize_detections(self, detections):
        for det in detections:
            boxes = det["boxes"]
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths > 0) & (heights > 0) & (det["scores"] > 0.05)
            det["boxes"] = boxes[valid]
            det["scores"] = det["scores"][valid]
            det["labels"] = det["labels"][valid]
        return detections

    def get_k_best_scores(self, detections, targets, k: int = 3):
        detection_loss = torch.tensor(0.0, device=self.device)

        for det, target in zip(detections, targets):
            if len(det["boxes"]) == 0:
                continue

            scores = det["scores"]
            if len(scores) > k:
                top_k_scores, top_k_indices = torch.topk(scores, k)
                k_boxes = det["boxes"][top_k_indices]
                k_labels = det["labels"][top_k_indices].float()
            else:
                k_boxes = det["boxes"]
                k_labels = det["labels"].float()

            target_boxes = target["boxes"]
            target_labels = target["labels"].float()

            if len(target_boxes) > 0:
                try:
                    detection_loss += self.box_loss(
                        k_boxes, target_boxes[: len(k_boxes)]
                    )

                except RuntimeError as e:
                    print(f"Error in loss calculation: {str(e)}")
                    print(f"k_labels shape: {k_labels.shape}")
                    print(f"target_labels shape: {target_labels.shape}")
                    continue

        batch_size = len(detections)
        if batch_size > 0:
            detection_loss = detection_loss / batch_size

        return detection_loss

    def eval_loss(self, detections, birads_logits, density_logits, targets):
        loss_dict = {}

        
        birads_targets = torch.stack([t["birads"] for t in targets]).flatten().long()
        density_targets = torch.stack([t["density"] for t in targets]).flatten().long()

        
        detection_loss = self.get_k_best_scores(detections, targets)

        
        birads_loss = self.birads_loss(birads_logits, birads_targets)
        density_loss = self.density_loss(density_logits, density_targets)

        
        loss_dict.update(
            {
                "detection_loss": detection_loss,
                "birads_loss": birads_loss,
                "density_loss": density_loss,
            }
        )

        return sum(loss for loss in loss_dict.values()), loss_dict

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
