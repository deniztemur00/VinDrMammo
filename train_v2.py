import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        self.box_loss = nn.SmoothL1Loss(beta=1.0 / 9.0)

        self.map_metric = MeanAveragePrecision(class_metrics=True)
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
            T_max=epochs * len(train_loader),
            eta_min=1e-6,
        )

        self.name = name if name else ""
        self.train_losses = []
        self.val_losses = []
        self.aux_loss_weight = model.config.aux_loss_weight  # Get from model config

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
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

                    self.optimizer.step()
                    self.scheduler.step()

                    current_loss = total_loss.item()
                    epoch_loss += current_loss
                    lr = self.scheduler.get_last_lr()[0]

                    # Update progress bar
                    train_loader.set_postfix(
                        {
                            "cls_loss": f"{loss_dict['classification'].item():.4f}",
                            "reg_loss": f"{loss_dict['box_reg'].item():.4f}",
                            "birads_loss": f"{loss_dict['birads_loss'].item():.4f}",
                            "density_loss": f"{loss_dict['density_loss'].item():.4f}",
                            "total_loss": f"{current_loss:.4f}",
                            "LR": f"{lr:.5f}",
                        }
                    )

                epoch_loss /= len(self.train_loader)
                self.train_losses.append(epoch_loss)

                # Validation
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
            print("Training interrupted. Saving model...")
            self.save(os.path.join(self.save_dir, "interrupted_model.pth"))
            self.save_loss_plot()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        birad_preds = []
        birad_targets = []
        density_preds = []
        density_targets = []

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
            birads_logits = outputs["birads_logits"]
            density_logits = outputs["density_logits"]

            # Calculate losses
            loss, loss_dict = self.eval_loss(
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

            birad_results = evaluate_classification(birad_preds, birad_targets)
            density_results = evaluate_classification(density_preds, density_targets)

            # Update progress bar
            val_loader.set_postfix(
                {
                    "val_loss": f"{loss.item():.4f}",
                    "det_loss": f"{loss_dict['detection_loss'].item():.4f}",
                    "mAP": f"{map_results['map'].item():.4f}",
                    "birads_f1": f"{birad_results['f1']:.4f}",
                    "density_f1": f"{density_results['f1']:.4f}",
                    "birads_loss": f"{loss_dict['birads_loss'].item():.4f}",
                    "density_loss": f"{loss_dict['density_loss'].item():.4f}",
                }
            )

        # Calculate final metrics
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        # Classification metrics

        # Detection metrics
        map_results = self.map_metric.compute()
        print(f"\nValidation mAP: {map_results['map'].item():.4f}")
        self.birad_preds = birad_preds
        self.birad_targets = birad_targets
        self.density_preds = density_preds
        self.density_targets = density_targets
        self.map_list = map_results

        return val_loss

    def sanitize_detections(self, detections):
        for det in detections:
            if len(det["boxes"]) > 0:
                valid = (det["boxes"][:, 2] > det["boxes"][:, 0]) & (
                    det["boxes"][:, 3] > det["boxes"][:, 1]
                )
                for key in ["boxes", "scores", "labels"]:
                    det[key] = det[key][valid]
        return detections

    def eval_loss(self, detections, birads_logits, density_logits, targets):
        # Detection loss (Smooth L1 on boxes)
        detection_loss = torch.tensor(0.0, device=self.device)
        for det, tgt in zip(detections, targets):
            if len(det["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                continue
            detection_loss += self.box_loss(det["boxes"], tgt["boxes"])

        # Classification losses
        birads_targets = torch.stack([t["birads"] for t in targets]).long().flatten()
        density_targets = torch.stack([t["density"] for t in targets]).long().flatten()

        birads_loss = self.birads_loss(birads_logits, birads_targets)
        density_loss = self.density_loss(density_logits, density_targets)

        detection_loss /= len(detections)
        # Combine losses
        total_loss = detection_loss + self.aux_loss_weight * (
            birads_loss + density_loss
        )

        return total_loss, {
            "detection_loss": detection_loss,
            "birads_loss": birads_loss,
            "density_loss": density_loss,
        }

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
