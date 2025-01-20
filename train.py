import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation import evaluate_classification


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
        )  # beta=1/9 as per Faster R-CNN paper
        self.detection_cls_loss = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-2,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
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

                    self.optimizer.step()
                    self.scheduler.step()

                    current_loss = losses.item()
                    epoch_loss += current_loss
                    lr = self.scheduler.get_last_lr()[0]

                    train_loader.set_postfix(
                        {
                            "cls_loss": f"{loss_dict['loss_classifier']:.4f}",
                            "box_loss": f"{loss_dict['loss_box_reg']:.4f}",
                            "objectness_loss": f"{loss_dict['loss_objectness']:.4f}",
                            "rpn_box_loss": f"{loss_dict['loss_rpn_box_reg']:.4f}",
                            "birads_loss": f"{loss_dict['birads_loss']:.4f}",
                            "density_loss": f"{loss_dict['density_loss']:.4f}",
                            "LR": f"{lr:.4f}",
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
                ]  # Process each target dict

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

                current_loss = loss.item()
                val_loss += current_loss

                val_loader.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "bbox_l1": f"{loss_dict['detection_loss']:.4f}",
                        "bbox_cls": f"{loss_dict['cls_loss']:.4f}",
                        "birads": f"{loss_dict['birads_loss']:.4f}",
                        "density": f"{loss_dict['density_loss']:.4f}",
                        "BIRADS_f1": f"{birad_f1:.4f}",
                        "Density_f1": f"{density_f1:.4f}",
                    }
                )

        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)

        self.model.train()

        return val_loss

    def get_k_best_scores(self, detections, targets, k: int = 3):
        detection_loss = 0
        cls_loss = 0

        for det, target in zip(detections, targets):
            if len(det["boxes"]) == 0:
                continue

            scores = det["scores"]

            if len(scores) > k:
                top_k_scores, top_k_indices = torch.topk(scores, k)
                k_boxes = det["boxes"][top_k_indices]
                k_labes = det["labels"][top_k_indices]

            else:
                k_boxes = det["boxes"]
                k_labes = det["labels"]

            target_boxes = target["boxes"]
            target_labels = target["labels"]

            print(k_boxes, target_boxes)
            print("*" * 50)
            print(k_labes, target_labels)

            if len(target_boxes) > 0:
                detection_loss += self.box_loss(k_boxes, target_boxes)

                cls_loss += self.detection_cls_loss(k_labes, target_labels)

        return detection_loss, cls_loss

    def eval_loss(
        self, detections, birads_logits, density_logits, targets
    ) -> torch.Tensor:
        """Compute detection and classification losses during evaluation"""
        loss_dict = {}
        birads_targets = torch.stack([t["birads"] for t in targets]).flatten().long()
        density_targets = torch.stack([t["density"] for t in targets]).flatten().long()

        detection_loss, cls_loss = self.get_k_best_scores(detections, targets)

        birads_loss = nn.CrossEntropyLoss()(birads_logits, birads_targets)
        density_loss = nn.CrossEntropyLoss()(density_logits, density_targets)

        loss_dict["detection_loss"] = detection_loss
        loss_dict["cls_loss"] = cls_loss
        loss_dict["birads_loss"] = birads_loss
        loss_dict["density_loss"] = density_loss

        return sum(loss for loss in loss_dict.values()), loss_dict

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
