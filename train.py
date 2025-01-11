import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.epochs = epochs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
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
                train_loss = 0.0
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
                    train_loss += losses.item()
                    current_loss = train_loss / (train_loader.n + 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    train_loader.set_postfix(
                        {"Loss": f"{current_loss:.4f}", "LR": f"{lr:.6f}"}
                    )
                train_loss /= self.train_loader.batch_size
                self.train_losses.append(train_loss)
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
                self.scheduler.step()
                print(
                    f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss if self.val_loader else "N/A"}'
                )
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
            val_loader = tqdm(self.val_loader, desc="Validation", leave=False)
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

                detections, birads_probs, density_probs = self.model(images)
                birad_preds.extend(birads_probs.argmax(dim=1).cpu().numpy())
                birad_targets.extend(
                    torch.stack([t["birads"] for t in targets]).flatten().cpu().numpy()
                )
                density_preds.extend(density_probs.argmax(dim=1).cpu().numpy())
                density_targets.extend(
                    torch.stack([t["density"] for t in targets]).flatten().cpu().numpy()
                )

                birad_results = evaluate_classification(birad_preds, birad_targets)
                density_results = evaluate_classification(
                    density_preds, density_targets, task="density"
                )

                loss = self.eval_loss(birads_probs, density_probs, targets)

                birad_f1 = birad_results["f1"]
                density_f1 = density_results["f1"]

                val_loss += loss.item() / self.val_loader.batch_size

                val_loader.set_postfix(
                    {
                        "Loss": f"{loss:.4f}",
                        "BIRADS_F1": f"{birad_f1:.4f}",
                        "Density_F1": f"{density_f1:.4f}",
                    }
                )

        self.val_losses.append(val_loss)

        self.model.train()

        return val_loss

    def eval_loss(self, birads_logits, density_logits, targets) -> torch.Tensor:
        """Compute detection and classification losses during evaluation"""
        loss_dict = {}

        birads_targets = torch.stack([t["birads"] for t in targets]).flatten().long()
        density_targets = torch.stack([t["density"] for t in targets]).flatten().long()

        birads_loss = nn.CrossEntropyLoss()(birads_logits, birads_targets)
        density_loss = nn.CrossEntropyLoss()(density_logits, density_targets)

        loss_dict["birads_loss"] = birads_loss
        loss_dict["density_loss"] = density_loss

        return sum(loss for loss in loss_dict.values())

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
