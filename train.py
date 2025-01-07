import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        device: str = "cpu",
        epochs: int = 10,
        lr: float = 1e-3,
        save_dir: str = "models/",
        name: str = None,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
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
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{self.name}_losses.png'))
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
                train_loss /= len(self.train_loader)
                self.train_losses.append(train_loss)
                if self.val_loader is not None:
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

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loader = tqdm(self.val_loader, desc="Validation", leave=False)
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                current_val_loss = val_loss / (val_loader.n + 1)
                val_loader.set_postfix({"Val Loss": f"{current_val_loss:.4f}"})
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        return val_loss

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
