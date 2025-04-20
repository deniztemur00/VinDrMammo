import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.01
    model_dir: str = "models/"
    plot_dir: str = "plots/"
    name: str = None
    birads_loss_weight: float = 0.6
    density_loss_weight: float = 0.4  # Example weight


class ClassificationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: TrainerConfig,
    ):
        self.model = model
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
        self.model.to(self.device)

        # Optimizer - adjust param_groups if your model has distinct parts with different LR needs
        # Simple optimizer for all parameters:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        # Example with param_groups if needed (adjust based on your model structure):
        # self.param_groups = [
        #     {"params": model.backbone.parameters(), "lr": lr * 0.1},
        #     {"params": model.birads_head.parameters(), "lr": lr},
        #     {"params": model.density_head.parameters(), "lr": lr},
        # ]
        # self.optimizer = torch.optim.AdamW(self.param_groups, weight_decay=weight_decay)

        # Loss functions
        self.birads_loss_fn = nn.CrossEntropyLoss()
        self.density_loss_fn = nn.CrossEntropyLoss()

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,  # T_max often set to total epochs
            eta_min=1e-6,
        )

        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.birads_f1_scores = []
        self.density_f1_scores = []
        self.current_epoch = 0

        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

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

    def train(self):
        best_val_loss = float("inf")
        try:
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.model.train()
                epoch_loss = 0.0
                epoch_birads_loss = 0.0  # Accumulate birads loss
                epoch_density_loss = 0.0  # Accumulate density loss
                batch_count = 0
                train_pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.epochs} - Training",
                    leave=False,
                )

                for images, targets in train_pbar:

                    outputs = self.model(images)

                    loss, loss_dict = self._calculate_loss(outputs, targets)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Optional: Gradient clipping
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_birads_loss += loss_dict["birads_loss"].item()
                    epoch_density_loss += loss_dict["density_loss"].item()
                    batch_count += 1

                    # Update progress bar postfix
                    lr = self.scheduler.get_last_lr()[0]  # Get current LR
                    train_pbar.set_postfix(
                        {
                            "avg_birads": f"{epoch_birads_loss / batch_count:.4f}",
                            "avg_density": f"{epoch_density_loss / batch_count:.4f}",
                            "avg_total": f"{epoch_loss / batch_count:.4f}",
                            "LR": f"{lr:.6f}",
                        }
                    )

                # Step the scheduler after each epoch
                self.scheduler.step()

                avg_epoch_loss = epoch_loss / len(self.train_loader)
                self.train_losses.append(avg_epoch_loss)
                print(
                    f"Epoch {epoch+1}/{self.epochs} - Avg Training Loss: {avg_epoch_loss:.4f}"
                )
                self.save_loss_plot()

                # Validation
                if self.val_loader:
                    val_loss, metrics = self.validate()
                    self.val_losses.append(val_loss)
                    self.birads_f1_scores.append(metrics["birads_f1"])
                    self.density_f1_scores.append(metrics["density_f1"])
                    self.save_metrics_plots()  # Save plots each epoch

                    print(
                        f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {val_loss:.4f}, "
                        f"BiRADS F1: {metrics['birads_f1']:.4f}, Density F1: {metrics['density_f1']:.4f}"
                    )

                    if val_loss < best_val_loss:
                        print(
                            f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model..."
                        )
                        best_val_loss = val_loss
                        self.save(
                            os.path.join(self.model_dir, f"{self.name}_best_model.pth")
                        )
                else:
                    # Save model every epoch if no validation
                    self.save(
                        os.path.join(self.model_dir, f"{self.name}_epoch_{epoch+1}.pth")
                    )

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

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        val_birads_loss = 0.0
        val_density_loss = 0.0
        val_batch_count = 0
        all_birads_preds = []
        all_birads_targets = []
        all_density_preds = []
        all_density_targets = []

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs} - Validation",
            leave=False,
        )

        for images, targets in val_pbar:

            outputs = self.model(images)

            loss, loss_dict = self._calculate_loss(outputs, targets)
            total_val_loss += (
                loss.item()
            )  # Keep accumulating total loss for final epoch average
            val_birads_loss += loss_dict["birads_loss"].item()
            val_density_loss += loss_dict["density_loss"].item()
            val_batch_count += 1

            # Get predictions
            birads_preds = torch.argmax(outputs["birads_logits"], dim=1)
            density_preds = torch.argmax(outputs["density_logits"], dim=1)

            # Store predictions and targets for metric calculation
            all_birads_preds.extend(birads_preds.cpu().numpy())
            all_birads_targets.extend(targets["birads"].cpu().numpy())
            all_density_preds.extend(density_preds.cpu().numpy())
            all_density_targets.extend(targets["density"].cpu().numpy())

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
                    "avg_birads_loss": f"{val_birads_loss / val_batch_count:.4f}",
                    "avg_density_loss": f"{val_density_loss/ val_batch_count:.4f}",
                    "avg_total_loss": f"{total_val_loss / val_batch_count:.4f}",  # Use total_val_loss here as it's already accumulated
                }
            )

        avg_val_loss = total_val_loss / len(self.val_loader)

        # Or using sklearn directly:
        birads_f1 = f1_score(
            all_birads_targets, all_birads_preds, average="macro", zero_division=0
        )
        density_f1 = f1_score(
            all_density_targets, all_density_preds, average="macro", zero_division=0
        )

        metrics = {
            "birads_f1": birads_f1,  # Get F1 or default to 0
            "density_f1": density_f1,  # Get F1 or default to 0
            # Add other metrics from evaluate_classification if needed
        }

        return avg_val_loss, metrics

    def save(self, filename):
        """Saves the model state dictionary."""
        print(f"Saving model to {filename}")
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Loads the model state dictionary."""
        print(f"Loading model from {filename}")
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.to(
            self.device
        )  # Ensure model is on the correct device after loading

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
        print(f"Loss plots saved to {plot_path}")
        plt.close()

    def save_metrics_plots(self):
        """Saves plots for validation metrics (F1 scores)."""
        if (
            not self.val_loader or not self.birads_f1_scores
        ):  # Check if validation happened
            return

        epochs = list(
            range(1, self.current_epoch + 2)
        )  # +1 for current epoch, +1 for range end

        plt.figure(figsize=(12, 5))

        # Plot BiRADS F1
        plt.subplot(1, 2, 1)
        plt.plot(
            epochs[: len(self.birads_f1_scores)],
            self.birads_f1_scores,
            "b-o",
            label="BiRADS F1 Score",
        )
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score (Macro)")
        plt.title("BiRADS Classification F1 Score")
        plt.grid(True)
        plt.legend()

        # Plot Density F1
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs[: len(self.density_f1_scores)],
            self.density_f1_scores,
            "g-o",
            label="Density F1 Score",
        )
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score (Macro)")
        plt.title("Breast Density Classification F1 Score")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f"{self.name}_f1_scores.png")
        plt.savefig(plot_path)
        print(f"Metric plots saved to {plot_path}")
        plt.close()
