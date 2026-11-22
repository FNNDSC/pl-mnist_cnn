"""The training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from .config import TrainConfig
from .utils import accuracy, get_device, save_checkpoint, set_seed


class Trainer:
    """Train a model, keeping the checkpoint with the best validation accuracy."""

    def __init__(self, model: nn.Module, lr: float, weight_decay: float,
                 device: torch.device, amp: bool, out_dir: str):
        self.device = device
        self.model  = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=amp)
        self.amp = amp
        self.best_acc = 0.0
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader, val_loader, epochs: int) -> dict[str, list[float]]:
        """Run ``epochs`` epochs; return the loss and accuracy of each."""
        history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch, epochs)
            val_loss, val_acc = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(val_loss)
            history["validation_accuracy"].append(val_acc)

            # Only save the best ckpt based on the val.
            if val_acc > self.best_acc or epoch == 1:
                self.best_acc = val_acc
                save_checkpoint(self.model, self.out_dir / "best.ckpt")
        return history

    def train_one_epoch(self, loader, epoch: int, epochs: int) -> float:
        """One pass over the training set; return its mean loss."""
        self.model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False, disable=None)
        for x, y in pbar:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running / len(loader.dataset)

    def evaluate(self, loader) -> tuple[float, float]:
        """Return the mean loss and the accuracy, as a percentage, over ``loader``."""
        return evaluate_model(self.model, loader, self.device, self.amp)


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader, device: torch.device, amp: bool = False
) -> tuple[float, float]:
    """Return the mean loss and the accuracy, as a percentage, of ``model`` over ``loader``."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), accuracy(correct, total)


def train_model(config: TrainConfig, loaders, model_cls) -> tuple[dict[str, list[float]], Path]:
    """Train ``model_cls`` as ``config`` says; return the history and the best checkpoint's path."""
    set_seed(config.seed)
    device = get_device(config.device)
    # Mixed precision is a GPU feature; asking for it on a CPU only produces warnings.
    amp = config.amp and device.type == "cuda"
    trainer = Trainer(model_cls(), config.lr, config.weight_decay, device, amp, config.out_dir)
    hist = trainer.train(*loaders, epochs=config.epochs)
    return hist, trainer.out_dir / "best.ckpt"
