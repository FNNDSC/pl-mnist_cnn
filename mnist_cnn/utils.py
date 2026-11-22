"""Devices, seeds, checkpoints and image preparation."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from .data import MNIST_MEAN_STD


def get_device(pref: str = "auto") -> torch.device:
    """Pick the device asked for, or the best one available; fall back to the CPU."""
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every random number generator a training run draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(correct: int, total: int) -> float:
    """Percentage correct, 0 when there is nothing to count."""
    return 100.0 * correct / max(1, total)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Write the model's weights to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


def load_checkpoint(
    model: nn.Module, path: str | Path, map_location: torch.device | None = None
) -> nn.Module:
    """Load weights written by ``save_checkpoint`` into ``model``."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return model


def resolve_path(p: str | None, inputdir: Path) -> Path | None:
    """Resolve a path given on the command line; a relative one is under ``inputdir``."""
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else inputdir / path


def preprocess_image(path: Path) -> torch.Tensor:
    """Read an image as the network expects it: greyscale, 28x28, normalized, shape [1, 1, 28, 28]."""
    mean, std = MNIST_MEAN_STD
    tfm = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return tfm(Image.open(path).convert("L")).unsqueeze(0)


@torch.no_grad()
def predict_image(model: nn.Module, path: Path, device: torch.device) -> tuple[int, float]:
    """Classify one image. Returns the digit and the model's confidence in it, as a percentage."""
    model.eval()
    probs = torch.softmax(model(preprocess_image(path).to(device)), dim=1).squeeze(0)
    digit = int(torch.argmax(probs).item())
    return digit, float(probs[digit].item()) * 100.0
