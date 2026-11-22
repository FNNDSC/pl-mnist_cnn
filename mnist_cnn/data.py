"""Data loaders for the MNIST training and test sets."""

import tempfile
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN_STD: tuple[float, float] = (0.1307, 0.3081)

# The dataset is downloaded on first use. It is kept out of the plugin's output
# directory, so a training run's output holds the checkpoint and metrics only.
DEFAULT_DATA_ROOT: Path = Path(tempfile.gettempdir()) / "mnist_cnn" / "data"


def mnist_loaders(
    batch_size: int, num_workers: int = 2, root: Path = DEFAULT_DATA_ROOT
) -> tuple[DataLoader, DataLoader]:
    """Return the (train, test) loaders, downloading MNIST to ``root`` if it is not there."""
    mean, std = MNIST_MEAN_STD
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    root.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
