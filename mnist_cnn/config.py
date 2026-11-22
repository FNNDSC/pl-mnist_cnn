"""Default hyperparameters for training."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """What one training run needs. The defaults are the plugin's defaults."""

    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 1337
    num_workers: int = 2
    out_dir: str = "outputs"
    device: str = "auto"  # "auto", "cpu", "cuda" or "mps"
    amp: bool = False  # mixed precision; only useful on a GPU
    weight_decay: float = 0.0
