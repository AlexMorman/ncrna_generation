"""utils.py — Shared helpers (config loading, seeding, plotting)."""

import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Filesystem path to the ``.yaml`` file.

    Returns:
        Parsed configuration as a nested dict.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
) -> None:
    """Plot and save training / validation loss curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        save_path:    Output image path (e.g. ``"loss_curves.png"``).
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
