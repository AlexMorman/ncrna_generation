"""utils.py — Shared helpers (config loading, seeding, plotting)."""

import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

NUCLEOTIDES = ["A", "U", "G", "C"]


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
    val_accs: list[float],
    save_path: str,
) -> None:
    """Plot and save training/validation loss curves and validation accuracy.

    Two-panel figure:

    - **Top:** train + val cross-entropy loss with a random-baseline reference
      line at ``ln(4) ≈ 1.386`` (4-class uniform distribution).
    - **Bottom:** validation accuracy with a random-baseline reference at 0.25.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        val_accs:     Per-epoch validation accuracies (0–1).
        save_path:    Output image path (e.g. ``"loss_curves.png"``).
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # ── Loss panel ──────────────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.axhline(
        y=np.log(4), color="gray", linestyle="--", alpha=0.7,
        label=f"Random baseline (ln 4 ≈ {np.log(4):.3f})",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy panel ──────────────────────────────────────────────────────
    ax2.plot(epochs, val_accs, label="Val Accuracy", color="green")
    ax2.axhline(
        y=0.25, color="gray", linestyle="--", alpha=0.7,
        label="Random baseline (0.25)",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_diagnostics(
    conf_matrix: np.ndarray,
    pred_counts: list[int],
    target_counts: list[int],
    bp_rate: float,
    save_path: str,
) -> None:
    """Plot a 2×2 post-training diagnostics figure saved to *save_path*.

    Panels:

    - **Top-left:**    Confusion matrix (row-normalised) with raw counts.
    - **Top-right:**   Per-class accuracy bar chart (A / U / G / C) vs random
      baseline of 0.25.
    - **Bottom-left:** Nucleotide frequency — ground-truth vs model predictions.
    - **Bottom-right:** Base-pair satisfaction rate vs random baseline and
      perfect score.

    Args:
        conf_matrix:   ``(4, 4)`` int array where ``conf_matrix[true][pred]``
                       is the count of samples with true class *true* predicted
                       as *pred*.  Row order: A=0, U=1, G=2, C=3.
        pred_counts:   ``[A, U, G, C]`` total predicted nucleotide counts over
                       the validation set.
        target_counts: ``[A, U, G, C]`` total ground-truth nucleotide counts.
        bp_rate:       Fraction of base-paired positions where the two predicted
                       nucleotides form a valid Watson-Crick or G-U wobble pair.
        save_path:     Output image path (e.g. ``"diagnostics.png"``).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Confusion matrix (top-left) ─────────────────────────────────────────
    ax = axes[0, 0]
    row_sums = conf_matrix.sum(axis=1, keepdims=True).clip(min=1)
    normalized = conf_matrix / row_sums

    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(NUCLEOTIDES)
    ax.set_yticklabels(NUCLEOTIDES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i,
                f"{normalized[i, j]:.2f}\n({conf_matrix[i, j]:,})",
                ha="center", va="center", fontsize=8,
                color="white" if normalized[i, j] > 0.6 else "black",
            )

    # ── Per-class accuracy (top-right) ──────────────────────────────────────
    ax = axes[0, 1]
    per_class_acc = [
        conf_matrix[i, i] / max(int(conf_matrix[i].sum()), 1)
        for i in range(4)
    ]
    colours = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
    bars = ax.bar(NUCLEOTIDES, per_class_acc, color=colours)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Random baseline")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Nucleotide")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy (val set)")
    ax.legend()
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    # ── Nucleotide frequency (bottom-left) ──────────────────────────────────
    ax = axes[1, 0]
    total_pred = max(sum(pred_counts), 1)
    total_target = max(sum(target_counts), 1)
    pred_freq = [c / total_pred for c in pred_counts]
    target_freq = [c / total_target for c in target_counts]

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2, target_freq, width, label="Target", color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, pred_freq, width, label="Predicted", color="#2196F3", alpha=0.8)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Uniform (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(NUCLEOTIDES)
    ax.set_xlabel("Nucleotide")
    ax.set_ylabel("Frequency")
    ax.set_title("Nucleotide Frequency: Target vs Predicted")
    ax.legend()

    # ── Base-pair satisfaction (bottom-right) ────────────────────────────────
    # Random baseline: 6 valid ordered pairs out of 16 possible.
    # Valid Watson-Crick + wobble: (A,U),(U,A),(G,C),(C,G),(G,U),(U,G)
    random_bp_baseline = 6 / 16  # = 0.375
    ax = axes[1, 1]
    labels = ["Random\nBaseline", "Model", "Perfect"]
    values = [random_bp_baseline, bp_rate, 1.0]
    bar_colours = ["#9E9E9E", "#2196F3", "#4CAF50"]
    bars = ax.bar(labels, values, color=bar_colours, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of valid Watson-Crick pairs")
    ax.set_title("Base-Pair Satisfaction Rate")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
