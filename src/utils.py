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
    val_perplexities: list[float],
    val_recoveries: list[float],
    save_path: str,
) -> None:
    """Plot and save training/validation loss curves, perplexity, and recovery.

    Three-panel figure:

    - **Top:**    Train + val cross-entropy loss with ``ln(4)`` random-baseline
      reference line.
    - **Middle:** Val perplexity (exp(val_loss)).
    - **Bottom:** Val strict recovery with a 0.25 random-baseline reference
      (uniform 4-class guess).

    Args:
        train_losses:     Per-epoch training losses.
        val_losses:       Per-epoch validation losses.
        val_perplexities: Per-epoch validation perplexities.
        val_recoveries:   Per-epoch validation recovery rates (0–1).
        save_path:        Output image path.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))

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

    # ── Perplexity panel ────────────────────────────────────────────────────
    ax2.plot(epochs, val_perplexities, label="Val Perplexity", color="orange")
    ax2.axhline(
        y=4.0, color="gray", linestyle="--", alpha=0.7,
        label="Random baseline (4.0)",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Recovery panel ──────────────────────────────────────────────────────
    ax3.plot(epochs, val_recoveries, label="Val Recovery", color="green")
    ax3.axhline(
        y=0.25, color="gray", linestyle="--", alpha=0.7,
        label="Random baseline (0.25)",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Recovery")
    ax3.set_title("Validation Strict Recovery")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_diagnostics(
    conf_matrix: np.ndarray,
    pred_counts: list[int],
    target_counts: list[int],
    save_path: str,
) -> None:
    """Plot a 3-panel post-training diagnostics figure saved to *save_path*.

    Panels:

    - **Left:**   Confusion matrix (row-normalised) with raw counts.
    - **Centre:** Per-class accuracy bar chart (A / U / G / C) vs 0.25 baseline.
    - **Right:**  Nucleotide frequency — ground-truth vs model predictions.

    Args:
        conf_matrix:   ``(4, 4)`` int array where ``conf_matrix[true][pred]``
                       is the count of samples with true class *true* predicted
                       as *pred*.  Row order: A=0, U=1, G=2, C=3.
        pred_counts:   ``[A, U, G, C]`` total predicted nucleotide counts.
        target_counts: ``[A, U, G, C]`` total ground-truth nucleotide counts.
        save_path:     Output image path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Confusion matrix ────────────────────────────────────────────────────
    ax = axes[0]
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

    # ── Per-class accuracy ──────────────────────────────────────────────────
    ax = axes[1]
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

    # ── Nucleotide frequency ────────────────────────────────────────────────
    ax = axes[2]
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_position_recovery(
    recoveries: list[float],
    consensus_structure: str,
    save_path: str,
) -> None:
    """Plot per-position strict recovery vs. consensus structure.

    The x-axis is annotated with the dot-bracket consensus structure
    characters so the viewer can relate recovery to structural context.

    Args:
        recoveries:          Per-position recovery rates (0–1), length N.
        consensus_structure: Dot-bracket string of length N.
        save_path:           Output image path.
    """
    n = len(recoveries)
    positions = list(range(n))

    fig, ax = plt.subplots(figsize=(max(10, n // 5), 5))
    ax.plot(positions, recoveries, color="#2196F3", linewidth=1.5, label="Recovery")
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Random baseline (0.25)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Consensus position")
    ax.set_ylabel("Strict recovery")
    ax.set_title("Per-Position Strict Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate x-axis with structure characters at a coarse stride
    if n <= 80:
        stride = 1
    elif n <= 200:
        stride = 5
    else:
        stride = 10
    tick_positions = list(range(0, n, stride))
    tick_labels = [consensus_structure[i] for i in tick_positions]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Structure character")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_recovery_histogram(
    per_window_recoveries: list[float],
    save_path: str,
) -> None:
    """Histogram of per-window mean recovery rates.

    Args:
        per_window_recoveries: Mean recovery rate for each evaluated window.
        save_path:             Output image path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_window_recoveries, bins=20, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(x=0.25, color="gray", linestyle="--", alpha=0.7, label="Random baseline (0.25)")
    ax.set_xlabel("Mean recovery per window")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Window Mean Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
