"""train.py — Training loop for the pure-GNN ncRNA sequence designer.

Usage::

    python train.py --config configs/config.yaml

Reads ``.ss.ct`` structure-sequence pairs via :class:`RNAGraphDataset`,
trains the :class:`NcRNADesigner` model, saves the best checkpoint to
``best_model.pt``, and writes:

- ``loss_curves.png``  — train/val loss, val perplexity, val recovery.
- ``diagnostics.png``  — confusion matrix, per-class accuracy, nucleotide
  frequency (3-panel).
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.dataset import RNAGraphDataset
from src.model import build_model
from src.utils import load_config, plot_diagnostics, plot_loss_curves, set_seed


# ── Data + model setup ──────────────────────────────────────────────────────


def setup_data(config: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and validation PyG DataLoaders from config.

    Args:
        config: Parsed YAML configuration.

    Returns:
        ``(train_loader, val_loader)``
    """
    dataset = RNAGraphDataset(
        root=config["data"]["root"],
        target_family=config["data"]["target_family"],
        max_seq_len=config["data"].get("max_seq_len", 0),
    )
    print(f"Dataset size: {len(dataset)} samples")

    n = len(dataset)
    split = int(n * config["data"]["train_split"])
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(
        dataset, [split, n - split], generator=generator
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def setup_model(
    config: dict, device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
    """Build model, optimizer, and LR scheduler from config.

    Args:
        config: Parsed YAML configuration.
        device: Torch device.

    Returns:
        ``(model, optimizer, scheduler)``
    """
    model = build_model(config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["training"]["scheduler_patience"],
        factor=config["training"]["scheduler_factor"],
    )
    return model, optimizer, scheduler


# ── Training / evaluation steps ─────────────────────────────────────────────


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch.

    The model returns ``(N_total, vocab_size)`` logits.  Loss is computed
    directly as ``F.cross_entropy(logits, batch.y)`` — no masking or padding
    is needed because PyG batches variable-length graphs natively.

    Args:
        model:     :class:`NcRNADesigner` instance.
        loader:    PyG DataLoader over the training split.
        optimizer: Optimizer instance.
        device:    Torch device.
        grad_clip: Max gradient norm for clipping.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # logits: (N_total, vocab_size),  batch.y: (N_total,)
        loss = F.cross_entropy(logits, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a data split.

    Args:
        model:  :class:`NcRNADesigner` instance.
        loader: PyG DataLoader.
        device: Torch device.

    Returns:
        Dict with keys:

        - **loss** *(float)*: mean cross-entropy over all positions.
        - **perplexity** *(float)*: ``exp(loss)``.
        - **recovery** *(float)*: fraction of positions where
          ``argmax(logits) == target``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "recovery": correct / max(total, 1),
    }


@torch.no_grad()
def compute_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Compute post-training diagnostics on a data split.

    Args:
        model:  :class:`NcRNADesigner` instance.
        loader: PyG DataLoader.
        device: Torch device.

    Returns:
        ``(conf_matrix, pred_counts, target_counts)`` where:

        - **conf_matrix** *(4×4 int ndarray)*: ``[true][pred]`` counts.
        - **pred_counts** *([A, U, G, C] ints)*: predicted totals.
        - **target_counts** *([A, U, G, C] ints)*: ground-truth totals.
    """
    model.eval()
    conf_matrix = np.zeros((4, 4), dtype=int)
    pred_counts = [0, 0, 0, 0]
    target_counts = [0, 0, 0, 0]

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch.y.cpu().numpy()
        for t, p in zip(targets, preds):
            conf_matrix[t, p] += 1
        for i in range(4):
            pred_counts[i] += int((preds == i).sum())
            target_counts[i] += int((targets == i).sum())

    return conf_matrix, pred_counts, target_counts


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ncRNA Designer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = setup_data(config)
    model, optimizer, scheduler = setup_model(config, device)

    # ── Training loop ───────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_perplexities: list[float] = []
    val_recoveries: list[float] = []

    best_val_loss = float("inf")
    grad_clip = config["training"]["grad_clip"]
    num_epochs = config["training"]["num_epochs"]
    es_patience = config["training"]["early_stopping_patience"]
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["loss"])

        train_losses.append(train_loss)
        val_losses.append(metrics["loss"])
        val_perplexities.append(metrics["perplexity"])
        val_recoveries.append(metrics["recovery"])

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.4f} | "
            f"Val {metrics['loss']:.4f} | "
            f"Perp {metrics['perplexity']:.3f} | "
            f"Rec {metrics['recovery']:.4f} | "
            f"LR {lr:.2e}"
        )

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_family": config["data"]["target_family"],
                },
                "best_model.pt",
            )
            print(f"  -> Saved best model (val_loss={metrics['loss']:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {es_patience} epochs)."
                )
                break

    # ── Wrap-up ─────────────────────────────────────────────────────────────
    plot_loss_curves(
        train_losses, val_losses, val_perplexities, val_recoveries,
        "loss_curves.png",
    )
    print("Loss curves saved to loss_curves.png")

    # Reload best checkpoint for final diagnostics
    checkpoint = torch.load("best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    conf_matrix, pred_counts, target_counts = compute_diagnostics(
        model, val_loader, device
    )
    plot_diagnostics(conf_matrix, pred_counts, target_counts, "diagnostics.png")
    print("Diagnostics saved to diagnostics.png")
    print("Training complete.")


if __name__ == "__main__":
    main()
