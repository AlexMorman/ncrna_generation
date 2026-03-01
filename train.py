"""train.py — Training loop with Teacher Forcing and Cross-Entropy loss.

Usage::

    python train.py --config configs/config.yaml

Reads RNA structure–sequence pairs via :class:`RNAGraphDataset`, trains the
GAT encoder + GRU decoder model, saves the best checkpoint to
``best_model.pt``, and plots loss curves to ``loss_curves.png``.
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.dataset import RNAGraphDataset
from src.model import build_model
from src.utils import load_config, plot_loss_curves, set_seed


# ── Training / evaluation steps ─────────────────────────────────────────────


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tf_ratio: float,
    grad_clip: float,
) -> float:
    """Run one training epoch.

    Args:
        model:     :class:`NcRNAGenerator` instance.
        loader:    PyG DataLoader over the training split.
        optimizer: Optimizer instance.
        device:    Torch device.
        tf_ratio:  Teacher-forcing ratio (1.0 = always use ground truth).
        grad_clip: Max gradient norm for clipping.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, targets, mask = model(batch, teacher_forcing_ratio=tf_ratio)
        # logits:  (B, max_len, vocab_size)
        # targets: (B, max_len)
        # mask:    (B, max_len)

        loss = F.cross_entropy(logits[mask], targets[mask])

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
) -> tuple[float, float]:
    """Evaluate model on a data split (no teacher forcing).

    Args:
        model:  :class:`NcRNAGenerator` instance.
        loader: PyG DataLoader.
        device: Torch device.

    Returns:
        ``(avg_loss, accuracy)`` over all valid nucleotide positions.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        logits, targets, mask = model(batch, teacher_forcing_ratio=0.0)

        logits_flat = logits[mask]    # (num_valid, vocab_size)
        targets_flat = targets[mask]  # (num_valid,)

        total_loss += F.cross_entropy(logits_flat, targets_flat).item()

        preds = logits_flat.argmax(dim=-1)
        correct += (preds == targets_flat).sum().item()
        total += targets_flat.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ncRNA Generator")
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

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = RNAGraphDataset(
        root="data",
        max_seq_len=config["data"].get("max_seq_len", 0),
    )
    print(f"Dataset size: {len(dataset)} samples")

    n = len(dataset)
    split = int(n * config["data"]["train_split"])
    train_dataset = dataset[:split]
    val_dataset = dataset[split:]
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ── Model ───────────────────────────────────────────────────────────────
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

    # ── Training loop ───────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    tf_ratio = config["training"]["teacher_forcing_ratio"]
    grad_clip = config["training"]["grad_clip"]

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, tf_ratio, grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Acc {val_acc:.4f} | "
            f"LR {lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # ── Wrap-up ─────────────────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses, "loss_curves.png")
    print("Training complete.  Loss curves saved to loss_curves.png")


if __name__ == "__main__":
    main()
