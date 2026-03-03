"""train.py — Training loop with Teacher Forcing and Cross-Entropy loss.

Usage::

    python train.py --config configs/config.yaml

Reads RNA structure–sequence pairs via :class:`RNAGraphDataset`, trains the
GAT encoder + GRU decoder model, saves the best checkpoint to
``best_model.pt``, and writes the following output files:

- ``loss_curves.png``  — train/val loss + val accuracy over epochs.
- ``diagnostics.png``  — post-training 2×2 panel: confusion matrix,
  per-class accuracy, nucleotide frequency distribution, and base-pair
  satisfaction rate vs random baseline.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.dataset import RNAGraphDataset
from src.model import build_model
from src.utils import load_config, plot_diagnostics, plot_loss_curves, set_seed


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


@torch.no_grad()
def compute_final_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int], float]:
    """Compute post-training diagnostics on a data split (no teacher forcing).

    Collects predictions across the full split to produce:

    - A 4×4 confusion matrix (rows = true class, cols = predicted class).
    - Per-nucleotide predicted and target counts.
    - Base-pair satisfaction rate: fraction of base-paired position-pairs
      whose predicted nucleotides form a valid Watson-Crick or G-U wobble pair.

    Watson-Crick + wobble pairs (A=0, U=1, G=2, C=3):
    ``(A,U), (U,A), (G,C), (C,G), (G,U), (U,G)``

    Args:
        model:  :class:`NcRNAGenerator` instance.
        loader: PyG DataLoader over the evaluation split.
        device: Torch device.

    Returns:
        ``(conf_matrix, pred_counts, target_counts, bp_rate)`` where:

        - **conf_matrix** *(4, 4 int ndarray)*: ``[true][pred]`` counts.
        - **pred_counts** *([A, U, G, C] ints)*: predicted nucleotide totals.
        - **target_counts** *([A, U, G, C] ints)*: ground-truth totals.
        - **bp_rate** *(float)*: base-pair satisfaction (0–1).
    """
    model.eval()

    conf_matrix = np.zeros((4, 4), dtype=int)
    pred_counts = [0, 0, 0, 0]
    target_counts = [0, 0, 0, 0]

    # Valid ordered Watson-Crick + wobble pairs: (A,U),(U,A),(G,C),(C,G),(G,U),(U,G)
    valid_wc = torch.zeros(4, 4, dtype=torch.bool, device=device)
    for s, d in [(0, 1), (1, 0), (2, 3), (3, 2), (2, 1), (1, 2)]:
        valid_wc[s, d] = True

    bp_correct = 0
    bp_total = 0

    for batch in loader:
        batch = batch.to(device)
        logits, targets, mask = model(batch, teacher_forcing_ratio=0.0)
        # logits:  (B, max_len, vocab_size)
        # targets: (B, max_len)
        # mask:    (B, max_len)

        all_preds = logits.argmax(dim=-1)  # (B, max_len)

        # ── Confusion matrix & nucleotide counts ────────────────────────────
        targets_flat = targets[mask].cpu().numpy()
        preds_flat = all_preds[mask].cpu().numpy()
        for t, p in zip(targets_flat, preds_flat):
            conf_matrix[t, p] += 1
        for i in range(4):
            pred_counts[i] += int((preds_flat == i).sum())
            target_counts[i] += int((targets_flat == i).sum())

        # ── Base-pair satisfaction ──────────────────────────────────────────
        # Map each node in the batch → its predicted nucleotide.
        # batch.batch[n] = graph index g for node n.
        # Position of node n within graph g = n - offset[g].
        n_nodes = batch.batch.size(0)
        sizes = torch.bincount(batch.batch)          # (B,)
        offsets = torch.zeros_like(sizes)
        offsets[1:] = sizes[:-1].cumsum(0)
        positions = (
            torch.arange(n_nodes, device=device) - offsets[batch.batch]
        )                                            # (N,)
        node_preds = all_preds[batch.batch, positions]  # (N,)

        # Base-pair edges: edge_attr[:, 1] == 1
        bp_mask = batch.edge_attr[:, 1] == 1
        if bp_mask.any():
            bp_src = batch.edge_index[0][bp_mask]
            bp_dst = batch.edge_index[1][bp_mask]
            # Each pair is stored twice (bidirectional) — keep src < dst only.
            unique = bp_src < bp_dst
            bp_src = bp_src[unique]
            bp_dst = bp_dst[unique]

            if bp_src.numel() > 0:
                src_preds = node_preds[bp_src]
                dst_preds = node_preds[bp_dst]
                bp_correct += valid_wc[src_preds, dst_preds].sum().item()
                bp_total += bp_src.numel()

    bp_rate = bp_correct / max(bp_total, 1)
    return conf_matrix, pred_counts, target_counts, bp_rate


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
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(
        dataset, [split, n - split], generator=generator
    )
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
    val_accs: list[float] = []
    best_val_loss = float("inf")
    tf_start = config["training"]["teacher_forcing_start"]
    tf_end = config["training"]["teacher_forcing_end"]
    grad_clip = config["training"]["grad_clip"]
    num_epochs = config["training"]["num_epochs"]
    es_patience = config["training"]["early_stopping_patience"]
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        tf_ratio = tf_start - (tf_start - tf_end) * (epoch - 1) / max(num_epochs - 1, 1)

        train_loss = train_epoch(
            model, train_loader, optimizer, device, tf_ratio, grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Acc {val_acc:.4f} | "
            f"LR {lr:.2e} | "
            f"TF {tf_ratio:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {es_patience} epochs)."
                )
                break

    # ── Wrap-up ─────────────────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses, val_accs, "loss_curves.png")
    print("Loss curves saved to loss_curves.png")

    # Reload best checkpoint for final diagnostics
    model.load_state_dict(
        torch.load("best_model.pt", map_location=device, weights_only=True)
    )
    conf_matrix, pred_counts, target_counts, bp_rate = compute_final_metrics(
        model, val_loader, device
    )
    plot_diagnostics(conf_matrix, pred_counts, target_counts, bp_rate, "diagnostics.png")
    print(f"Diagnostics saved to diagnostics.png")
    print(f"  Base-pair satisfaction: {bp_rate:.3f}  (random baseline: {6/16:.3f})")
    print("Training complete.")


if __name__ == "__main__":
    main()
