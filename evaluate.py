"""evaluate.py — Consensus-structure evaluation for the ncRNA designer.

Usage::

    python evaluate.py --config configs/config.yaml \
                       --model_path best_model.pt \
                       --output_dir results/

Loads the trained :class:`NcRNADesigner`, locates the ``.sto`` file for the
configured family, runs the consensus-structure window evaluation, and writes
output artifacts to ``<output_dir>/<family>_<timestamp>/``.

Output artifacts:
  - ``results.txt``               — human-readable plaintext report.
  - ``summary.json``              — structured metrics for programmatic use.
  - ``per_position_recovery.png`` — recovery vs. consensus position plot.
  - ``window_count.png``          — windows covering each position.
  - ``recovery_histogram.png``    — histogram of per-window recovery rates.
"""

import argparse
import datetime
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.dataset import RNAGraphDataset
from src.evaluation import (
    evaluate_on_consensus,
    evaluate_perplexity,
    extract_consensus_from_sto,
)
from src.model import build_model
from src.utils import (
    load_config,
    plot_per_position_recovery,
    plot_recovery_histogram,
    set_seed,
)


def _find_sto_file(data_root: str, target_family: str) -> str:
    """Return the path to the single ``.sto`` file for *target_family*.

    Args:
        data_root:     Dataset root directory (``data/``).
        target_family: RFAM family identifier.

    Returns:
        Absolute path to the ``.sto`` file.

    Raises:
        SystemExit: If zero or multiple ``.sto`` files are found.
    """
    family_dir = os.path.join(data_root, "raw", target_family)
    matches = glob.glob(os.path.join(family_dir, "*.sto"))
    if len(matches) == 0:
        sys.exit(
            f"ERROR: No .sto file found in {family_dir}. "
            "Provide exactly one Stockholm alignment for evaluation."
        )
    if len(matches) > 1:
        sys.exit(
            f"ERROR: Multiple .sto files found in {family_dir}: {matches}. "
            "Provide exactly one."
        )
    return matches[0]


def _load_model(model_path: str, config: dict, device: torch.device, target_family: str):
    """Load model from checkpoint, verifying target_family matches config.

    Args:
        model_path:    Path to ``best_model.pt``.
        config:        Parsed YAML config.
        device:        Torch device.
        target_family: Expected family from config.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_family = checkpoint.get("target_family", None)
    if ckpt_family is not None and ckpt_family != target_family:
        print(
            f"WARNING: checkpoint target_family='{ckpt_family}' does not match "
            f"config target_family='{target_family}'. Proceeding anyway."
        )
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _plot_window_count(windows_per_position: list, save_path: str) -> None:
    """Bar plot of window coverage per consensus position."""
    fig, ax = plt.subplots(figsize=(max(10, len(windows_per_position) // 5), 4))
    ax.bar(range(len(windows_per_position)), windows_per_position,
           color="#FF9800", alpha=0.8, width=1.0)
    ax.set_xlabel("Consensus position")
    ax.set_ylabel("Window count")
    ax.set_title("Windows Covering Each Consensus Position")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ncRNA Designer")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config."
    )
    parser.add_argument(
        "--model_path", type=str, default="best_model.pt",
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Parent directory for output artifacts."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    target_family = config["data"]["target_family"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Family: {target_family}")

    # ── Validation DataLoader ────────────────────────────────────────────────
    dataset = RNAGraphDataset(
        root=config["data"]["root"],
        target_family=target_family,
        max_seq_len=config["data"].get("max_seq_len", 0),
    )
    n = len(dataset)
    split = int(n * config["data"]["train_split"])
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    _, val_dataset = random_split(
        dataset, [split, n - split], generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    # ── Load model ───────────────────────────────────────────────────────────
    model = _load_model(args.model_path, config, device, target_family)

    # ── Val perplexity ────────────────────────────────────────────────────────
    val_perplexity = evaluate_perplexity(model, val_loader, device)
    print(f"Val perplexity: {val_perplexity:.4f}")

    # ── Consensus structure ───────────────────────────────────────────────────
    sto_path = _find_sto_file(config["data"]["root"], target_family)
    print(f"Loading consensus from: {sto_path}")
    try:
        consensus_structure, consensus_sequence, valid_nt_map = \
            extract_consensus_from_sto(sto_path)
    except ValueError as exc:
        sys.exit(f"ERROR parsing {sto_path}: {exc}")

    consensus_len = len(consensus_structure)
    print(f"Consensus length: {consensus_len}")

    eval_cfg = config.get("evaluation", {})
    min_len = eval_cfg.get("min_window_length", 25)
    max_len = eval_cfg.get("max_window_length", 50)

    if consensus_len < min_len:
        sys.exit(
            f"ERROR: consensus length ({consensus_len}) < min_window_length "
            f"({min_len}). Cannot evaluate."
        )

    # ── Consensus evaluation ──────────────────────────────────────────────────
    print(f"Evaluating on consensus windows (len {min_len}–{max_len})…")
    results = evaluate_on_consensus(
        model, consensus_structure, consensus_sequence, valid_nt_map,
        min_len, max_len, device,
    )
    mean_recovery = results["mean_recovery"]
    num_windows = results["num_windows"]
    print(f"Windows evaluated: {num_windows}")
    print(f"Mean consensus recovery: {mean_recovery:.4f}")

    # ── Consensus-window perplexity ───────────────────────────────────────────
    # Not a loader-based perplexity — we report mean cross-entropy proxy
    # from per-window recovery as a note; val perplexity is the primary metric.

    # ── Output directory ──────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{target_family}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_per_position_recovery(
        results["per_position_recovery"],
        consensus_structure,
        os.path.join(out_dir, "per_position_recovery.png"),
    )
    _plot_window_count(
        results["windows_per_position"],
        os.path.join(out_dir, "window_count.png"),
    )
    plot_recovery_histogram(
        results["per_window_recovery"],
        os.path.join(out_dir, "recovery_histogram.png"),
    )

    # ── results.txt ──────────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        f"ncRNA Designer Evaluation Report",
        "=" * 60,
        f"Timestamp:              {timestamp}",
        f"Family:                 {target_family}",
        f"Checkpoint:             {os.path.abspath(args.model_path)}",
        f"Stockholm file:         {os.path.abspath(sto_path)}",
        f"Consensus length:       {consensus_len}",
        f"Window range:           {min_len}–{max_len} nt",
        f"Windows evaluated:      {num_windows}",
        "",
        "--- Metrics ---",
        f"Val perplexity:         {val_perplexity:.4f}",
        f"Mean consensus recovery:{mean_recovery:.4f}",
        "",
        "--- Per-Position Recovery (first 20 positions) ---",
    ]
    for i, r in enumerate(results["per_position_recovery"][:20]):
        report_lines.append(
            f"  pos {i:3d} [{consensus_structure[i]}]: {r:.3f}"
        )
    if consensus_len > 20:
        report_lines.append(f"  ... ({consensus_len - 20} more positions)")
    report_lines.append("=" * 60)

    report_path = os.path.join(out_dir, "results.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    print(f"Report: {report_path}")

    # ── summary.json ─────────────────────────────────────────────────────────
    summary = {
        "timestamp": timestamp,
        "target_family": target_family,
        "checkpoint": os.path.abspath(args.model_path),
        "sto_file": os.path.abspath(sto_path),
        "consensus_length": consensus_len,
        "min_window_length": min_len,
        "max_window_length": max_len,
        "num_windows": num_windows,
        "val_perplexity": val_perplexity,
        "mean_consensus_recovery": mean_recovery,
        "per_position_recovery": results["per_position_recovery"],
        "windows_per_position": results["windows_per_position"],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
