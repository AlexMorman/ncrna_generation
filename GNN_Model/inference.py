"""inference.py — Single-structure nucleotide sequence designer.

Usage::

    python inference.py --structure "((...))" --model_path best_model.pt

Given a target dot-bracket secondary structure, runs the trained
:class:`NcRNADesigner` with greedy argmax to produce a single nucleotide
sequence.  With ``--verbose``, also prints per-position confidence
(max softmax value).
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.dataset import IDX_TO_NUC, parse_dot_bracket, structure_to_data
from src.model import build_model
from src.utils import load_config


def design_sequence(
    model: torch.nn.Module,
    structure: str,
    device: torch.device,
) -> tuple[str, list[float]]:
    """Design a nucleotide sequence for a dot-bracket structure.

    Args:
        model:     Trained :class:`NcRNADesigner` in eval mode.
        structure: Dot-bracket string (must be balanced).
        device:    Torch device.

    Returns:
        ``(sequence, confidences)`` where *sequence* is the designed
        nucleotide string and *confidences* is a per-position list of
        max softmax values (0–1).
    """
    data = structure_to_data(structure)
    batch_obj = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        logits = model(
            batch_obj.x, batch_obj.edge_index, batch_obj.edge_attr, batch_obj.batch
        )  # (N, vocab_size)
        probs = F.softmax(logits, dim=-1)
        pred_indices = probs.argmax(dim=-1).cpu().tolist()
        confidences = probs.max(dim=-1).values.cpu().tolist()

    sequence = "".join(IDX_TO_NUC[i] for i in pred_indices)
    return sequence, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description="Design ncRNA sequence for a target structure")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--model_path", type=str, default="best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--structure", type=str, required=True,
        help='Target dot-bracket structure, e.g. "(((...)))".',
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-position confidence (max softmax value).",
    )
    args = parser.parse_args()

    # Validate dot-bracket before loading the model
    try:
        parse_dot_bracket(args.structure)
    except ValueError as exc:
        import sys
        sys.exit(f"ERROR: Invalid structure — {exc}")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence, confidences = design_sequence(model, args.structure, device)

    print(f"Structure:  {args.structure}")
    print(f"Sequence:   {sequence}")

    if args.verbose:
        print("\nPer-position confidence:")
        for i, (nuc, conf) in enumerate(zip(sequence, confidences)):
            print(f"  pos {i:3d}  {args.structure[i]}  {nuc}  {conf:.3f}")


if __name__ == "__main__":
    main()
