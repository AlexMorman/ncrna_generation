"""evaluation.py — Reusable evaluation logic for the consensus-structure protocol.

Used by ``evaluate.py`` (the CLI orchestrator).  All functions operate on
pure Python / NumPy / PyTorch objects so they can be unit-tested without
a full training run.
"""

import math
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .dataset import (
    IDX_TO_NUC,
    NUC_TO_IDX,
    _build_pair_map,
    _degap_sequence_and_structure,
    structure_to_data,
)


# ── Consensus extraction ─────────────────────────────────────────────────────


def extract_consensus_from_sto(
    sto_path: str,
) -> Tuple[str, str, Dict[int, Set[str]]]:
    """Parse a Stockholm file to extract the consensus structure and sequence.

    Reads ``#=GC SS_cons`` from the file, removes gap columns, and builds:

    - A clean dot-bracket consensus structure.
    - A modal consensus sequence (most frequent non-gap nucleotide per column).
    - A per-position valid-nucleotide map (all distinct non-gap nucleotides
      observed at each consensus position).

    Pseudoknot bracket types (``<>``, ``[]``, ``{}``) are normalised to
    standard ``()``.

    Args:
        sto_path: Path to the ``.sto`` Stockholm file.

    Returns:
        ``(consensus_structure, consensus_sequence, valid_nt_map)`` where:

        - **consensus_structure** *(str)*: dot-bracket string over
          non-gap consensus positions.
        - **consensus_sequence** *(str)*: per-column modal non-gap nucleotide.
        - **valid_nt_map** *(Dict[int, Set[str]])*: position → set of
          observed non-gap nucleotides (0-indexed, matching
          *consensus_structure*).

    Raises:
        ValueError: If ``#=GC SS_cons`` is absent or no sequences are found.
    """
    sequences: dict[str, str] = {}
    ss_parts: list[str] = []

    with open(sto_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("//"):
                continue
            if line.startswith("#=GC SS_cons"):
                parts = line.split(None, 2)
                if len(parts) == 3:
                    ss_parts.append(parts[2])
            elif line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) == 2:
                    name, fragment = parts
                    sequences[name] = sequences.get(name, "") + fragment

    if not ss_parts:
        raise ValueError(
            f"No '#=GC SS_cons' line found in '{sto_path}'. "
            "Ensure the file is a valid Stockholm/RFAM alignment."
        )
    if not sequences:
        raise ValueError(
            f"No alignment sequences found in '{sto_path}'."
        )

    raw_ss = "".join(ss_parts)
    pair_map = _build_pair_map(raw_ss)  # handles <>, (), [], {}

    # Identify non-gap columns in any sequence (use all columns present
    # in SS_cons that are non-gap in at least one member)
    aln_len = len(raw_ss)

    # Per aligned column: collect non-gap nucleotides from all members
    col_nucs: list[list[str]] = [[] for _ in range(aln_len)]
    for aligned_seq in sequences.values():
        if len(aligned_seq) != aln_len:
            continue
        for col, ch in enumerate(aligned_seq):
            if ch not in "-. \t":
                nuc = ch.upper().replace("T", "U")
                if nuc in NUC_TO_IDX:
                    col_nucs[col].append(nuc)

    # Keep only consensus (non-gap) columns: those where SS_cons is not '-' or '.'
    # More precisely: a consensus column is one where SS_cons is not a gap
    # character.  Gap chars in SS_cons are '-' and '.'.
    consensus_cols: list[int] = [
        col for col, ch in enumerate(raw_ss) if ch not in "-. \t"
    ]

    # Build consensus structure: re-index pair_map to consensus columns
    col_to_consensus_pos: dict[int, int] = {
        col: pos for pos, col in enumerate(consensus_cols)
    }

    consensus_struct_chars: list[str] = []
    for col in consensus_cols:
        if col not in pair_map:
            consensus_struct_chars.append(".")
        else:
            partner_col = pair_map[col]
            if partner_col not in col_to_consensus_pos:
                consensus_struct_chars.append(".")
            elif partner_col > col:
                consensus_struct_chars.append("(")
            else:
                consensus_struct_chars.append(")")
    consensus_structure = "".join(consensus_struct_chars)

    # Build valid_nt_map and consensus_sequence for consensus positions only
    valid_nt_map: Dict[int, Set[str]] = {}
    consensus_seq_chars: list[str] = []
    for pos, col in enumerate(consensus_cols):
        nucs = col_nucs[col]
        if nucs:
            valid_nt_map[pos] = set(nucs)
            # Modal nucleotide
            modal = max(set(nucs), key=nucs.count)
            consensus_seq_chars.append(modal)
        else:
            valid_nt_map[pos] = set()
            consensus_seq_chars.append("N")  # no observation

    consensus_sequence = "".join(consensus_seq_chars)
    return consensus_structure, consensus_sequence, valid_nt_map


# ── Window enumeration ───────────────────────────────────────────────────────


def enumerate_consensus_windows(
    consensus_structure: str,
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int, str]]:
    """Enumerate all valid sliding windows over a consensus structure.

    Each window's bracket notation is re-balanced: if a bracket's partner
    lies outside the window, the bracket is converted to ``'.'``.  This
    prevents feeding an unbalanced dot-bracket to :func:`structure_to_data`.

    Args:
        consensus_structure: Dot-bracket string.
        min_len:             Minimum window length (inclusive).
        max_len:             Maximum window length (inclusive).

    Returns:
        List of ``(start, length, window_structure)`` triples for all
        windows where ``min_len ≤ length ≤ max_len`` and
        ``start + length ≤ len(consensus_structure)``.

    Raises:
        ValueError: If ``len(consensus_structure) < min_len``.
    """
    n = len(consensus_structure)
    if n < min_len:
        raise ValueError(
            f"Consensus length ({n}) is shorter than min_len ({min_len})."
        )

    # Build full pair map for the consensus structure (standard '(' / ')')
    full_pair: dict[int, int] = {}
    stack: list[int] = []
    for i, ch in enumerate(consensus_structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                full_pair[j] = i
                full_pair[i] = j

    windows: List[Tuple[int, int, str]] = []
    for start in range(n):
        for length in range(min_len, max_len + 1):
            end = start + length
            if end > n:
                break
            # Re-balance: brackets whose partner is outside [start, end)
            window_chars: list[str] = []
            for i in range(start, end):
                ch = consensus_structure[i]
                if ch in "()" and i in full_pair:
                    partner = full_pair[i]
                    if not (start <= partner < end):
                        ch = "."
                window_chars.append(ch)
            windows.append((start, length, "".join(window_chars)))

    return windows


# ── Model evaluation on consensus ────────────────────────────────────────────


@torch.no_grad()
def evaluate_on_consensus(
    model: torch.nn.Module,
    consensus_structure: str,
    consensus_sequence: str,
    valid_nt_map: Dict[int, Set[str]],
    min_len: int,
    max_len: int,
    device: torch.device,
) -> dict:
    """Evaluate the model on all windows of the consensus structure.

    For each window, builds a PyG Data object via :func:`structure_to_data`,
    runs model greedy argmax, and maps predictions back to consensus positions.

    Strict recovery at each position = fraction of covering windows where the
    model's argmax matches the modal consensus nucleotide at that position.

    Args:
        model:               :class:`NcRNADesigner` instance (eval mode).
        consensus_structure: Full consensus dot-bracket string (length N).
        consensus_sequence:  Modal consensus sequence (length N), used as
                             the strict recovery reference.
        valid_nt_map:        Position → set of valid non-gap nucleotides.
        min_len:             Minimum window length.
        max_len:             Maximum window length.
        device:              Torch device.

    Returns:
        Dict with keys:

        - **per_position_recovery** *(List[float])*: length-N recovery rates.
        - **mean_recovery** *(float)*: mean over positions with at least one
          covering window.
        - **num_windows** *(int)*: total number of windows evaluated.
        - **windows_per_position** *(List[int])*: how many windows cover each
          position.
        - **per_window_recovery** *(List[float])*: mean recovery per window.
    """
    model.eval()
    windows = enumerate_consensus_windows(consensus_structure, min_len, max_len)

    n = len(consensus_structure)
    position_correct = [0] * n
    position_total = [0] * n
    per_window_recovery: List[float] = []

    for start, length, win_struct in windows:
        data = structure_to_data(win_struct)
        batch_obj = Batch.from_data_list([data]).to(device)
        logits = model(
            batch_obj.x, batch_obj.edge_index, batch_obj.edge_attr, batch_obj.batch
        )
        preds = logits.argmax(dim=-1).cpu().tolist()

        win_correct = 0
        for local_pos, pred_idx in enumerate(preds):
            global_pos = start + local_pos
            if global_pos >= n:
                break
            pred_nuc = IDX_TO_NUC[pred_idx]
            ref_nuc = consensus_sequence[global_pos]
            match = (pred_nuc == ref_nuc and ref_nuc in NUC_TO_IDX)
            position_correct[global_pos] += int(match)
            position_total[global_pos] += 1
            win_correct += int(match)

        per_window_recovery.append(win_correct / length)

    per_position_recovery = [
        position_correct[i] / position_total[i]
        if position_total[i] > 0 else 0.0
        for i in range(n)
    ]
    covered_positions = [i for i in range(n) if position_total[i] > 0]
    mean_recovery = (
        sum(per_position_recovery[i] for i in covered_positions)
        / len(covered_positions)
        if covered_positions else 0.0
    )

    return {
        "per_position_recovery": per_position_recovery,
        "mean_recovery": mean_recovery,
        "num_windows": len(windows),
        "windows_per_position": position_total,
        "per_window_recovery": per_window_recovery,
    }


# ── Perplexity (reusable) ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss and return perplexity.

    Args:
        model:  :class:`NcRNADesigner` instance.
        loader: PyG DataLoader.
        device: Torch device.

    Returns:
        ``exp(mean_cross_entropy)`` — perplexity over the loader.
    """
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        total_loss += F.cross_entropy(logits, batch.y).item()
    avg_loss = total_loss / max(len(loader), 1)
    return math.exp(avg_loss)
