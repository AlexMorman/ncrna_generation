"""dataset.py — PyTorch Geometric Dataset class and graph construction.

Parses Eterna/bpRNA files in FASTA-like format::

    >sample_id
    AUGCAUGC
    ((....))

Converts dot-bracket secondary structures into graphs:
  - **Nodes:** one per nucleotide position
  - **Backbone edges:** (i, i+1) bidirectional for consecutive positions
  - **Base-pair edges:** (i, j) bidirectional for each matched bracket pair
  - **Node features:** one-hot encoding of structure character ('.', '(', ')')
  - **Edge attributes:** one-hot encoding of edge type (backbone, base-pair)
  - **Target y:** integer-encoded nucleotide (A=0, U=1, G=2, C=3)
"""

import os
from typing import List, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset

# ── Vocabulary mappings ─────────────────────────────────────────────────────

NUC_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3}
IDX_TO_NUC = {v: k for k, v in NUC_TO_IDX.items()}
STRUCT_TO_IDX = {".": 0, "(": 1, ")": 2}
SOS_TOKEN = 4
PAD_TOKEN = 5


# ── Dot-bracket parsing ────────────────────────────────────────────────────


def parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """Parse dot-bracket notation into a list of base-pair indices.

    Uses a stack to match opening '(' with closing ')'.

    Args:
        structure: Dot-bracket string, e.g. ``"(((...)))"``

    Returns:
        List of ``(i, j)`` tuples where *i < j* and positions *i*, *j* are
        paired.

    Raises:
        ValueError: If brackets are unbalanced.
    """
    pairs: List[Tuple[int, int]] = []
    stack: List[int] = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if not stack:
                raise ValueError(
                    f"Unbalanced closing bracket at position {i} "
                    f"in structure: {structure}"
                )
            j = stack.pop()
            pairs.append((j, i))
    if stack:
        raise ValueError(
            f"Unbalanced opening bracket(s) at position(s) {stack} "
            f"in structure: {structure}"
        )
    return pairs


# ── Graph construction ──────────────────────────────────────────────────────


def structure_to_data(structure: str, sequence: str | None = None) -> Data:
    """Convert a dot-bracket structure (and optional sequence) to a PyG Data object.

    Args:
        structure: Dot-bracket string defining the secondary structure.
        sequence:  Nucleotide string (``A/U/G/C``).  When provided the target
                   tensor ``y`` is included; omit for inference.

    Returns:
        ``torch_geometric.data.Data`` with:

        - **x** *(N, 3)*: one-hot node features for structure characters
        - **edge_index** *(2, E)*: COO edge list (backbone + base-pair, bidir.)
        - **edge_attr** *(E, 2)*: one-hot edge type (col 0 = backbone,
          col 1 = base-pair)
        - **y** *(N,)* *(optional)*: integer-encoded nucleotide targets
        - **seq_len** *int*: sequence length

    Raises:
        ValueError: If *sequence* and *structure* differ in length or contain
            invalid characters.
    """
    n = len(structure)
    if n == 0:
        raise ValueError("Structure string must not be empty.")
    if sequence is not None and len(sequence) != n:
        raise ValueError(
            f"Sequence length ({len(sequence)}) != structure length ({n})."
        )

    # ── Node features: one-hot for '.', '(', ')' ──────────────────────────
    x = torch.zeros(n, 3)
    for i, char in enumerate(structure):
        idx = STRUCT_TO_IDX.get(char)
        if idx is None:
            raise ValueError(
                f"Invalid structure character '{char}' at position {i}."
            )
        x[i, idx] = 1.0

    # ── Edge construction ──────────────────────────────────────────────────
    src: List[int] = []
    dst: List[int] = []
    edge_types: List[int] = []

    # Backbone edges (bidirectional)
    for i in range(n - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])
        edge_types.extend([0, 0])

    # Base-pair edges (bidirectional)
    pairs = parse_dot_bracket(structure)
    for i, j in pairs:
        src.extend([i, j])
        dst.extend([j, i])
        edge_types.extend([1, 1])

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge attributes: one-hot [backbone, base-pair]
    edge_attr = torch.zeros(len(edge_types), 2)
    for k, t in enumerate(edge_types):
        edge_attr[k, t] = 1.0

    # ── Build Data object ──────────────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.seq_len = torch.tensor(n, dtype=torch.long)

    if sequence is not None:
        seq_upper = sequence.upper().replace("T", "U")
        y_list = []
        for i, char in enumerate(seq_upper):
            nuc_idx = NUC_TO_IDX.get(char)
            if nuc_idx is None:
                raise ValueError(
                    f"Invalid nucleotide '{char}' at position {i}."
                )
            y_list.append(nuc_idx)
        data.y = torch.tensor(y_list, dtype=torch.long)

    return data


# ── File parsing ────────────────────────────────────────────────────────────


def parse_rna_file(filepath: str) -> List[Tuple[str, str, str]]:
    """Parse a FASTA-like RNA data file.

    Supported formats::

        >sample_id
        SEQUENCE
        STRUCTURE

    or headerless pairs::

        SEQUENCE
        STRUCTURE

    Args:
        filepath: Path to the text file.

    Returns:
        List of ``(name, sequence, structure)`` tuples.
    """
    samples: List[Tuple[str, str, str]] = []
    with open(filepath, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            name = lines[i][1:].strip()
            if i + 2 >= len(lines):
                break
            sequence = lines[i + 1]
            structure = lines[i + 2]
            samples.append((name, sequence, structure))
            i += 3
        else:
            sequence = lines[i]
            if i + 1 >= len(lines):
                break
            structure = lines[i + 1]
            samples.append((f"sample_{len(samples)}", sequence, structure))
            i += 2
    return samples


# ── PyG Dataset ─────────────────────────────────────────────────────────────


class RNAGraphDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset for RNA structure–sequence pairs.

    Reads ``.txt`` files from ``<root>/raw/``, parses them with
    :func:`parse_rna_file`, converts each sample to a graph via
    :func:`structure_to_data`, and caches the result in
    ``<root>/processed/data.pt``.

    Args:
        root:          Dataset root directory (expects ``raw/`` and
                       ``processed/`` subdirectories).
        max_seq_len:   Discard samples longer than this.  ``0`` means no limit.
        transform:     Optional PyG transform applied at access time.
        pre_transform: Optional PyG transform applied during processing.
    """

    def __init__(
        self,
        root: str,
        max_seq_len: int = 0,
        transform=None,
        pre_transform=None,
    ):
        self.max_seq_len = max_seq_len
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw data files found in ``raw/``."""
        if os.path.isdir(self.raw_dir):
            return sorted(
                f
                for f in os.listdir(self.raw_dir)
                if f.endswith((".txt", ".bprna"))
            )
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self) -> None:
        """No automatic download — users must provide their own data."""

    def process(self) -> None:
        """Read raw files, build graphs, and save processed dataset."""
        data_list: List[Data] = []

        raw_files = [
            os.path.join(self.raw_dir, f)
            for f in os.listdir(self.raw_dir)
            if f.endswith((".txt", ".bprna"))
        ]

        if not raw_files:
            raise FileNotFoundError(
                f"No .txt or .bprna files found in {self.raw_dir}. "
                "Please add RNA data files in FASTA-like format:\n"
                "  >sample_id\n  SEQUENCE\n  STRUCTURE"
            )

        for filepath in sorted(raw_files):
            samples = parse_rna_file(filepath)
            for name, sequence, structure in samples:
                if self.max_seq_len > 0 and len(sequence) > self.max_seq_len:
                    continue
                try:
                    data = structure_to_data(structure, sequence)
                    data.name = name
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                except ValueError as exc:
                    print(f"Skipping '{name}': {exc}")

        if not data_list:
            raise ValueError(
                "No valid samples produced after processing. "
                "Check your raw data format."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed {len(data_list)} RNA samples → {self.processed_paths[0]}")
