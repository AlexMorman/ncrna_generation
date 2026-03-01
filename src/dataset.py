"""dataset.py — PyTorch Geometric Dataset class and graph construction.

Supports two raw data formats:

**FASTA-like** (``.txt`` / ``.bprna``)::

    >sample_id
    AUGCAUGC
    ((....))

**Stockholm / RFAM** (``.sto``)::

    # STOCKHOLM 1.0
    seq1   GCUUCG...
    seq2   GCUUCG...
    #=GC SS_cons  <<<<<...>>>>>
    //

Stockholm ``SS_cons`` notation is converted to standard dot-bracket:
``< ( [ {`` → ``(``,  ``> ) ] }`` → ``)``,  everything else → ``.``.
Base pairs are retained only when **both** partners are present in the
individual aligned sequence; gapped-out partners become unpaired.

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


# ── Stockholm parsing ───────────────────────────────────────────────────────


def _build_pair_map(ss_cons: str) -> dict[int, int]:
    """Build a position-to-partner map from a Stockholm SS_cons string.

    Handles the full RFAM bracket alphabet: ``<>``, ``()``, ``[]``, ``{}``.
    Pseudoknot letters (uppercase/lowercase A-Z outside the above) are
    ignored and treated as unpaired.

    Args:
        ss_cons: Raw ``SS_cons`` string from a Stockholm file.

    Returns:
        Dict mapping each paired column index to its partner index.
    """
    pair_map: dict[int, int] = {}
    stack: list[int] = []
    for i, c in enumerate(ss_cons):
        if c in "<([{":
            stack.append(i)
        elif c in ">)]}":
            if stack:
                j = stack.pop()
                pair_map[j] = i
                pair_map[i] = j
    return pair_map


def _degap_sequence_and_structure(
    aligned_seq: str,
    pair_map: dict[int, int],
) -> Tuple[str, str]:
    """Remove gap columns and produce a valid dot-bracket structure.

    A base pair is kept only when **both** partner columns contain a
    nucleotide in this particular sequence; otherwise the position is
    written as unpaired.  Ambiguous nucleotides (N, R, Y, …) are skipped.

    Args:
        aligned_seq: Aligned sequence string (gaps are ``-`` or ``.``).
        pair_map:    Precomputed column pairing from :func:`_build_pair_map`.

    Returns:
        ``(sequence, structure)`` — gap-free nucleotide string and its
        dot-bracket structure.
    """
    kept: set[int] = {
        i for i, c in enumerate(aligned_seq) if c not in "-. \t"
    }

    seq_chars: list[str] = []
    struct_chars: list[str] = []

    for i in sorted(kept):
        nuc = aligned_seq[i].upper().replace("T", "U")
        if nuc not in NUC_TO_IDX:
            continue  # ambiguous nucleotide — skip position entirely
        seq_chars.append(nuc)

        if i in pair_map:
            partner = pair_map[i]
            if partner in kept:
                struct_chars.append("(" if partner > i else ")")
            else:
                struct_chars.append(".")
        else:
            struct_chars.append(".")

    return "".join(seq_chars), "".join(struct_chars)


def parse_stockholm_file(filepath: str) -> List[Tuple[str, str, str]]:
    """Parse an RFAM Stockholm alignment file (``.sto``).

    Reads all sequence rows and the ``#=GC SS_cons`` consensus structure
    (concatenating fragments across multiple alignment blocks), then
    degaps each sequence and derives its individual dot-bracket structure
    via :func:`_degap_sequence_and_structure`.

    Args:
        filepath: Path to the ``.sto`` file.

    Returns:
        List of ``(name, sequence, structure)`` tuples.

    Raises:
        ValueError: If no ``#=GC SS_cons`` annotation is found.
    """
    sequences: dict[str, str] = {}
    ss_parts: list[str] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("//"):
                continue
            if line.startswith("#=GC SS_cons"):
                parts = line.split(None, 2)
                if len(parts) == 3:
                    ss_parts.append(parts[2])
            elif line.startswith("#"):
                continue  # GF / GS / GR annotations
            else:
                parts = line.split()
                if len(parts) == 2:
                    name, fragment = parts
                    sequences[name] = sequences.get(name, "") + fragment

    if not ss_parts:
        raise ValueError(
            f"No '#=GC SS_cons' line found in {filepath}. "
            "Ensure the file is a valid Stockholm/RFAM alignment."
        )

    ss_cons = "".join(ss_parts)
    pair_map = _build_pair_map(ss_cons)

    samples: List[Tuple[str, str, str]] = []
    for name, aligned_seq in sequences.items():
        if len(aligned_seq) != len(ss_cons):
            continue  # malformed block — skip
        seq, struct = _degap_sequence_and_structure(aligned_seq, pair_map)
        if len(seq) >= 2:
            samples.append((name, seq, struct))

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
                if f.endswith((".txt", ".bprna", ".sto"))
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
            if f.endswith((".txt", ".bprna", ".sto"))
        ]

        if not raw_files:
            raise FileNotFoundError(
                f"No data files found in {self.raw_dir}. "
                "Accepted formats:\n"
                "  .sto  — Stockholm/RFAM alignment\n"
                "  .txt / .bprna  — FASTA-like (>id / sequence / structure)"
            )

        for filepath in sorted(raw_files):
            if filepath.endswith(".sto"):
                samples = parse_stockholm_file(filepath)
            else:
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
