"""oracle.py — ViennaRNA wrapper for thermodynamic filtering.

Provides structure prediction (MFE folding) and candidate evaluation.
Candidates are scored on:

- **Minimum Free Energy (MFE):** must fall below a configurable threshold.
- **Structure similarity:** base-pair F1 between the predicted fold and the
  target dot-bracket structure must exceed a configurable threshold.
"""

from typing import Any

from .dataset import parse_dot_bracket

try:
    import RNA

    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False


class ViennaRNAOracle:
    """Thermodynamic oracle backed by ViennaRNA.

    Args:
        config: Parsed YAML configuration dict.  Relevant keys::

            oracle:
              mfe_threshold: -5.0
              similarity_threshold: 0.8
    """

    def __init__(self, config: dict):
        oracle_cfg = config.get("oracle", {})
        self.mfe_threshold: float = oracle_cfg.get("mfe_threshold", -5.0)
        self.similarity_threshold: float = oracle_cfg.get(
            "similarity_threshold", 0.8
        )

    @staticmethod
    def _check_available() -> None:
        if not VIENNA_AVAILABLE:
            raise ImportError(
                "ViennaRNA Python bindings not found.  Install with:\n"
                "  conda install -c bioconda viennarna\n"
                "or see https://www.tbi.univie.ac.at/RNA/"
            )

    # ── Core folding ────────────────────────────────────────────────────────

    def fold(self, sequence: str) -> tuple[str, float]:
        """Predict MFE secondary structure for *sequence*.

        Args:
            sequence: Nucleotide string (A/U/G/C).

        Returns:
            ``(structure, mfe)`` — dot-bracket string and free energy in
            kcal/mol.
        """
        self._check_available()
        structure, mfe = RNA.fold(sequence)
        return structure, mfe

    # ── Structure comparison ────────────────────────────────────────────────

    @staticmethod
    def structure_similarity(predicted: str, target: str) -> float:
        """Compute base-pair F1 between two dot-bracket structures.

        Args:
            predicted: Predicted dot-bracket structure.
            target:    Target dot-bracket structure.

        Returns:
            F1 score in ``[0, 1]``.
        """
        pred_pairs = set(parse_dot_bracket(predicted))
        target_pairs = set(parse_dot_bracket(target))

        if not target_pairs and not pred_pairs:
            return 1.0
        if not target_pairs or not pred_pairs:
            return 0.0

        tp = len(pred_pairs & target_pairs)
        precision = tp / len(pred_pairs)
        recall = tp / len(target_pairs)

        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    # ── Single-candidate evaluation ─────────────────────────────────────────

    def evaluate(
        self,
        sequence: str,
        target_structure: str,
    ) -> dict[str, Any]:
        """Fold *sequence* and compare with *target_structure*.

        Args:
            sequence:         Candidate nucleotide string.
            target_structure: Desired dot-bracket structure.

        Returns:
            Dict with keys ``predicted_structure``, ``mfe``, ``similarity``,
            ``passes_mfe``, ``passes_similarity``, ``passes``.
        """
        predicted_structure, mfe = self.fold(sequence)
        similarity = self.structure_similarity(predicted_structure, target_structure)

        return {
            "predicted_structure": predicted_structure,
            "mfe": mfe,
            "similarity": similarity,
            "passes_mfe": mfe <= self.mfe_threshold,
            "passes_similarity": similarity >= self.similarity_threshold,
            "passes": (
                mfe <= self.mfe_threshold
                and similarity >= self.similarity_threshold
            ),
        }

    # ── Batch filtering ─────────────────────────────────────────────────────

    def filter_candidates(
        self,
        candidates: list[tuple[str, float]],
        target_structure: str,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Evaluate and filter a list of beam-search candidates.

        Args:
            candidates:       List of ``(sequence, log_prob)`` tuples from beam
                              search.
            target_structure: Target dot-bracket structure.

        Returns:
            Filtered list of ``(sequence, log_prob, eval_dict)`` tuples that
            pass both MFE and similarity thresholds, sorted by similarity
            descending.
        """
        results: list[tuple[str, float, dict[str, Any]]] = []
        for sequence, score in candidates:
            eval_result = self.evaluate(sequence, target_structure)
            if eval_result["passes"]:
                results.append((sequence, score, eval_result))

        results.sort(key=lambda r: r[2]["similarity"], reverse=True)
        return results
