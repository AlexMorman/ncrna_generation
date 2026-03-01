"""inference.py — Beam search generation + oracle filtering.

Usage::

    python inference.py --structure "(((...)))" --model_path best_model.pt

Given a target dot-bracket secondary structure, runs the trained model with
beam search to produce candidate nucleotide sequences, then filters them
through the ViennaRNA thermodynamic oracle.
"""

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.dataset import IDX_TO_NUC, structure_to_data
from src.decoder import GRUDecoder
from src.model import build_model
from src.oracle import ViennaRNAOracle
from src.utils import load_config


# ── Beam search ─────────────────────────────────────────────────────────────


def beam_search(
    model: torch.nn.Module,
    data,
    beam_k: int,
    device: torch.device,
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """Beam-search decoding for a single target structure.

    Args:
        model:       Trained :class:`NcRNAGenerator`.
        data:        PyG ``Data`` object with structure graph (no ``y``
                     needed).
        beam_k:      Number of beams (top-k paths retained per step).
        device:      Torch device.
        temperature: Softmax temperature for logit scaling (1.0 = neutral).

    Returns:
        List of ``(sequence_str, log_probability)`` tuples sorted by score
        descending.  Length is at most *beam_k*.
    """
    model.eval()

    with torch.no_grad():
        batch = Batch.from_data_list([data]).to(device)

        # Encode
        node_embs = model.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        enc_out, mask = model._unbatch_and_pad(node_embs, batch.batch)
        # enc_out: (1, L, hidden_dim),  mask: (1, L)

        seq_len = int(mask.sum().item())

        # Initial decoder hidden state
        hidden = model.decoder._init_hidden(enc_out, mask)
        # hidden: (num_layers, 1, hidden_dim)

        sos = GRUDecoder.SOS_TOKEN
        input_token = torch.full((1,), sos, dtype=torch.long, device=device)

        # Each beam: (token_list, cumulative_log_prob, hidden_state)
        beams: list[tuple[list[int], float, torch.Tensor]] = [
            ([], 0.0, hidden)
        ]

        for t in range(seq_len):
            candidates: list[tuple[list[int], float, torch.Tensor]] = []

            for tokens, score, h in beams:
                if t == 0:
                    inp = input_token
                else:
                    inp = torch.tensor(
                        [tokens[-1]], dtype=torch.long, device=device
                    )

                logits, h_new = model.decoder.step(inp, h, enc_out, mask)
                # logits: (1, vocab_size)

                scaled = logits.squeeze(0) / temperature
                log_probs = F.log_softmax(scaled, dim=-1)

                topk_vals, topk_idxs = log_probs.topk(beam_k)
                for k in range(beam_k):
                    candidates.append((
                        tokens + [topk_idxs[k].item()],
                        score + topk_vals[k].item(),
                        h_new,
                    ))

            # Keep top beam_k candidates
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = candidates[:beam_k]

    results: list[tuple[str, float]] = []
    for tokens, score, _ in beams:
        seq = "".join(IDX_TO_NUC[t] for t in tokens)
        results.append((seq, score))

    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ncRNA sequences")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--structure",
        type=str,
        required=True,
        help='Target dot-bracket structure, e.g. "(((...)))".',
    )
    parser.add_argument(
        "--no_oracle",
        action="store_true",
        help="Skip ViennaRNA oracle filtering (useful if ViennaRNA is not installed).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(config, device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Build structure graph ───────────────────────────────────────────────
    data = structure_to_data(args.structure)

    # ── Beam search ─────────────────────────────────────────────────────────
    beam_k = config["inference"]["beam_k"]
    temperature = config["inference"].get("temperature", 1.0)
    candidates = beam_search(model, data, beam_k, device, temperature)

    print(f"\nTarget structure: {args.structure}")
    print(f"Generated {len(candidates)} candidate(s):\n")

    # ── Oracle filtering ────────────────────────────────────────────────────
    if args.no_oracle:
        for i, (seq, score) in enumerate(candidates, 1):
            print(f"  {i}. {seq}  (log_prob={score:.4f})")
        return

    oracle = ViennaRNAOracle(config)
    filtered = oracle.filter_candidates(candidates, args.structure)

    if filtered:
        print(f"{len(filtered)} candidate(s) passed oracle filter:\n")
        for i, (seq, score, ev) in enumerate(filtered, 1):
            print(f"  {i}. {seq}")
            print(f"     Log-prob:    {score:.4f}")
            print(f"     MFE:         {ev['mfe']:.2f} kcal/mol")
            print(f"     Similarity:  {ev['similarity']:.4f}")
            print(f"     Fold:        {ev['predicted_structure']}")
            print()
    else:
        print("No candidates passed oracle filter.  All beam results:\n")
        for i, (seq, score) in enumerate(candidates, 1):
            ev = oracle.evaluate(seq, args.structure)
            print(f"  {i}. {seq}")
            print(f"     Log-prob:    {score:.4f}")
            print(f"     MFE:         {ev['mfe']:.2f} kcal/mol")
            print(f"     Similarity:  {ev['similarity']:.4f}")
            print(f"     Fold:        {ev['predicted_structure']}")
            print()


if __name__ == "__main__":
    main()
