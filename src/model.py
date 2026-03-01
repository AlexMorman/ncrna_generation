"""model.py — Wraps Encoder + Decoder, defines forward() pass.

Kept minimal so that encoder and decoder components are independently
swappable.  All heavy logic lives in ``encoder.py`` and ``decoder.py``.
"""

import torch
import torch.nn as nn

from .encoder import GATEncoder
from .decoder import GRUDecoder


class NcRNAGenerator(nn.Module):
    """Full ncRNA generation model: GAT encoder → GRU decoder.

    Args:
        encoder: :class:`GATEncoder` instance.
        decoder: :class:`GRUDecoder` instance.
    """

    def __init__(self, encoder: GATEncoder, decoder: GRUDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # ── Utility ─────────────────────────────────────────────────────────────

    def _unbatch_and_pad(
        self,
        values: torch.Tensor,
        batch_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert PyG concatenated tensors into a padded batch.

        Args:
            values:       ``(total_nodes, D)`` or ``(total_nodes,)`` —
                          concatenated per-node values from a PyG Batch.
            batch_vector: ``(total_nodes,)`` — graph-assignment indices.

        Returns:
            padded: ``(B, max_len, D)`` or ``(B, max_len)`` — zero-padded.
            mask:   ``(B, max_len)`` — boolean, ``True`` at valid positions.
        """
        device = values.device
        sizes = torch.bincount(batch_vector)
        batch_size = sizes.size(0)
        max_len = int(sizes.max().item())

        if values.dim() == 1:
            padded = values.new_zeros(batch_size, max_len)
        else:
            padded = values.new_zeros(batch_size, max_len, values.size(-1))

        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        offsets = torch.zeros_like(sizes)
        offsets[1:] = sizes[:-1].cumsum(0)

        for i in range(batch_size):
            n = int(sizes[i].item())
            s = int(offsets[i].item())
            padded[i, :n] = values[s : s + n]
            mask[i, :n] = True

        return padded, mask

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        batch,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode structure graph, then decode nucleotide sequence.

        Args:
            batch: PyG ``Batch`` object containing ``x``, ``edge_index``,
                ``edge_attr``, ``y`` (targets), and ``batch`` (assignment).
            teacher_forcing_ratio: Probability of using ground-truth as
                decoder input at each step.

        Returns:
            logits:  ``(B, max_len, vocab_size)``
            targets: ``(B, max_len)`` — padded target sequences.
            mask:    ``(B, max_len)`` — valid-position boolean mask.
        """
        node_embeddings = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        encoder_outputs, mask = self._unbatch_and_pad(node_embeddings, batch.batch)
        targets, _ = self._unbatch_and_pad(batch.y, batch.batch)
        targets = targets.long()

        logits = self.decoder(encoder_outputs, targets, mask, teacher_forcing_ratio)

        return logits, targets, mask


def build_model(config: dict, device: torch.device) -> NcRNAGenerator:
    """Factory: construct :class:`NcRNAGenerator` from a config dict.

    Args:
        config: Parsed YAML configuration (see ``configs/config.yaml``).
        device: Target torch device.

    Returns:
        Model moved to *device*.
    """
    enc_cfg = config["model"]["encoder"]
    dec_cfg = config["model"]["decoder"]

    encoder = GATEncoder(
        node_input_dim=enc_cfg["node_input_dim"],
        hidden_dim=enc_cfg["hidden_dim"],
        num_heads=enc_cfg["num_heads"],
        num_layers=enc_cfg["num_layers"],
        dropout=enc_cfg["dropout"],
        edge_dim=enc_cfg["edge_dim"],
    )

    decoder = GRUDecoder(
        vocab_size=dec_cfg["vocab_size"],
        embed_dim=dec_cfg["embed_dim"],
        hidden_dim=dec_cfg["hidden_dim"],
        num_layers=dec_cfg["num_layers"],
        dropout=dec_cfg["dropout"],
        encoder_dim=enc_cfg["hidden_dim"],
    )

    return NcRNAGenerator(encoder, decoder).to(device)
