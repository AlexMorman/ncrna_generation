"""encoder.py — Pure-GNN non-autoregressive RNA sequence designer.

Consumes PyTorch Geometric graph data (node features + edge structure) and
produces per-node nucleotide logits directly — no autoregressive decoder.

Architecture:
    1. Linear projection of raw node features → hidden_dim
    2. *num_layers* stacked GATv2Conv layers with:
       - Multi-head attention (concatenated heads)
       - Edge-attribute-aware attention (backbone vs. base-pair)
       - LayerNorm + ELU activation + residual connections
    3. Per-node classifier head: Linear → ELU → Linear → vocab_size logits

Input/output shapes:
    - Input  x:          (N_total, node_input_dim)
    - Output logits:     (N_total, vocab_size)

where N_total = sum of sequence lengths across the batch.  No padding or
masking is needed — PyTorch Geometric handles variable-length graphs natively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class NcRNADesigner(nn.Module):
    """Pure-GNN per-node nucleotide designer for RNA structures.

    Args:
        node_input_dim: Dimensionality of raw node features (3 for structure
            one-hot: ``'.'``, ``'('``, ``')'``).
        hidden_dim:     Hidden embedding dimension. Must be divisible by
            *num_heads*.
        num_heads:      Number of attention heads per GATv2Conv layer.
        num_layers:     Number of stacked GATv2Conv layers.
        dropout:        Dropout probability applied after each GAT layer and
            inside the classifier head.
        edge_dim:       Dimensionality of edge attributes (2 for one-hot
            backbone / base-pair).
        vocab_size:     Number of output classes (4 for A/U/G/C).
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        edge_dim: int,
        vocab_size: int,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"num_heads ({num_heads})."
        )

        self.input_proj = nn.Linear(node_input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-node nucleotide logits from a structure graph.

        Args:
            x:          Node features, shape ``(N_total, node_input_dim)``.
            edge_index: COO edge list, shape ``(2, E)``.
            edge_attr:  Edge attributes, shape ``(E, edge_dim)`` or ``None``.
            batch:      Batch assignment vector, shape ``(N_total,)`` or
                        ``None`` (single graph).

        Returns:
            Per-node logits, shape ``(N_total, vocab_size)``.  The caller
            applies ``F.cross_entropy(logits, batch.y)`` directly — no
            padding or masking is required.
        """
        x = self.input_proj(x)  # (N_total, hidden_dim)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)  # (N_total, hidden_dim)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            x = x + residual

        logits = self.classifier(x)  # (N_total, vocab_size)
        return logits
