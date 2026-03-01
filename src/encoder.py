"""encoder.py — Graph Attention Network encoder.

Consumes PyTorch Geometric graph data (node features + edge structure) and
produces per-node structural embeddings.

Architecture:
    1. Linear projection of raw node features → hidden_dim
    2. *N* stacked GATConv layers with:
       - Multi-head attention (concatenated heads)
       - Edge-attribute-aware attention (backbone vs. base-pair)
       - Residual connections
       - LayerNorm + ELU activation
    3. Output: per-node embedding of size ``hidden_dim``
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder for RNA structure graphs.

    Args:
        node_input_dim: Dimensionality of raw node features (default 3 for
            structure one-hot: ``'.'``, ``'('``, ``')'``).
        hidden_dim: Hidden / output embedding dimension.  Must be divisible
            by *num_heads*.
        num_heads: Number of attention heads per GAT layer.
        num_layers: Number of stacked GAT layers.
        dropout: Dropout probability applied after each layer.
        edge_dim: Dimensionality of edge attributes (default 2 for one-hot
            backbone / base-pair).
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        edge_dim: int,
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
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode graph nodes into structural embeddings.

        Args:
            x:          Node features, shape ``(N, node_input_dim)``.
            edge_index: COO edge list, shape ``(2, E)``.
            edge_attr:  Edge attributes, shape ``(E, edge_dim)`` or ``None``.
            batch:      Batch assignment vector, shape ``(N,)`` or ``None``.

        Returns:
            Node embeddings, shape ``(N, hidden_dim)``.
        """
        # Project raw features → hidden dimension
        x = self.input_proj(x)  # (N, hidden_dim)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)  # (N, hidden_dim)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            x = x + residual  # residual connection

        return x  # (N, hidden_dim)
