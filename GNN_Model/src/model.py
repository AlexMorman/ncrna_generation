"""model.py — Factory for the NcRNADesigner model.

Kept minimal: only imports and the build_model factory.  All architecture
logic lives in encoder.py so the model is independently swappable.
"""

import torch

from .encoder import NcRNADesigner


def build_model(config: dict, device: torch.device) -> NcRNADesigner:
    """Construct :class:`NcRNADesigner` from a config dict.

    Reads the flat ``config["model"]`` section.  Expected keys:
    ``node_input_dim``, ``hidden_dim``, ``num_heads``, ``num_layers``,
    ``dropout``, ``edge_dim``, ``vocab_size``.

    Args:
        config: Parsed YAML configuration (see ``configs/config.yaml``).
        device: Target torch device.

    Returns:
        Model moved to *device*.
    """
    m = config["model"]
    model = NcRNADesigner(
        node_input_dim=m["node_input_dim"],
        hidden_dim=m["hidden_dim"],
        num_heads=m["num_heads"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        edge_dim=m["edge_dim"],
        vocab_size=m["vocab_size"],
    )
    return model.to(device)
