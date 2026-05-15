# ncRNA Generation Project

## Project Overview
A deep learning pipeline for generating novel non-coding RNA sequences
using a Graph Attention Network (GAT) encoder and GRU decoder architecture.
Trained on RFAM seed alignments (RF00001–RF00500). Designed to run on Google Colab GPU hardware.

## Architecture
- **Encoder:** Graph Attention Network (GAT) via PyTorch Geometric
- **Decoder:** GRU with attention mechanism, autoregressive generation
- **Training:** Scheduled teacher forcing (annealed 1.0→0.1) with Cross-Entropy loss, early stopping
- **Inference:** Beam Search (top-k paths) + ViennaRNA thermodynamic filtering

## File Structure
```
ncrna_generation/
├── data/
│   ├── raw/                # RFAM Stockholm (.sto) or FASTA-like (.txt/.bprna) files
│   └── processed/          # PyTorch Geometric Data objects (cached tensors)
├── configs/
│   └── config.yaml         # All hyperparameters (lr, batch_size, beam_k, etc.)
├── src/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Geometric Dataset class, graph construction
│   ├── encoder.py          # GAT nn.Module (node/edge tensors → structural embeddings)
│   ├── decoder.py          # GRU nn.Module (autoregressive generation + attention)
│   ├── model.py            # Wraps Encoder + Decoder, defines forward() pass
│   ├── oracle.py           # ViennaRNA wrapper for thermodynamic filtering
│   └── utils.py            # Helpers (loss curve plotting, etc.)
├── train.py                # Training loop (DataLoader, optimizer, Teacher Forcing)
└── inference.py            # Beam search generation + oracle filtering
```

## Execution Flow
1. `dataset.py` parses dot-bracket strings → backbone/base-pair edges → PyG Data objects
2. `encoder.py` GAT consumes Data objects → rich structural embeddings
3. `decoder.py` GRU takes embeddings → autoregressively generates sequences
4. `model.py` connects encoder and decoder via forward() pass
5. `train.py` runs the training loop with scheduled teacher forcing (annealed 1.0→0.1 over epochs), early stopping (patience=15), and a seeded random 80/20 train/val split
6. `inference.py` runs Beam Search on new dot-bracket inputs → oracle filters results

## Key Constraints
- Must run on Google Colab (GPU: T4 or A100)
- Dependencies: PyTorch, PyTorch Geometric, ViennaRNA, PyYAML, matplotlib
- All hyperparameters must live in configs/config.yaml, never hardcoded
- Each file must have clear docstrings explaining inputs/outputs and tensor shapes
- model.py must be kept minimal — only imports and forward() — so components are swappable

## Git
- Initialize a git repo in this directory
- Make an initial commit once all files are created
- Use conventional commit messages
