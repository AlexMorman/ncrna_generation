#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
#import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(r"C:\Users\USER\Downloads\RhoDesign-main")
CHECKPOINT = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_runs\shared_all_families\best.pt")
PDB_PATH = Path(r"C:\Users\USER\OneDrive - University of Kansas\some_input_structure.pdb")
OUTDIR = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_single_pdb")

DEVICE = "cuda"   
SEED = 7
NUM_DESIGNS = 5
TEMPERATURE = 1.0
GREEDY_GENERATION = False


TARGET_SEQUENCE_OVERRIDE = None   
FAMILY_LABEL = None               




class ArgsClass:
    def __init__(self, encoder_embed_dim: int = 512, decoder_embed_dim: int = 512, dropout: float = 0.1):
        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        self.epochs = 100
        self.lr = 1e-5
        self.batch_size = 1
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.dropout = dropout
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256
        self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32
        self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3
        self.gvp_dropout = 0.1
        self.encoder_layers = 3
        self.encoder_attention_heads = 4
        self.attention_dropout = 0.1
        self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3
        self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_seq(seq: str) -> str:
    seq = str(seq).replace("T", "U")
    return "".join(ch if ch in {"A", "G", "C", "U", "X"} else "X" for ch in seq)


def ensure_repo_imports(repo_root: Path) -> None:
    src = repo_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"Could not find src/ under repo root: {repo_root}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def get_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def maybe_unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
    return checkpoint_obj


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    obj = torch.load(checkpoint_path, map_location=device)
    state_dict = maybe_unwrap_state_dict(obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] checkpoint: {checkpoint_path}")
    print(f"[load] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")


def sample_sequence(model, alphabet, coords_np, device: torch.device, temperature: float = 1.0, greedy: bool = False,
                    partial_seq: Optional[str] = None, seed: Optional[int] = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    from util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    batch_coords, confidence, _, _, padding_mask, _ = batch_converter([(coords_np, None, None, None)], device=device)
    coords = batch_coords[:, :, [0, 1, 2], :]
    adjunct_coords = batch_coords
    padding_mask = padding_mask.bool()

    mask_idx = alphabet.get_idx("<mask>")
    start_idx = alphabet.get_idx("<cath>")
    L = len(coords_np)

    sampled_tokens = torch.full((1, 1 + L), mask_idx, dtype=torch.long, device=device)
    sampled_tokens[0, 0] = start_idx

    if partial_seq is not None:
        partial_seq = normalize_seq(partial_seq)
        for i, ch in enumerate(partial_seq[:L]):
            sampled_tokens[0, i + 1] = alphabet.get_idx(ch)

    incremental_state = {}
    encoder_out = model.encoder(coords, adjunct_coords, padding_mask, confidence)

    with torch.no_grad():
        for i in range(1, L + 1):
            if sampled_tokens[0, i].item() != mask_idx:
                continue
            logits, _ = model.decoder(sampled_tokens[:, :i], encoder_out, incremental_state=incremental_state)
            next_logits = logits[0, :, -1] / max(temperature, 1e-8)
            if greedy:
                next_tok = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())
            sampled_tokens[0, i] = next_tok

    toks = sampled_tokens[0, 1:].tolist()
    seq = []
    for tok in toks:
        ch = alphabet.get_tok(tok)
        seq.append(ch if ch in {"A", "C", "G", "U"} else "X")
    return "".join(seq)


def teacher_forced_metrics(model, alphabet, coords_np, native_seq: str, device: torch.device) -> Dict[str, float]:
    from util import CoordBatchConverter

    native_seq = normalize_seq(native_seq)
    batch_converter = CoordBatchConverter(alphabet)
    batch_coords, confidence, _, tokens, padding_mask, _ = batch_converter([(coords_np, None, native_seq, None)], device=device)
    coords = batch_coords[:, :, [0, 1, 2], :]
    adjunct_coords = batch_coords

    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    valid = target != alphabet.padding_idx

    with torch.no_grad():
        logits, _ = model(coords, adjunct_coords, padding_mask.bool(), confidence, prev_output_tokens)
        loss = F.cross_entropy(logits, target, reduction="none")

    loss_sum = float(loss[valid].sum().item())
    n_tokens = int(valid.sum().item())
    pred = logits.argmax(dim=1)
    correct = int(((pred == target) & valid).sum().item())
    mean_loss = loss_sum / max(n_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "token_acc": correct / max(n_tokens, 1),
        "n_tokens": n_tokens,
    }


def seq_recovery(native_seq: str, pred_seq: str) -> float:
    if len(native_seq) != len(pred_seq):
        raise ValueError(f"Length mismatch: native={len(native_seq)} pred={len(pred_seq)}")
    return float(sum(a == b for a, b in zip(native_seq, pred_seq)) / max(len(native_seq), 1))


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected Nx3 arrays with same shape, got {P.shape} and {Q.shape}")
    P_cent = P - P.mean(axis=0, keepdims=True)
    Q_cent = Q - Q.mean(axis=0, keepdims=True)
    H = P_cent.T @ Q_cent
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    P_aligned = P_cent @ R
    return float(np.sqrt(np.mean(np.sum((P_aligned - Q_cent) ** 2, axis=1))))


def structure_consistency_metrics(target_coords: np.ndarray, refolded_coords: np.ndarray) -> Dict[str, float]:
    if target_coords.shape[0] != refolded_coords.shape[0]:
        raise ValueError(
            f"Target and refolded structures have different residue counts: {target_coords.shape[0]} vs {refolded_coords.shape[0]}"
        )
    finite_mask = np.isfinite(target_coords).all(axis=-1) & np.isfinite(refolded_coords).all(axis=-1)
    P = target_coords[finite_mask]
    Q = refolded_coords[finite_mask]
    if len(P) < 3:
        raise ValueError("Not enough comparable atoms to compute RMSD")
    return {
        "n_compared_atoms": int(len(P)),
        "aligned_backbone_rmsd": kabsch_rmsd(P, Q),
    }


def parse_cmsearch_tblout(tblout_path: Path) -> Dict[str, object]:
    best = None
    with tblout_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 18:
                continue
            row = {
                "target_name": parts[0],
                "target_accession": parts[1],
                "query_name": parts[2],
                "query_accession": parts[3],
                "mdl": parts[4],
                "mdl_from": parts[5],
                "mdl_to": parts[6],
                "seq_from": parts[7],
                "seq_to": parts[8],
                "strand": parts[9],
                "trunc": parts[10],
                "pass": parts[11],
                "gc": parts[12],
                "bias": parts[13],
                "bit_score": float(parts[14]),
                "evalue": parts[15],
                "inc": parts[16],
                "description": " ".join(parts[17:]),
            }
            if best is None or row["bit_score"] > best["bit_score"]:
                best = row
    return best if best is not None else {}


def run_cmsearch(cmsearch_exe: str, cm_file: Path, sequence: str, outdir: Path, prefix: str) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    fasta_path = outdir / f"{prefix}.fasta"
    tblout_path = outdir / f"{prefix}.cmsearch.tblout"
    with fasta_path.open("w", encoding="utf-8") as f:
        f.write(f">{prefix}\n{sequence}\n")

    cmd = [cmsearch_exe, "--tblout", str(tblout_path), str(cm_file), str(fasta_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "tblout_path": str(tblout_path),
    }
    if proc.returncode == 0 and tblout_path.exists():
        result["best_hit"] = parse_cmsearch_tblout(tblout_path)
    return result


def main() -> None:
    set_seed(SEED)
    device = get_device(DEVICE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ensure_repo_imports(REPO_ROOT)

    from RhoDesign_without2d import RhoDesignModel
    from alphabet import Alphabet
    from util import extract_coords_from_structure, load_structure

    model_args = ArgsClass(512, 512, 0.1)
    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    model = RhoDesignModel(model_args, alphabet).to(device)
    load_checkpoint(model, CHECKPOINT, device)
    model.eval()

    structure = load_structure(str(PDB_PATH))
    coords_np, native_seq_from_pdb = extract_coords_from_structure(structure)
    native_seq = normalize_seq(TARGET_SEQUENCE_OVERRIDE or native_seq_from_pdb)

    native_metrics = teacher_forced_metrics(model, alphabet, coords_np, native_seq, device)

    rows: List[Dict[str, object]] = []
    fasta_lines: List[str] = []
    

    for i in range(NUM_DESIGNS):
        design_id = f"{PDB_PATH.stem}_design{i + 1}"
        designed_seq = sample_sequence(
            model=model,
            alphabet=alphabet,
            coords_np=coords_np,
            device=device,
            temperature=TEMPERATURE,
            greedy=GREEDY_GENERATION,
            seed=SEED + i,
        )
        rec = seq_recovery(native_seq, designed_seq)

        row: Dict[str, object] = {
            "design_id": design_id,
            "family_label": FAMILY_LABEL,
            "pdb_path": str(PDB_PATH),
            "native_sequence": native_seq,
            "designed_sequence": designed_seq,
            "recovery_rate": rec,
            "temperature": TEMPERATURE,
            "greedy": GREEDY_GENERATION,
            "native_loss": native_metrics["loss"],
            "native_perplexity": native_metrics["perplexity"],
            "native_token_acc": native_metrics["token_acc"],
            "seq_len": len(native_seq),
        }
        """
        if RUN_CMSEARCH and CM_FILE:
            cm_result = run_cmsearch(CMSEARCH_EXE, Path(CM_FILE), designed_seq, cmsearch_dir, design_id)
            best_hit = cm_result.get("best_hit", {})
            row["cmsearch_returncode"] = cm_result.get("returncode")
            row["cmsearch_bit_score"] = best_hit.get("bit_score")
            row["cmsearch_evalue"] = best_hit.get("evalue")
            row["cmsearch_query_name"] = best_hit.get("query_name")
            row["cmsearch_target_name"] = best_hit.get("target_name")
            row["cmsearch_tblout"] = cm_result.get("tblout_path")
        
        if REFOLDED_PDB and i == 0:
            refolded_structure = load_structure(str(REFOLDED_PDB))
            refolded_coords, _ = extract_coords_from_structure(refolded_structure)
            rmsd_metrics = structure_consistency_metrics(coords_np, refolded_coords)
            row.update(rmsd_metrics)
            row["refolded_pdb"] = str(REFOLDED_PDB)
        """
        rows.append(row)
        fasta_lines.append(f">{design_id}")
        fasta_lines.append(designed_seq)

    df = pd.DataFrame(rows)
    csv_path = OUTDIR / "single_pdb_designs.csv"
    fasta_path = OUTDIR / "single_pdb_designs.fasta"
    summary_path = OUTDIR / "single_pdb_summary.json"

    df.to_csv(csv_path, index=False)
    with fasta_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(fasta_lines) + "\n")

    summary = {
        "pdb_path": str(PDB_PATH),
        "checkpoint": str(CHECKPOINT),
        "family_label": FAMILY_LABEL,
        "native_sequence": native_seq,
        "native_metrics": native_metrics,
        "best_design_by_recovery": None,
        "n_designs": int(len(df)),
    }
    if len(df) > 0:
        best_idx = df["recovery_rate"].astype(float).idxmax()
        summary["best_design_by_recovery"] = df.loc[int(best_idx)].to_dict()
        summary["mean_recovery_rate"] = float(df["recovery_rate"].mean())
        summary["median_recovery_rate"] = float(df["recovery_rate"].median())

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SINGLE PDB DESIGN SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Artifacts written to: {OUTDIR}")
    print(f" - {csv_path}")
    print(f" - {fasta_path}")
    print(f" - {summary_path}")
    


if __name__ == "__main__":
    main()
