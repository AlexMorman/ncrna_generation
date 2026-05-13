#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(r"C:\Users\USER\Downloads\RhoDesign-main")
TRAIN_CSV = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_manifests\train.csv")
VAL_CSV = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_manifests\val.csv")
OUTDIR = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_runs\shared_all_families")
CHECKPOINT = Path(r"C:\Users\USER\Downloads\no_ss_apexp_best.pth")


# Example: FAMILIES = ["RF04135"]
FAMILIES: List[str] = []

EPOCHS = 15
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 3e-5
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
ENCODER_FREEZE_EPOCHS = 2
BALANCE_FAMILIES = True
AMP = True
NUM_WORKERS = 0     
SEED = 7
DEVICE = "cuda"   
LOG_EVERY = 25
SAVE_EVERY_EPOCH = False



class ArgsClass:
    """Matches the public repo's inference_without2d.py defaults."""

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


def load_pretrained(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    obj = torch.load(checkpoint_path, map_location=device)
    state_dict = maybe_unwrap_state_dict(obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] checkpoint: {checkpoint_path}")
    print(f"[load] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("[load] first missing keys:", missing[:10])
    if unexpected:
        print("[load] first unexpected keys:", unexpected[:10])


class RhoDesignCSVWithout2D(Dataset):
    def __init__(self, csv_path: Path, families: Optional[Sequence[str]] = None):
        self.df = pd.read_csv(csv_path)
        required = {"family", "sample_id", "pdb_path", "sequence"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        if families:
            fam_set = set(map(str, families))
            self.df = self.df[self.df["family"].astype(str).isin(fam_set)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows left after filtering {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        return {
            "family": str(row["family"]),
            "sample_id": str(row["sample_id"]),
            "pdb_path": str(row["pdb_path"]),
            "sequence": str(row["sequence"]),
        }

def save_training_plots(
    history: List[Dict[str, float]],
    outdir: Path,
    unfreeze_epoch: Optional[int] = None,
):
    if not history:
        return

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(plots_dir / "history.csv", index=False)

    def add_unfreeze_marker():
        if unfreeze_epoch is not None:
            plt.axvline(unfreeze_epoch, linestyle="--", label="encoder_unfreeze")

    # 1) Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")
    add_unfreeze_marker()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 2) Token accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_token_acc"], marker="o", label="train_token_acc")
    plt.plot(df["epoch"], df["val_token_acc"], marker="o", label="val_token_acc")
    add_unfreeze_marker()
    plt.xlabel("Epoch")
    plt.ylabel("Token Accuracy")
    plt.title("Training and Validation Token Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "token_accuracy_curve.png", dpi=200)
    plt.close()

    # 3) Learning rate curve
    if "decoder_lr" in df.columns or "encoder_lr" in df.columns:
        plt.figure(figsize=(8, 5))
        if "decoder_lr" in df.columns:
            plt.plot(df["epoch"], df["decoder_lr"], marker="o", label="decoder_lr")
        if "encoder_lr" in df.columns and df["encoder_lr"].notna().any():
            plt.plot(df["epoch"], df["encoder_lr"], marker="o", label="encoder_lr")
        add_unfreeze_marker()
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "lr_curve.png", dpi=200)
        plt.close()

@dataclass
class BatchMeta:
    families: List[str]
    sample_ids: List[str]
    seqs: List[str]


class Without2DCollator:
    def __init__(self, alphabet, batch_converter, load_structure_fn, extract_coords_fn, device: torch.device):
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        self.load_structure_fn = load_structure_fn
        self.extract_coords_fn = extract_coords_fn
        self.device = device

    def __call__(self, batch: Sequence[Dict[str, object]]):
        raw_batch = []
        families: List[str] = []
        sample_ids: List[str] = []
        seqs: List[str] = []

        for item in batch:
            pdb_path = str(item["pdb_path"])
            seq = str(item["sequence"]).replace("T", "U")
            seq = "".join(ch if ch in {"A", "G", "C", "U", "X"} else "X" for ch in seq)

            structure = self.load_structure_fn(pdb_path)
            coords, pdb_seq = self.extract_coords_fn(structure)
            pdb_seq = pdb_seq.replace("T", "U")
            pdb_seq = "".join(ch if ch in {"A", "G", "C", "U", "X"} else "X" for ch in pdb_seq)

            # Use manifest sequence as target, but keep a sanity check.
            if len(seq) != len(pdb_seq):
                raise ValueError(
                    f"Length mismatch for {pdb_path}: manifest sequence has len {len(seq)} but PDB has len {len(pdb_seq)}"
                )

            raw_batch.append((coords, None, seq, None))
            families.append(str(item["family"]))
            sample_ids.append(str(item["sample_id"]))
            seqs.append(seq)

        batch_coords, confidence, _, tokens, padding_mask, _ = self.batch_converter(raw_batch, device=self.device)
        coords = batch_coords[:, :, [0, 1, 2], :]
        adjunct_coords = batch_coords
        meta = BatchMeta(families=families, sample_ids=sample_ids, seqs=seqs)
        return coords, adjunct_coords, confidence, tokens, padding_mask, meta


def family_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df.groupby("family").size().to_dict()
    weights = [1.0 / counts[str(f)] for f in df["family"].astype(str).tolist()]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, alphabet, device, amp: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for coords, adjunct_coords, confidence, tokens, padding_mask, _meta in loader:
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        target_padding_mask = target == alphabet.padding_idx

        with autocast(enabled=amp and device.type == "cuda"):
            logits, _ = model(
                coords,
                adjunct_coords,
                padding_mask.bool(),
                confidence,
                prev_output_tokens,
            )
            loss = F.cross_entropy(logits, target, reduction="none")

        valid = ~target_padding_mask
        total_loss += float(loss[valid].sum().item())
        total_tokens += int(valid.sum().item())
        pred = logits.argmax(dim=1)
        total_correct += int(((pred == target) & valid).sum().item())

    if total_tokens == 0:
        return {"loss": float("nan"), "token_acc": float("nan")}
    return {
        "loss": total_loss / total_tokens,
        "token_acc": total_correct / total_tokens,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    alphabet,
    device,
    amp: bool,
    grad_accum: int,
    max_grad_norm: float,
    epoch_idx: int,
    log_every: int,
) : # returns Dict[str, float]
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (coords, adjunct_coords, confidence, tokens, padding_mask, _meta) in enumerate(loader, start=1):
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        target_padding_mask = target == alphabet.padding_idx

        with autocast(enabled=amp and device.type == "cuda"):
            logits, _ = model(
                coords,
                adjunct_coords,
                padding_mask.bool(),
                confidence,
                prev_output_tokens,
            )
            #print(logits.shape)
            loss_per_token = F.cross_entropy(logits, target, reduction="none")
            valid = ~target_padding_mask
            loss = loss_per_token[valid].mean()
            loss_to_backprop = loss / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        total_loss += float(loss_per_token[valid].sum().item())
        total_tokens += int(valid.sum().item())
        pred = logits.argmax(dim=1)
        total_correct += int(((pred == target) & valid).sum().item())

        if step % grad_accum == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0:
            mean_loss = total_loss / max(total_tokens, 1)
            mean_acc = total_correct / max(total_tokens, 1)
            print(
                f"epoch={epoch_idx} step={step}/{len(loader)} "
                f"loss={mean_loss:.4f} token_acc={mean_acc:.4f}"
            )

    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_acc": total_correct / max(total_tokens, 1),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, metrics: Dict[str, float], cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    set_seed(SEED)
    device = get_device(DEVICE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ensure_repo_imports(REPO_ROOT)

    from RhoDesign_without2d import RhoDesignModel
    from alphabet import Alphabet
    from util import CoordBatchConverter, extract_coords_from_structure, load_structure

    families = FAMILIES if FAMILIES else None
    train_ds = RhoDesignCSVWithout2D(TRAIN_CSV, families=families)
    val_ds = RhoDesignCSVWithout2D(VAL_CSV, families=families)

    model_args = ArgsClass(512, 512, 0.1)
    alphabet = Alphabet(["A", "G", "C", "U", "X"])
    batch_converter = CoordBatchConverter(alphabet)
    collator = Without2DCollator(
        alphabet=alphabet,
        batch_converter=batch_converter,
        load_structure_fn=load_structure,
        extract_coords_fn=extract_coords_from_structure,
        device=device,
    )

    train_sampler = None
    shuffle = True
    if BALANCE_FAMILIES and families is None and len(train_ds.df["family"].unique()) > 1:
        train_sampler = family_sampler(train_ds.df)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    model = RhoDesignModel(model_args, alphabet).to(device)
    load_pretrained(model, CHECKPOINT, device)

    if ENCODER_FREEZE_EPOCHS > 0:
        set_encoder_trainable(model, False)

    optimizer = torch.optim.AdamW([
        {"params": model.decoder.parameters(), "lr": LR},
        {"params": model.encoder.parameters(), "lr": 0.0},
    ], weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=AMP and device.type == "cuda")

    cfg = {
        "repo_root": str(REPO_ROOT),
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "outdir": str(OUTDIR),
        "checkpoint": str(CHECKPOINT),
        "families": list(FAMILIES),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "encoder_freeze_epochs": ENCODER_FREEZE_EPOCHS,
        "balance_families": BALANCE_FAMILIES,
        "amp": AMP,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "device_requested": DEVICE,
        "device_resolved": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "train_families": sorted(train_ds.df["family"].astype(str).unique().tolist()),
        "val_families": sorted(val_ds.df["family"].astype(str).unique().tolist()),
    }
    with open(OUTDIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(json.dumps({
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "trainable_params": count_trainable_params(model),
        "families": cfg["train_families"],
        "device": str(device),
    }, indent=2))

    best_val_loss = math.inf
    history: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        if epoch == ENCODER_FREEZE_EPOCHS + 1 and ENCODER_FREEZE_EPOCHS > 0:
            set_encoder_trainable(model, True)
            optimizer.param_groups[1]["lr"] = LR * 0.5
            print(f"[epoch {epoch}] unfroze encoder; trainable_params={count_trainable_params(model)}")
            
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            alphabet=alphabet,
            device=device,
            amp=AMP,
            grad_accum=GRAD_ACCUM,
            max_grad_norm=MAX_GRAD_NORM,
            epoch_idx=epoch,
            log_every=LOG_EVERY,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            alphabet=alphabet,
            device=device,
            amp=AMP,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_token_acc": train_metrics["token_acc"],
            "val_loss": val_metrics["loss"],
            "val_token_acc": val_metrics["token_acc"],
            "decoder_lr": optimizer.param_groups[0]["lr"],
            "encoder_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None,
        }
        history.append(row)
        save_training_plots(
            history=history,
            outdir=OUTDIR,
            unfreeze_epoch=(ENCODER_FREEZE_EPOCHS + 1) if ENCODER_FREEZE_EPOCHS > 0 else None,
        )
        with open(OUTDIR / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(json.dumps(row, indent=2))

        save_checkpoint(OUTDIR / "last.pt", model, optimizer, epoch, row, cfg)
        if SAVE_EVERY_EPOCH:
            save_checkpoint(OUTDIR / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, row, cfg)

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            save_checkpoint(OUTDIR / "best.pt", model, optimizer, epoch, row, cfg)
            print(f"[epoch {epoch}] saved new best checkpoint with val_loss={best_val_loss:.4f}")

    with open(OUTDIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Artifacts written to: {OUTDIR}")


if __name__ == "__main__":
    main()
