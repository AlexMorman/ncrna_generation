#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(r"C:\Users\USER\OneDrive - University of Kansas\Rhofold Results")
OUTDIR = Path(r"C:\Users\USER\OneDrive - University of Kansas\rhodesign_manifests")
PDB_SUBDIR = "unrelaxed"          # using unrelaxed data
FAMILY_GLOB = "RF*.subseqs"       # matching my folders 
MAX_PER_FAMILY = 1000              # stop at 1000 per family since its more than that
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 7

GROUP_MAP_CSV = None               

# not necessary, but doesn't hurt to include
RNA_RESNAME_TO_BASE = {  
    "A": "A", "G": "G", "C": "C", "U": "U",
    "ADE": "A", "GUA": "G", "CYT": "C", "URA": "U",
    "DA": "A", "DG": "G", "DC": "C", "DT": "U",
}


@dataclass
class Record:
    family: str
    family_folder: str
    sample_id: str
    group_id: str
    pdb_path: str
    seq_len: int
    sequence: str


def load_group_map(path) -> Dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    sample_col = None
    for cand in ("sample_id", "stem", "id"):
        if cand in lower:
            sample_col = lower[cand]
            break
    if sample_col is None or "group_id" not in lower:
        raise ValueError("group_map CSV must contain sample_id/stem/id and group_id columns")
    return {str(row[sample_col]): str(row[lower["group_id"]]) for _, row in df.iterrows()}


def extract_sequence_from_pdb(pdb_path: Path) :
    residues: List[str] = []
    seen = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resid = (chain, resseq, icode, resname)
            if resid in seen:
                continue
            seen.add(resid)
            residues.append(RNA_RESNAME_TO_BASE.get(resname, "X"))
    seq = "".join(residues)
    if not seq:
        raise ValueError(f"Could not extract sequence from PDB: {pdb_path}")
    return seq


def discover_records(
    root: Path,
    family_glob: str,
    pdb_subdir: str,
    group_map: Dict[str, str],
    max_per_family: int,
    seed: int,
) :# returnsList[Record]
    rng = random.Random(seed)
    family_dirs = sorted([p for p in root.glob(family_glob) if p.is_dir()])
    if not family_dirs:
        raise FileNotFoundError(f"No family folders matched {family_glob!r} under {root}")

    all_records: List[Record] = []

    for family_dir in family_dirs:
        family = family_dir.name.split("_")[0]
        pdb_dir = family_dir / pdb_subdir
        if not pdb_dir.exists():
            print(f"[skip] missing folder: {pdb_dir}")
            continue

        pdb_files = sorted(pdb_dir.glob("*.pdb"))
        if not pdb_files:
            print(f"[skip] no pdb files in: {pdb_dir}")
            continue

        original_count = len(pdb_files)
        if max_per_family is not None and len(pdb_files) > max_per_family:
            pdb_files = sorted(rng.sample(pdb_files, k=max_per_family))

        print(f"[family] {family}: using {len(pdb_files)} / {original_count} PDBs from {pdb_dir}")

        for pdb_path in pdb_files:
            stem = pdb_path.stem
            try:
                seq = extract_sequence_from_pdb(pdb_path)
            except Exception as e:
                print(f"[warn] could not parse {pdb_path.name}: {e}")
                continue
            group_id = group_map.get(stem, stem)
            all_records.append(
                Record(
                    family=family,
                    family_folder=family_dir.name,
                    sample_id=stem,
                    group_id=group_id,
                    pdb_path=str(pdb_path.resolve()),
                    seq_len=len(seq),
                    sequence=seq,
                )
            )

    if not all_records:
        raise RuntimeError("No usable PDB examples were found.")
    return all_records


def grouped_split(
    records: Sequence[Record],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) :# returns Tuple[List[Record], List[Record], List[Record]]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    rng = random.Random(seed)
    by_family: Dict[str, List[Record]] = {}
    for rec in records:
        by_family.setdefault(rec.family, []).append(rec)

    train: List[Record] = []
    val: List[Record] = []
    test: List[Record] = []

    for family, fam_records in sorted(by_family.items()):
        groups: Dict[str, List[Record]] = {}
        for rec in fam_records:
            groups.setdefault(rec.group_id, []).append(rec)

        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        n = len(group_ids)

        if n == 1:
            train_ids, val_ids, test_ids = set(group_ids), set(), set()
        elif n == 2:
            train_ids, val_ids, test_ids = {group_ids[0]}, {group_ids[1]}, set()
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1

            train_ids = set(group_ids[:n_train])
            val_ids = set(group_ids[n_train:n_train + n_val])
            test_ids = set(group_ids[n_train + n_val:n_train + n_val + n_test])

        for gid in train_ids:
            train.extend(groups[gid])
        for gid in val_ids:
            val.extend(groups[gid])
        for gid in test_ids:
            test.extend(groups[gid])

        print(
            f"[split] {family}: groups train/val/test = "
            f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)} | "
            f"examples = {sum(len(groups[g]) for g in train_ids)}/"
            f"{sum(len(groups[g]) for g in val_ids)}/"
            f"{sum(len(groups[g]) for g in test_ids)}"
        )

    return train, val, test


def write_csv(path: Path, records: Sequence[Record]) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    if len(df) == 0:
        df = pd.DataFrame(columns=["family", "family_folder", "sample_id", "group_id", "pdb_path", "seq_len", "sequence"])
    else:
        df = df.sort_values(["family", "sample_id"]).reset_index(drop=True)
    df.to_csv(path, index=False)


def summarize(records: Sequence[Record]) -> Dict[str, object]:
    df = pd.DataFrame([asdict(r) for r in records])
    if len(df) == 0:
        return {"n_records": 0, "families": [], "per_family": {}}
    return {
        "n_records": int(len(df)),
        "families": sorted(df["family"].astype(str).unique().tolist()),
        "per_family": {str(k): int(v) for k, v in df.groupby("family").size().to_dict().items()},
        "seq_len": {
            "min": int(df["seq_len"].min()),
            "median": float(df["seq_len"].median()),
            "mean": float(df["seq_len"].mean()),
            "max": int(df["seq_len"].max()),
        },
    }


def main() -> None:
    print("ROOT:", ROOT)
    print("OUTDIR:", OUTDIR)
    print("PDB_SUBDIR:", PDB_SUBDIR)
    print("MAX_PER_FAMILY:", MAX_PER_FAMILY)

    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    group_map = load_group_map(GROUP_MAP_CSV)

    records = discover_records(
        root=ROOT,
        family_glob=FAMILY_GLOB,
        pdb_subdir=PDB_SUBDIR,
        group_map=group_map,
        max_per_family=MAX_PER_FAMILY,
        seed=SEED,
    )

    train_records, val_records, test_records = grouped_split(
        records,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    write_csv(OUTDIR / "manifest.csv", records)
    write_csv(OUTDIR / "train.csv", train_records)
    write_csv(OUTDIR / "val.csv", val_records)
    write_csv(OUTDIR / "test.csv", test_records)

    summary = {
        "config": {
            "root": str(ROOT),
            "outdir": str(OUTDIR),
            "pdb_subdir": PDB_SUBDIR,
            "family_glob": FAMILY_GLOB,
            "max_per_family": MAX_PER_FAMILY,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "seed": SEED,
            "group_map_csv": None if GROUP_MAP_CSV is None else str(GROUP_MAP_CSV),
        },
        "manifest": summarize(records),
        "train": summarize(train_records),
        "val": summarize(val_records),
        "test": summarize(test_records),
    }

    with (OUTDIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print("Wrote:")
    print(" -", OUTDIR / "manifest.csv")
    print(" -", OUTDIR / "train.csv")
    print(" -", OUTDIR / "val.csv")
    print(" -", OUTDIR / "test.csv")
    print(" -", OUTDIR / "summary.json")


if __name__ == "__main__":
    main()
