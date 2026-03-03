# Running ncRNA Generation on Google Colab

## Before you start

In Colab, go to **Runtime → Change runtime type** and select **GPU** (T4 or A100).

---

## Cell 1 — Clone the repository

```python
!git clone https://github.com/AlexMorman/ncrna_generation.git
%cd ncrna_generation
```

---

## Cell 2 — Install dependencies

PyTorch is pre-installed on Colab.  Install the remaining packages:

```python
# PyTorch Geometric
!pip install torch-geometric --quiet

# ViennaRNA Python bindings (for oracle filtering at inference time)
!pip install ViennaRNA --quiet

# Config and plotting
!pip install pyyaml matplotlib --quiet
```

> **Note:** If you see import errors for `torch_scatter` or `torch_sparse`
> after installing `torch-geometric`, run the cell below to install the
> optional compiled extensions matching your Colab CUDA version:
>
> ```python
> import torch
> v = torch.__version__.split('+')[0].replace('.', '')
> cuda = 'cu' + torch.version.cuda.replace('.', '')
> !pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{v}+{cuda}.html --quiet
> ```

---

## Cell 3 — Download RFAM training data (RF00001–RF00500)

RFAM seed alignments are available in Stockholm format from the EBI.
The cell below fetches the seed alignment for each family and saves it
directly into `data/raw/`. Downloading 500 families takes ~10–15 minutes
depending on connection speed; the loop is rate-limited to one request per
second to be polite to the EBI servers.

```python
import requests, pathlib, time

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ACCESSIONS = [f"RF{i:05d}" for i in range(1, 501)]  # RF00001 … RF00500

for acc in ACCESSIONS:
    out_path = RAW_DIR / f"{acc}.sto"
    if out_path.exists():
        print(f"{acc} already downloaded, skipping.")
        continue

    # RFAM REST API — seed alignment in Stockholm format
    url = f"https://rfam.xfam.org/family/{acc}/alignment?acc={acc}&format=stockholm&download=1"
    print(f"Downloading {acc} …", end=" ")
    # Added verify=False to bypass SSL certificate verification
    resp = requests.get(url, timeout=60, verify=False)
    if resp.status_code == 200:
        out_path.write_bytes(resp.content)
        print(f"saved ({len(resp.content) // 1024} KB)")
    else:
        print(f"FAILED (HTTP {resp.status_code}) — download manually from rfam.xfam.org")
    time.sleep(1)   # be polite to the EBI servers

print("\nFiles in data/raw/")
!ls -lh data/raw/
```

> **Alternative:** You can download files manually from
> [rfam.xfam.org](https://rfam.xfam.org), navigate to each family
> page → *Alignments* → *Seed (Stockholm)*, and upload them to
> `data/raw/` using the Colab file browser (left sidebar → Files icon).

---

## Cell 4 — Verify the data parses correctly

```python
from src.dataset import parse_stockholm_file
import pathlib

for sto_file in sorted(pathlib.Path("data/raw").glob("*.sto")):
    samples = parse_stockholm_file(str(sto_file))
    print(f"{sto_file.name}: {len(samples)} sequences, "
          f"e.g. len={len(samples[0][1])} — {samples[0][1][:20]}…")
```

Expected output (lengths vary by family):

```
RF00001.sto: 712 sequences, e.g. len=119 — GCCUACGGCCAUACCACGUU…
RF00002.sto: 954 sequences, e.g. len=71  — GCGGAUUUAGCUCAGUUGGG…
...
RF00500.sto: <N> sequences, e.g. len=<L> — …
```

> **Note:** Not all accessions in RF00001–RF00500 exist; the download cell
> will print `FAILED (HTTP 404)` for missing families and continue — this is
> expected.

---

## Cell 5 — Train

```python
!python train.py --config configs/config.yaml
```

> **Note:** The first run with 500 families will spend several minutes
> building the processed graph cache (`data/processed/data.pt`) before
> training begins. Subsequent runs load the cache instantly.

Training prints one line per epoch.  Teacher forcing (`TF`) is annealed
from 1.0 down to 0.1 over the run, and training stops automatically when
validation loss has not improved for 15 consecutive epochs:

```
Epoch   1 | Train 1.3862 | Val 1.3801 | Acc 0.2643 | LR 1.00e-03 | TF 1.00
  -> Saved best model (val_loss=1.3801)
Epoch   2 | Train 1.2104 | Val 1.1893 | Acc 0.3511 | LR 1.00e-03 | TF 0.99
  -> Saved best model (val_loss=1.1893)
...
Early stopping at epoch 42 (no improvement for 15 epochs).
```

The best checkpoint is saved to `best_model.pt` in the current directory.
Loss curves are saved to `loss_curves.png`.

---

## Cell 6 — View training visualizations

Training produces two image files. Display them to analyse the run:

### `loss_curves.png` — Loss and accuracy over epochs

```python
from IPython.display import Image, display
display(Image("loss_curves.png"))
```

Two panels:

| Panel | What it shows | What to look for |
|---|---|---|
| **Top — Loss** | Train loss (blue) and val loss (orange) per epoch. A dashed grey line marks the random-guessing baseline (ln 4 ≈ 1.386). | Val loss should trend *below* the baseline. A large train/val gap means overfitting. |
| **Bottom — Accuracy** | Validation accuracy per epoch. A dashed grey line marks the random baseline (0.25). | Accuracy should rise above 0.25 and plateau. |

---

### `diagnostics.png` — Post-training diagnostics (best checkpoint)

```python
display(Image("diagnostics.png"))
```

Four panels computed on the validation set using the best saved checkpoint:

| Panel | What it shows | What to look for |
|---|---|---|
| **Top-left — Confusion matrix** | Row-normalised 4×4 heatmap (A/U/G/C true vs predicted). Raw counts appear below each fraction. | A healthy model has a bright diagonal. A single bright column means mode collapse — the model predicts one nucleotide for everything. |
| **Top-right — Per-class accuracy** | Accuracy for each nucleotide separately vs the 0.25 random baseline. | Large imbalances (e.g. G/C accuracy high, A/U low) reveal class-specific failures. |
| **Bottom-left — Nucleotide frequency** | Grouped bars comparing the ground-truth nucleotide distribution (green) against the model's predicted distribution (blue). | Bars should roughly match. A blue bar that towers over its green partner signals a strong prediction bias. |
| **Bottom-right — Base-pair satisfaction** | Fraction of base-paired positions where the two predicted nucleotides form a valid Watson-Crick or G-U wobble pair, vs the random baseline (≈ 0.375) and a perfect score (1.0). | This is the primary domain metric. A model above 0.375 has learned *something* about RNA pairing; a model near 0.375 has not. |

---

## Cell 7 — Run inference (after training)

Generate candidate sequences for a target dot-bracket structure:

```python
!python inference.py \
    --config configs/config.yaml \
    --model_path best_model.pt \
    --structure "((((((.......))))))"
```

If ViennaRNA installed correctly the oracle will filter results by MFE
and structure similarity.  Pass `--no_oracle` to skip that step:

```python
!python inference.py \
    --config configs/config.yaml \
    --model_path best_model.pt \
    --structure "((((((.......))))))" \
    --no_oracle
```

---

## Saving your checkpoint to Google Drive

Colab sessions are ephemeral.  Save `best_model.pt` to Drive so it
persists between sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy("best_model.pt", "/content/drive/MyDrive/ncrna_best_model.pt")
print("Checkpoint saved to Google Drive.")
```

---

## Tuning hyperparameters

All hyperparameters live in `configs/config.yaml`.  Edit them directly
in Colab:

```python
# Example: double batch size, increase beam width, extend early stopping patience
import yaml, pathlib

cfg_path = pathlib.Path("configs/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["training"]["batch_size"] = 64
cfg["training"]["early_stopping_patience"] = 20
cfg["inference"]["beam_k"] = 10

cfg_path.write_text(yaml.dump(cfg))
print("Config updated.")
```

Then re-run Cell 5.

---

## Re-processing data

If you add new `.sto` files to `data/raw/` after the first training run,
delete the cache so it rebuilds:

```python
import pathlib
cache = pathlib.Path("data/processed/data.pt")
if cache.exists():
    cache.unlink()
    print("Cache cleared — re-run training to reprocess.")
```
