# Training Improvement Plan

Analysis of `context/loss_curves.png` (100 epochs, RF00001–RF00010) revealed
severe overfitting: train loss converged to ~1.04 while validation loss
plateaued at ~1.50 — **worse than the random-guessing baseline of ln(4) = 1.386**.
The model memorises its training set and cannot generalise.

The changes below are ordered by expected impact. Each is a self-contained
action item that can be executed independently (though some interact).

---

## Action 1 — Scale training data to 500 RFAM families

**Problem:** 11 RFAM families is far too little data for a model with ~200K+
parameters. The model memorises instead of learning generalisable
structure→sequence preferences.

**Changes:**
- `COLAB_SETUP.md` Cell 3: change the download range from
  `range(1, 11)` to `range(1, 501)` (RF00001–RF00500).
- Add a `time.sleep(1)` already present — confirm it stays to respect EBI rate
  limits.
- `COLAB_SETUP.md` Cell 4: update the "expected output" example to reflect
  ~500 families.
- `COLAB_SETUP.md` Cell 5: add a note that first training run on 500 families
  will take longer to process.
- `README.md`: update to reflect recommended training data scale.

**No source code changes required** — the dataset/training pipeline already
handles arbitrary numbers of `.sto` files.

---

## Action 2 — Add early stopping to the training loop

**Problem:** Validation loss plateaus by epoch ~10–15 then worsens. Training
for 100 epochs deepens overfitting with no benefit.

**Changes:**
- `configs/config.yaml`: add `early_stopping_patience: 15` under `training`.
- `train.py`: implement early stopping logic — track epochs since last
  val_loss improvement; break the loop when patience is exceeded. Print a
  message like `"Early stopping at epoch {epoch} (patience={patience})"`.
- `COLAB_SETUP.md` Cell 5: update the example training output to show early
  stopping triggering.
- `CLAUDE.md`: mention early stopping in the execution flow or training
  description.

---

## Action 3 — Increase regularisation (dropout + weight decay)

**Problem:** `dropout: 0.1` and `weight_decay: 1e-5` are too weak. The model
overfits freely.

**Changes:**
- `configs/config.yaml`:
  - `model.encoder.dropout`: `0.1` → `0.3`
  - `model.decoder.dropout`: `0.1` → `0.3`
  - `training.weight_decay`: `1.0e-5` → `1.0e-3`

**No source code changes required** — values are already read from config.

---

## Action 4 — Scheduled teacher forcing annealing

**Problem:** Fixed `teacher_forcing_ratio: 0.5` creates exposure bias — during
training the decoder sees ground-truth tokens half the time, during evaluation
it sees none. This inflates the train/val gap.

**Changes:**
- `configs/config.yaml`: replace `teacher_forcing_ratio: 0.5` with:
  ```yaml
  teacher_forcing_start: 1.0
  teacher_forcing_end: 0.1
  ```
- `train.py`: compute `tf_ratio` per epoch via linear annealing:
  ```python
  tf_ratio = tf_start - (tf_start - tf_end) * (epoch - 1) / max(num_epochs - 1, 1)
  ```
  Print the current `tf_ratio` in the per-epoch log line.
- `CLAUDE.md`: update the training description to mention scheduled teacher
  forcing.
- `COLAB_SETUP.md` Cell 5: update example output to show `TF` column.

---

## Action 5 — Reduce model capacity

**Problem:** 128-dim hidden, 3 encoder layers, 2 decoder layers is
overparameterised for the available data. Even with more data, a smaller model
trains faster and generalises better when data is limited.

**Changes:**
- `configs/config.yaml`:
  - `model.encoder.hidden_dim`: `128` → `64`
  - `model.encoder.num_heads`: `4` → `4` (keep — 64/4 = 16 per head is fine)
  - `model.encoder.num_layers`: `3` → `2`
  - `model.decoder.hidden_dim`: `128` → `64`
  - `model.decoder.num_layers`: `2` → `1`

**No source code changes required** — architecture reads all dims from config.

---

## Action 6 — Randomised train/val split

**Problem:** The current sequential split (`dataset[:split]` /
`dataset[split:]`) puts entire RFAM families into one split. Validation tests
cross-family generalisation, which is much harder and arguably not the right
evaluation for early development.

**Changes:**
- `train.py`: replace the sequential slice with a shuffled split:
  ```python
  from torch.utils.data import random_split
  generator = torch.Generator().manual_seed(config.get("seed", 42))
  train_dataset, val_dataset = random_split(dataset, [split, n - split], generator=generator)
  ```
  Ensure reproducibility via the config seed.
- `CLAUDE.md`: note that train/val split is random (seeded).

---

## Documentation updates (cross-cutting)

After all action items above are complete, do a final pass over:

- **`README.md`** — update to reflect recommended data scale (500 families),
  training improvements (early stopping, TF annealing), and current model
  status.
- **`COLAB_SETUP.md`** — ensure all cells, example outputs, and instructions
  match the new config values and training behaviour.
- **`CLAUDE.md`** — ensure the architecture description, execution flow, and
  key constraints sections reflect all code changes (early stopping,
  TF annealing schedule, random split).
- **`configs/config.yaml`** — final review that all values match the plan.

---

## Execution order

Actions 1, 3, and 5 are config-only changes and can be done together.
Action 2 (early stopping) and Action 4 (TF annealing) modify `train.py` and
should be done carefully to avoid merge conflicts — but they touch different
parts of the training loop so can be done in either order.
Action 6 (random split) is a small isolated change in `train.py`.

Suggested order: **3 → 5 → 1 → 2 → 4 → 6 → docs pass**.
