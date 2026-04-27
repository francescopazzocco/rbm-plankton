# ARCHITECTURE.md

> Locked-in technical design decisions.
> Only add here once a choice is confirmed — not under active discussion.
> For open choices see DECISIONS.md.

---

## Dataset

| Property            | Value                                          |
| ------------------- | ---------------------------------------------- |
| Source              | `TimeSeries_countsuL_clean.csv`                |
| Reference           | Eyring et al. (2025), *Scientific Data* 12:653 |
| n_visible           | 83 (taxa columns)                              |
| Temporal resolution | Daily (1 row per calendar day)                 |
| Date range          | 2019-03-21 → 2024-12-31                        |

---

## Preprocessing pipeline (confirmed)

```
raw CSV
  → drop all-zero rows          (instrument downtime artifacts)
  → drop rows with any NaN      (ML classifier outage blocks)
  → log-transform: v ← log(v + ε)   (ε = TBD, see DECISIONS.md)
  → per-taxon z-score           (fit on train split only)
  → chronological train/val split
```

**What is explicitly NOT done:**

- Row normalisation (sum rows to 1): rejected — see CHANGELOG 2026-04-19
- Random train/val split: rejected — temporal structure must be preserved

**NaN rows post-training:** Used as structured test set. Procedure:

1. Clamp non-missing taxa as visible input
2. Run one Gibbs step to get reconstruction
3. Score reconstruction error on non-missing taxa only

---

## RBM base code

Adapted from `rbm_train_export.py` (originally MNIST/FPGA).

**Retained:**

- CD-k training engine
- RMSprop optimiser
- Training loop with batch size annealing
- Reconstruction MSE monitor (replaces PLL — PLL is binary-only)

**Removed:**

- FPGA export (`.coe`, `.bin`, `.npz` weight export)
- MNIST data loader
- Bernoulli visible unit math
- Pseudo-Log-Likelihood monitor

---

## Files

| File              | Purpose                      | Status      |
| ----------------- | ---------------------------- | ----------- |
| `preprocess.py`   | Full preprocessing pipeline  | Done        |
| `rbm_plankton.py` | Gaussian-Bernoulli RBM       | Not started |
| `train.py`        | Training entry point         | Not started |
| `analyse.py`      | Post-training interpretation | Not started |
| `DECISIONS.md`    | Open/closed branches         | Maintained  |
| `CHANGELOG.md`    | Chronological record         | Maintained  |
| `ARCHITECTURE.md` | This file                    | Maintained  |
