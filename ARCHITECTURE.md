# ARCHITECTURE.md

> Locked-in technical design decisions.
> Only add here once a choice is confirmed — not under active discussion.
> For open choices see DECISIONS.md.

---

## Dataset

| Property            | Value                                          |
| ------------------- | ---------------------------------------------- |
| Source              | `data/raw/TimeSeries_countsuL_clean.csv`                |
| Reference           | Eyring et al. (2025), *Scientific Data* 12:653 |
| n_visible           | 83 (taxa columns)                              |
| Temporal resolution | Daily (1 row per calendar day)                 |
| Date range          | 2019-03-21 → 2024-12-31                        |

---

## Preprocessing pipeline

Two paths, selected by `VISIBLE_MODEL` in `config.py`:

**Bernoulli path** (`load_and_binarise`):
```
raw CSV
  → drop all-zero rows              (instrument downtime)
  → drop rows with any NaN          (ML classifier outage blocks)
  → binarise: v > per-taxon median  (CLOSED: median > zero threshold)
  → chronological train/val split
```

**NB path** (`load_raw_counts`):
```
raw CSV
  → drop all-zero rows
  → drop rows with any NaN
  → multiply by COUNT_SCALE=1000    (organisms/μL → approximate integer counts)
  → chronological train/val split
```

**What is explicitly NOT done in both paths:**

- Row normalisation: rejected — introduces simplex constraint, discards biomass signal
- Random train/val split: rejected — temporal structure must be preserved
- Log-transform / z-score: not needed; NB handles raw counts directly, Bernoulli uses binarisation

**NaN rows post-training:** Reserved as structured test set (procedure TBD — clamp non-missing, reconstruct, score on non-missing only).

---

## Model architecture

Two visible unit models, both with Bernoulli hidden units. Selected by `VISIBLE_MODEL` in `config.py`.

### Bernoulli-Bernoulli RBM (`BernoulliRBM`)

- Visible: binary {0,1}, binarised at per-taxon median
- Hidden: Bernoulli, P(h_j=1|v) = σ(b_j + Σ_i W_ij v_i)
- Reconstruction: P(v_i=1|h) = σ(a_i + Σ_j W_ij h_j)
- Training: CD-k with RMSprop, L1 regularisation on W, a, b
- Monitor: reconstruction MSE + **PLL** (pseudo-log-likelihood, tractable exact proxy)

### NB-Bernoulli RBM (`NBRBM`)

- Visible: Negative Binomial, μ_i = exp(a_i + Σ_j W_ij h_j)
- Per-taxon dispersion θ_i = exp(log_θ_i), learned via autograd on positive-phase NLL
- Hidden: Bernoulli (same as above)
- Training: CD-k with RMSprop, L1 regularisation on W only (a is the log-mean baseline — shrinking it toward zero is incorrect)
- Monitor: reconstruction MSE + **NLL** (negative log-likelihood via lgamma)
- Numerical stability: η clamped at max=10.0 before exp (prevents float32 overflow with large L); log_θ clamped to [−10, 10]; NaN gradient guard on θ update

### Shared training mechanics

- Optimiser: RMSprop (β=0.9, ε=1e-4)
- Batch size: annealed from BATCH_I=10 to BATCH_F=256 quadratically over epochs
- LR decay: multiplicative per epoch (LR_DECAY=0.998)
- θ update (NB only): autograd on positive-phase NLL, separate LR (lr × 0.1)

---

## Source layout

```
src/
  config.py                  — all hyperparameters and paths
  main.py                    — training entry point
  data.py                    — load_and_binarise, load_raw_counts
  utils.py                   — device setup, weight save/load
  visualization.py           — training curves, weight heatmap, hidden activations
  models/
    base_rbm.py              — shared init and reconstruction_mse
    bernoulli_rbm.py         — BernoulliRBM with PLL
    nb_rbm.py                — NBRBM with NLL, θ update, numerical guards
  rbm_plankton.py            — legacy single-file version (kept for reference)

results/{model}_L{n_hidden}/ — one directory per run (e.g. nb_L5/, bernoulli_L10/)
  weights.npz
  rbm_training_curves.csv    — epoch, MSE, PLL/NLL, θ_mean, sat_lo/hi/mid
  rbm_weights.csv
  rbm_hidden_activations.csv
  rbm_training.png
  hidden_activations.png
```

## Training monitoring

| Model       | Progress bar       | Final metrics            | CSV columns                              |
| ----------- | ------------------ | ------------------------ | ---------------------------------------- |
| Bernoulli   | pll, val_pll       | MSE, PLL                 | train_mse, val_mse, train_pll, val_pll   |
| NB          | nll, val_nll, θ_mean, sat_mid | MSE, NLL, θ range, h saturation per-unit | train_mse, val_mse, train_nll, val_nll, theta_mean, sat_lo, sat_hi, sat_mid |

**Hidden unit saturation** (NB only): fraction of P(h=1|v) values below 0.1 (sat_lo), above 0.9 (sat_hi), and in between (sat_mid). Healthy binary model: sat_mid < 15%. Bias absorber detection: any unit with sat_hi > 90% across all timesteps is wasted capacity.
