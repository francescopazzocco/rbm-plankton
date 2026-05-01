# ARCHITECTURE.md

> Technical specification of what is built.
> No rationale, no open questions — those live in DECISION_LOG.md and ROADMAP.md.
> Update only when implementation changes.

---

## Dataset

| Property            | Value                                          |
| ------------------- | ---------------------------------------------- |
| Source              | `data/raw/TimeSeries_countsuL_clean.csv`       |
| Reference           | Eyring et al. (2025), [*Scientific Data* 12:653](https://doi.org/10.1038/s41597-025-04988-9) |
| n_visible (D)       | 83 taxa columns                                |
| Temporal resolution | Daily                                          |
| Date range          | 2019-03-21 → 2024-12-31                        |
| Clean rows          | 1906 (after dropping all-zero and NaN rows)    |
| Train / val split   | 85% / 15%, chronological                       |

---

## Preprocessing

Two paths, selected by `VISIBLE_MODEL` in `config.py`.

**Bernoulli path** (`load_and_binarise`):
```
raw CSV  →  drop all-zero rows  →  drop NaN rows  →  binarise v > per-taxon median  →  split
```

**NB path** (`load_raw_counts`):
```
raw CSV  →  drop all-zero rows  →  drop NaN rows  →  multiply by COUNT_SCALE=1000  →  split
```

NaN rows are retained separately as a structured post-training test set.

---

## Models

Both models use Bernoulli hidden units. Selected by `VISIBLE_MODEL` in `config.py`.

### BB-RBM — Bernoulli-Bernoulli

| Component   | Specification                                            |
| ----------- | -------------------------------------------------------- |
| Visible     | Binary {0,1}, binarised at per-taxon median              |
| Hidden      | Bernoulli: P(h_j=1\|v) = σ(b_j + Σ_i W_ij v_i)          |
| Reconstruct | P(v_i=1\|h) = σ(a_i + Σ_j W_ij h_j)                     |
| L1 scope    | W, a, b                                                  |
| Monitor     | Reconstruction MSE · PLL (pseudo-log-likelihood)         |

### NBB-RBM — Negative-Binomial–Bernoulli

| Component        | Specification                                                  |
| ---------------- | -------------------------------------------------------------- |
| Visible          | NB: μ_i = exp(a_i + Σ_j W_ij h_j)                             |
| Dispersion       | θ_i = exp(log_θ_i), one per taxon, learned via autograd on positive-phase NLL |
| Hidden           | Bernoulli (same as BB-RBM)                                     |
| L1 scope         | W only (a is log-mean baseline, not a logit parameter)         |
| Monitor          | Reconstruction MSE · NLL · θ_mean · hidden unit saturation     |
| η clamp          | max=10.0 before exp (float32 overflow guard)                   |
| log_θ clamp      | [−10, 10] after each update                                    |
| θ gradient guard | nan_to_num(nan=0.0) before RMSprop step                        |
| Negative phase   | PCD-1: 500 persistent fantasy particles, initialised from X_train, advanced 1 Gibbs step per batch and stored back |

### Shared training mechanics

| Parameter    | Value / formula                                              |
| ------------ | ------------------------------------------------------------ |
| Algorithm    | BB-RBM: CD-1 · NBB-RBM: PCD-1 (500 persistent particles)    |
| Optimiser    | RMSprop, β=0.9, ε=1e-4                                       |
| Batch size   | Annealed BATCH_I→BATCH_F quadratically over epochs           |
| LR schedule  | Multiplicative decay per epoch (LR × LR_DECAY)              |
| θ LR (NB)   | lr × 0.1, separate RMSprop accumulator                       |
| a init (BB)  | log(mean(v) / (1 − mean(v)))                                 |
| a init (NB)  | log(mean(v))                                                 |
| W init       | N(0, √(4/(D+L)))                                             |

---

## Training monitoring

| Model | Progress bar              | CSV columns                                               |
| ----- | ------------------------- | --------------------------------------------------------- |
| BB    | pll, val_pll              | epoch, train_mse, val_mse, train_pll, val_pll             |
| NB    | nll, val_nll, θ_mean, sat_mid | epoch, train_mse, val_mse, train_nll, val_nll, theta_mean, sat_lo, sat_hi, sat_mid |

**Hidden unit saturation** (NB): sat_lo = fraction of P(h=1|v) < 0.1; sat_hi = fraction > 0.9; sat_mid = remainder. Target: sat_mid < 15% at convergence. A unit with sat_hi > 90% across all timesteps is a bias absorber (wasted capacity).

**PLL**: negative pseudo-log-likelihood = −mean_{n,i} log p(v_i | v_{-i}). Computed exactly via free-energy difference. Lower is better.

**NLL**: negative NB log-likelihood = −mean_{n,i} log NB(v_i; μ_i, θ_i). Lower is better.

---

## Source layout

```
src/
  main_multiseed.py       training pipeline — parallel N-seed runs per (family, L)
  dataset_analysis.py     EDA pipeline — figures → results/figures/dataset_analysis/
  sweep_analysis.py       L-sweep pipeline — figures → results/figures/sweep/
  hidden_coactivation.py  hidden analysis pipeline — weight profiles + state timelines
  hidden_mean_activation.py  hidden analysis pipeline — mean activation per unit
  hidden_cross_model.py   hidden analysis pipeline — NB↔BB cross-model comparison
  models/
    io.py                 file I/O: training data loaders + results navigation
                            (load_and_binarise, load_raw_counts, best_seed_dir, METRIC_COL)
    utils.py              shared utilities: get_device, save_weights, load_weights
    visualization.py      all plotting functions, organised by calling pipeline
    base_rbm.py           shared RBM interface and initialisation
    bernoulli_rbm.py      BernoulliRBM: train (CD-1), pll, hidden_probs, reconstruct
    nb_rbm.py             NBRBM: train (PCD-1), nll, hidden_probs, reconstruct, θ update

results/
  training_runs/{family}_L{n}/seed_{k}/   training artifacts (canonical: multiseed PCD runs)
    weights.npz
    rbm_training_curves.csv
    rbm_weights.csv
    rbm_hidden_activations.csv
    train.log
  figures/
    training_runs/     training curves and weight heatmaps (populated when PLOT_RESULTS=True)
    dataset_analysis/  EDA figures from dataset_analysis.py
    hidden/            hidden activation analysis outputs
    sweep/             L-sweep analysis figures
```
