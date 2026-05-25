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

Two paths, selected by the `family` string in `code/train/config.py`.
Both apply `COUNT_SCALE=1000` (organisms/μL → organisms/mL) as a shared first step for numerical stability (see LOG-024).

**Bernoulli path** (`load_and_binarise`):
```
raw CSV  →  drop all-zero rows  →  drop NaN rows  →  multiply by COUNT_SCALE=1000  →  binarise v > per-taxon median  →  split
```
Binarisation is rank-invariant under positive scaling; the stored thresholds are in organisms/mL.

**NB / ZINB path** (`load_raw_counts`):
```
raw CSV  →  drop all-zero rows  →  drop NaN rows  →  multiply by COUNT_SCALE=1000  →  split
```

NaN rows are retained separately as a structured post-training test set.

---

## Models

Model families are selected by the `family` string in `L_VALUES` (see `code/train/config.py`).  
All models share the `BaseRBM` initialisation (`W`, `a`, `b`, scale_init).  
Hidden monitoring is injected via mixins from `_hidden_monitors.py`.

### BB-RBM — Bernoulli-Bernoulli

| Component   | Specification                                            |
| ----------- | -------------------------------------------------------- |
| Visible     | Binary {0,1}, binarised at per-taxon median              |
| Hidden      | Bernoulli: P(h_j=1\|v) = σ(b_j + Σ_i W_ij v_i)          |
| Reconstruct | P(v_i=1\|h) = σ(a_i + Σ_j W_ij h_j)                     |
| L1 scope    | W, a, b                                                  |
| Monitor     | Reconstruction MSE · PLL (pseudo-log-likelihood)         |

### NB-RBM — Negative-Binomial with Bernoulli hidden

| Component        | Specification                                                  |
| ---------------- | -------------------------------------------------------------- |
| Visible          | NB: μ_i = exp(a_i + Σ_j W_ij h_j)                             |
| Dispersion       | θ_i = exp(log_θ_i), one per taxon, learned via autograd on positive-phase NLL |
| Hidden           | Bernoulli: P(h_j=1\|v) = σ(b_j + Σ_i W_ij v_i)                |
| L1 scope         | W only (a is log-mean baseline, not a logit parameter)         |
| Monitor          | Reconstruction MSE · NLL · θ_mean · sat_lo/sat_hi/sat_mid     |
| η clamp          | max=10.0 before exp (float32 overflow guard)                   |
| log_θ clamp      | [−10, 10] after each update                                    |
| θ gradient guard | nan_to_num(nan=0.0) before RMSprop step                        |
| Negative phase   | PCD-1: 500 persistent fantasy particles, initialised from X_train, advanced 1 Gibbs step per batch and stored back |

### NB-ReLU-RBM — NB visible, ReLU hidden (abandoned)

| Component    | Specification                                                     |
| ------------ | ----------------------------------------------------------------- |
| Hidden       | ReLU: pre_j = b_j + Σ_i W_ij v_i; h_j = clamp(relu(pre_j), 0, 5) |
| Sampling     | h_j ~ clamp(relu(pre_j + N(0,1)), 0, 5)                           |
| Monitor      | h_mean · h_sparsity (fraction of dead units)                      |
| Notes        | [0,5] clamp required for count-scale data; 6/10 seeds still diverge at L≥6. **Not recommended.** See LOG-020. |

### NB-Sigmoid-RBM — NB visible, Sigmoid hidden (recommended)

| Component    | Specification                                                     |
| ------------ | ----------------------------------------------------------------- |
| Hidden       | Sigmoid: P(h_j=1\|v) = σ(b_j + Σ_i W_ij v_i), h ∈ (0,1)          |
| Sampling     | h_j ~ Bernoulli(σ(pre_j))                                         |
| Monitor      | h_mean (target 0.2–0.8)                                           |
| Notes        | Bounded → PCD-safe. Lowest NLL of any tested family (0.443 ± 0.019 at L=7). **Recommended NB hidden type.** See LOG-021. |

### NB-Softmax-RBM — NB visible, Softmax hidden (abandoned)

| Component    | Specification                                                     |
| ------------ | ----------------------------------------------------------------- |
| Hidden       | Softmax: p_j = softmax_j(b + V@W), Σ_j p_j = 1                   |
| Sampling     | h ~ one-hot(multinomial(p))                                       |
| Monitor      | h_entropy (max log₂L for uniform)                                 |
| Notes        | Collapses to near-deterministic assignments (H≈0.05). Distributed signals discarded. **Not recommended.** See LOG-022. |

### ZINB-RBM — Zero-Inflated NB visible, Bernoulli hidden

| Component        | Specification                                                  |
| ---------------- | -------------------------------------------------------------- |
| Visible          | ZINB: P(v) = π·δ₀(v) + (1−π)·NB(v; μ, θ)                     |
| Dispersion       | θ_i = exp(log_θ_i), one per taxon                             |
| Inflation        | π_i = σ(logit_pi_i), one per taxon                            |
| Hidden           | Bernoulli (same as NB-RBM)                                     |
| Reconstruct      | E[v|h] = (1−π)·μ = (1−π)·exp(a + Wh)                          |
| Monitor          | Reconstruction MSE · NLL · θ_mean · π_mean · sat_lo/sat_hi/sat_mid |
 | L1 scope         | W only (a is log-mean baseline, same rationale as NB-RBM; see LOG-009) |
 | log_θ clamp      | [−10, 10] after each update                                    |
 | logit_pi clamp   | [−10, 10] after each update                                    |
 | Negative phase   | PCD-1 (same as NB-RBM)                                         |

### ZINB-ReLU-RBM — ZINB visible, ReLU hidden

| Component    | Specification                                                     |
| ------------ | ----------------------------------------------------------------- |
| Hidden       | ReLU with [0,5] clamp (same as NB-ReLU-RBM)                       |
| Notes        | Stable at L=4–5 only; variance increases at L≥6.                  |

### Shared training mechanics

| Parameter    | Value / formula                                              |
| ------------ | ------------------------------------------------------------ |
| Algorithm    | BB-RBM: CD-1 · NB/ZINB family: PCD-1 (500 persistent particles) |
| Optimiser    | RMSprop, β=0.9, ε=1e-4                                       |
| Batch size   | Annealed BATCH_I→BATCH_F quadratically over epochs           |
| LR schedule  | Multiplicative decay per epoch (LR × LR_DECAY)              |
| θ LR (NB)   | lr × 0.1, separate RMSprop accumulator                       |
| π LR (ZINB) | lr × 0.1, separate RMSprop accumulator                       |
| a init (BB)  | log(mean(v) / (1 − mean(v)))                                 |
| a init (NB)  | log(mean(v))                                                 |
| W init       | N(0, √(4/(D+L)))                                             |

---

## Training monitoring

| Model | Progress bar | CSV columns |
| ----- | ------------ | ----------- |
| BB | pll, val_pll | epoch, train_mse, val_mse, train_pll, val_pll |
| NB (Bernoulli hidden) | nll, val_nll, θ_mean, sat_mid | epoch, train_mse, val_mse, train_nll, val_nll, theta_mean, sat_lo, sat_hi, sat_mid |
| NB (Sigmoid hidden) | nll, val_nll, θ_mean, h_mean | epoch, train_mse, val_mse, train_nll, val_nll, theta_mean, h_mean |
| NB (Softmax hidden) | nll, val_nll, θ_mean, H | epoch, train_mse, val_mse, train_nll, val_nll, theta_mean, h_entropy |
| ZINB | nll, val_nll, θ_mean, π_mean, sat_mid | epoch, train_mse, val_mse, train_nll, val_nll, theta_mean, pi_mean, sat_lo, sat_hi, sat_mid |

**Hidden unit metrics** (set by `_hidden_monitors.py` mixins):
- `BernoulliHiddenMonitor`: sat_lo/ sat_hi/ sat_mid — fraction of P(h=1|v) < 0.1, > 0.9, rest
- `ReLUHiddenMonitor`: h_mean, h_sparsity — mean activation, fraction of dead (pre_act < 0) units
- `SigmoidHiddenMonitor`: h_mean — mean activation (target 0.2–0.8)
- `SoftmaxHiddenMonitor`: h_entropy — entropy of the categorical distribution

**PLL**: negative pseudo-log-likelihood = −mean_{n,i} log p(v_i | v_{-i}). Computed exactly via free-energy difference. Lower is better.

**NLL**: negative NB log-likelihood = −mean_{n,i} log NB(v_i; μ_i, θ_i). Lower is better.

> **Known issue — NLL scale-dependence.** The `lgamma(V+1)` term in the NB log-likelihood grows as `V·log(V)`, so NLL values depend on COUNT_SCALE even though COUNT_SCALE contributes zero to all gradients. Stored `val_nll` values are larger than the true deviance NLL by a fixed additive constant `lgamma(X_val+1).mean()` that can be computed from the data. NLL values are internally consistent (all runs use COUNT_SCALE=1000) and valid for model selection within the NB family. Cross-family comparison with PLL (which is scale-free) is not meaningful on absolute values. A full treatment is in `doc/nll_count_scale_issue.md`.

---

## Source layout

```
code/
  src/models/                   core library (RBM model classes, I/O, utils, plotting)
  train/                        "make me" pipeline
    config.py                     experiment hyperparameters + data locations + single-run flag
    train.py                      single-run (default) or multi-seed sweep trainer
  train/                        training pipeline — "make me a model"
    config.py                     experiment hyperparameters + data locations + single-run flag
    dataset_analysis.py           pre-training EDA → results/01_exploratory/
    train.py                      single-run (default) or multi-seed sweep trainer
  diagnostic/                   model evaluation — "did the training work?"
    sweep_analysis.py             L-sweep metrics → results/04_model_selection/
    split_comparison.py           split strategy comparison → results/03_evaluation/
    nan_test_eval.py              NaN imputation evaluation → diagnostic_outputs/nan_eval_extended/ + results/03_evaluation/
    plot_training_runs.py         training curves from trained_models/
  analysis/                     ecological interpretation — "what do the hidden units mean?"
    hidden_coactivation.py        weight profiles + state timelines → results/02_model_analysis/
    hidden_mean_activation.py     mean activation per unit → results/02_model_analysis/
    hidden_cross_model.py         NB↔BB cross-model comparison → results/02_model_analysis/ + tables/hidden/
    plot_visible_by_hidden.py     visible-unit probabilities per hidden node
    archetype_rbm_comparison.py   quantitative comparison with Cheng's archetypes
  archive/                      one-off plot scripts (gitignored)
    plot_final_metric_nb.py       final NLL vs L for NB+ZINB sigmoid/softmax
    plot_sigmoid_nll.py           nb_sigmoid train NLL curves
    plot_zinb_nll.py              ZINB sigmoid/softmax train NLL curves
```
