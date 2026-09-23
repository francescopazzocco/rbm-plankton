# NLL scale-dependence: the COUNT_SCALE=1000 problem

## Background

All NB-family models (NB_RBM, NB_ReLU_RBM, NBSigmoidRBM, NBSoftmaxRBM, ZINB_RBM, …) are trained on data multiplied by `COUNT_SCALE = 1000` (in `main_multiseed.py` and `nan_test_eval.py`).
The raw data is in organisms/μL; the scale converts to organisms/mL so that the model's `exp(a)` mean lives on a numerically stable scale.

## The problem

The NB log-likelihood used for reporting `train_nll` / `val_nll` is (per `_nb_log_prob` in `nb_rbm.py` and `zinb_rbm.py`):

```
log NB(v; μ, θ) = lgamma(v + θ) − lgamma(θ) − lgamma(v + 1)
                + θ·log(θ/(θ+μ)) + v·log(μ/(θ+μ))
```

The problematic term is **`lgamma(v + 1) = log(v!)`**.

By Stirling's approximation, for large v:

```
lgamma(v + 1) ≈ v·log(v) − v
```

When v = COUNT_SCALE · v_raw = 1000 · v_raw:

```
lgamma(1000·v_raw + 1) ≈ 1000·v_raw · log(1000·v_raw)
                        = 1000·v_raw · (log(1000) + log(v_raw))
                        ≈ 6908·v_raw  +  1000·v_raw·log(v_raw)
```

versus:

```
lgamma(v_raw + 1) ≈ v_raw·log(v_raw)
```

The difference is **not a fixed offset** — it grows with v_raw.  Different taxa and different time points contribute different corrections, so the NLL shifts by a data-dependent amount that cannot be removed by a single additive constant.

## Consequence

`val_nll` as stored in every CSV is:

```
val_nll_stored = val_nll_true  +  C(X_val, COUNT_SCALE)
```

where `C` depends on both the data and the scale constant.  Crucially:

- `C` is **zero** for gradients: `d/d(params) lgamma(V+1) = 0`, so **training is completely unaffected** — only the reported metric is wrong.
- Because `C` is nonzero and data-dependent, all stored `val_nll` values are inflated and **incomparable to any metric computed on unscaled data** (e.g., Bernoulli-RBM's `val_pll`, which is scale-free and bounded in [−log 2, 0] per unit).

## Fix (option B — deviance NLL)

Subtract the saturated-model constant from every NLL call:

```python
# in _nb_log_prob  (nb_rbm.py and zinb_rbm.py)
return (log_nb - torch.lgamma(V + 1)).mean()
# which simplifies the formula to:
# lgamma(v+θ) − lgamma(θ) − 2·lgamma(v+1) + ...   ← NO: wrong
```

More precisely, the corrected per-element log-likelihood is:

```
log NB_dev(v; μ, θ) = lgamma(v + θ) − lgamma(θ) − lgamma(v + 1)
                    + θ·log(θ/(θ+μ)) + v·log(μ/(θ+μ))
                    − lgamma(v + 1)          ← subtract again? NO
```

Wait — `lgamma(v+1)` already appears once with a minus sign.  "Subtracting it again" would double-count.  The correct deviance is defined relative to the **saturated model** (one parameter per observation, achieving maximum likelihood `log NB(v; v, θ→∞) → 0` in the limit). The standard approach is simply:

**Report the NLL without the `lgamma(V+1)` term** (i.e., drop the log-factorial entirely).  This gives the **unnormalized log-likelihood** (also called the kernel or the sufficient-statistic part):

```
log NB_kernel(v; μ, θ) = lgamma(v + θ) − lgamma(θ)
                        + θ·log(θ/(θ+μ)) + v·log(μ/(θ+μ))
```

This is:
- **Independent of COUNT_SCALE** (multiplying v by c changes lgamma(cv+θ) and v·log(μ/(θ+μ)), but those terms are already absorbed into μ since the model learns μ ≈ v anyway).
- **Still a valid training signal** for θ (the gradient is unchanged).
- **Comparable across runs** with different COUNT_SCALE values, as long as the model's μ is on the same scale as V.

### Implementation

In `nb_rbm.py`, change `_nb_log_prob`:

```python
# BEFORE
log_nb = (torch.lgamma(V + theta)
          - torch.lgamma(theta)
          - torch.lgamma(V + 1)          # ← remove this line
          + theta * torch.log(theta / (theta + mu + eps))
          + V     * torch.log(mu    / (theta + mu + eps)))

# AFTER
log_nb = (torch.lgamma(V + theta)
          - torch.lgamma(theta)
          + theta * torch.log(theta / (theta + mu + eps))
          + V     * torch.log(mu    / (theta + mu + eps)))
```

Same change in `zinb_rbm.py` inside `_zinb_log_prob` (lines 113–115).

### Do existing runs need to be rerun?

**No.**  Since `lgamma(V+1)` contributes zero to all gradients, the saved model weights are correct.  The stored `train_nll`/`val_nll` columns in existing CSVs can be corrected post-hoc:

```
corrected_val_nll = val_nll_stored − lgamma(X_val + 1).mean()
```

where `lgamma(X_val + 1).mean()` is a single scalar computed from the validation split (which is fixed and deterministic given the data and `val_frac`).  This correction can be applied by a small script without touching any model weights.
