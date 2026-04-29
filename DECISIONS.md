# DECISIONS.md

> Current state of all branches — what is open, what is closed, what is pending.
> For the rationale behind each decision see CHANGELOG.md.
> For locked-in technical specs see ARCHITECTURE.md.

---

## CLOSED — Preprocessing

| Decision                | Choice                        | Reason                                                                                              |
| ----------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------- |
| All-zero rows           | Drop                          | Multi-day consecutive runs → instrument downtime, not ecology                                       |
| NaN rows                | Drop for training             | Structured block outages (ML classifier updates), not random                                        |
| Row normalisation       | Do NOT apply for RBM          | No technical benefit; introduces simplex constraint RBM cannot model; discards total-biomass signal |
| NaN rows post-training  | Reserved as structured test set | Clamp non-missing taxa, reconstruct missing, score on non-missing                                 |
| Visible layer transform (Bernoulli) | Binarise at per-taxon median | Median > zero threshold: avoids constant units for rare taxa; validated empirically |
| Visible layer transform (NB) | Raw counts × COUNT_SCALE=1000 | NB handles distribution directly; scaling to approximate integer counts required for lgamma stability |
| Log-transform / z-score | Not applied in either path    | Bernoulli uses binarisation; NB models the raw count distribution natively                          |

---

## CLOSED — Data understanding

| Finding                                           | Status                                                                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Row sum dominated by biology not instrument drift | Confirmed via time series                                                                         |
| No monotonic multi-year trend                     | Confirmed — annual median stable 2019-2024                                                        |
| 1740-day Lomb-Scargle peak                        | Artifact of 2022-2023 extreme events, not a real cycle                                            |
| Dominant period                                   | ~365 days (annual cycle) confirmed                                                                |
| January-February 2023 anomaly                     | Row sums 2 orders of magnitude above other years in same months — flagged, not yet resolved       |
| NaN structure                                     | Block outages affecting same 31 taxa simultaneously; consistent with classifier retraining events |
| October 2022                                      | All 31 days have same 31 taxa NaN — entire month lost under drop strategy                         |

---

## CLOSED — Model architecture

### Visible unit type

**STATUS**: CLOSED — two parallel implementations, both retained

| Model | Input | Distribution | Status |
|---|---|---|---|
| Bernoulli-Bernoulli | Binary (median threshold) | Bernoulli | Validated |
| NB-Bernoulli | Raw counts × 1000 | Negative Binomial | Experimental, promising |

Gaussian and Beta visible units deprioritised: Gaussian requires z-score (discards zero structure); Beta requires row normalisation (rejected).

### Hidden unit type

**STATUS**: CLOSED — Bernoulli hidden units confirmed

L=5 NB run: sat_mid = 3% at convergence (97% of activations are near 0 or 1). Discrete community states are the correct inductive bias for this dataset. Gaussian hidden units not warranted.

### NB-specific: COUNT_SCALE

**STATUS**: CLOSED — COUNT_SCALE=1000

COUNT_SCALE=1.0 (organisms/μL, values in [0, 0.44]) caused degenerate training: θ stuck at init, MSE near zero (mean-collapse), CD gradient near zero. Root cause: NB lgamma gradient collapses when all values ≪ 1. COUNT_SCALE=1000 brings data to approximate integer-count scale and resolves all three failure modes.

### NB-specific: L1 regularisation scope

**STATUS**: CLOSED — L1 on W only for NBRBM

`a` in NBRBM is the log-mean baseline (μ_i = exp(a_i + W_i·h)). Applying L1 to `a` biases all means toward exp(0)=1, which is ecologically wrong. L1 applies to W only. Bernoulli model regularises W, a, b (correct — all are logit-scale parameters).

### NB-specific: numerical stability

**STATUS**: CLOSED — three guards implemented

| Guard | Value | Reason |
|---|---|---|
| η clamp in `_mu` | max=10.0 | exp(η) overflows float32 at η>88; with L=10, multiple hidden units can sum weights to exceed this |
| log_θ clamp | [−10, 10] | Prevents runaway accumulation; θ range [4.5e-5, 22026] covers all ecological scenarios |
| Gradient nan_to_num | nan→0.0 | Catches residual NaN from inf in log-likelihood; zeroes the step rather than corrupting θ |

## OPEN — Architecture

### n_hidden

**STATUS**: OPEN — sweep in progress

- L=5 validated: 4 seasonal community states + 1 bias absorber (h1 always-on, wasted capacity)
- Ecological states identified: early spring (h4), summer cyanobacteria/zooplankton (h0), peak summer (h2), diatom-associated summer (h3)
- Sweep [3, 5, 7, 10] initiated to cross-validate against archetype analysis
- Criterion: are the same ecological community states consistently recovered across L?
- Output isolation: `results/{model}_L{n_hidden}/` per run

### Bias absorber problem

**STATUS**: OPEN — mitigation not yet applied

At L=5, h1 is always-on (sat_hi > 99%). Its weights W[:,1] function as a second bias vector, not a community state. Candidate fixes: increase L (sweep may resolve naturally), initialise b[j] with large negative values to force hidden units off, or add per-unit entropy regularisation. Defer until sweep results are available.

---

## PENDING — Unresolved questions

- January-February 2023 anomaly: bloom event or instrument artifact? Needs taxon-level zoom
- Chronological split fraction: currently 85/15 — confirm with professor
- ε for log-transform: moot for current paths (no log-transform applied); revisit if Gaussian visible is reconsidered
- Val NLL plateau (NB, L=5): train NLL 0.585→0.435, val NLL 0.647→0.564 (gap widens after epoch ~100). Diagnosed as temporal distribution shift (val = 2024, train = 2019–2023), not overfitting. Monitor across L values in sweep.
