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
| NaN rows post-training  | Use as structured test set    | Clamp non-missing taxa, reconstruct missing, score on non-missing                                   |
| Visible layer transform | Log-transform: v ← log(v + ε) | All taxa are zero-inflated and heavily right-skewed (skew 3–11); z-score alone insufficient         |

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

## OPEN — Architecture

### Visible unit type

**STATUS**: Pending professor + inter-group coordination

Two paths under consideration, proposed as a group split:

- **Group A**: Gaussian visible + log-transform
- **Group B**: Beta visible + row-normalisation

Key asymmetry: Beta requires data in (0,1) strictly — zero-inflation needs explicit
handling (zero-inflated Beta or additive constant). Group B should be made aware before starting.

### Hidden unit type

**STATUS**: Open — test both within each visible unit choice

- **Bernoulli hidden**: discrete community states, binary h_j
- **Gaussian hidden**: continuous latent trajectory, more natural for smooth seasonal oscillations

Decision criterion (post-training): if h(t) activations cluster near 0/1 → discrete
architecture was appropriate. If spread continuously → Gaussian hidden units warranted.

### n_hidden

**STATUS**: Open

- Start with 5 (archetype analysis reference point)
- 5 is a lower bound on community states, not a prescription for RBM
- Seasonal oscillations alone may require more units
- Plan: train with 5, then sweep 3 / 7 / 10 and compare reconstruction MSE

### σ² in Gaussian visible units

**STATUS**: Open

- Fixed at 1 (standard; works if log-transform + z-score is well-calibrated)
- Learned per unit (more flexible but unstable)
- Start fixed, revisit if reconstruction MSE plateaus

---

## PENDING — Unresolved questions

- January-February 2023 anomaly: bloom event or instrument artifact? Needs taxon-level zoom
- ε for log-transform: use minimum non-zero value across dataset, or fixed constant? Affects how zeros are handled
- Chronological split fraction: currently 85/15 — confirm with professor
- Whether to z-score after log-transform or use raw log values

---

## NOT YET STARTED

- RBM implementation (Gaussian-Bernoulli, adapted from rbm_train_export.py)
- Training script
- Interpretation / analysis script
- GitHub repository setup
