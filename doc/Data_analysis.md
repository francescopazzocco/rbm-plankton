# Data Analysis Report — Lake Greifen Plankton Dataset

**Dataset**: `data/raw/TimeSeries_countsuL_clean.csv`  
**Output figures**: `results/data_analysis/`  
**Reference**: Eyring et al. (2025), *Scientific Data* 12:653  
**Analysis date**: 2026-04-26

---

## 1. Dataset Overview

The raw CSV contains 2113 rows × 84 columns: one `date` column and 83 taxa columns
reporting daily plankton abundance in organisms/μL. Each row is one calendar day,
spanning 2019-03-21 to 2024-12-31 with no missing dates.

| Property                     | Value                     |
| ---------------------------- | ------------------------- |
| Total rows                   | 2113                      |
| Taxa                         | 83                        |
| Date range                   | 2019-03-21 → 2024-12-31   |
| All-zero rows                | 47                        |
| Rows with any NaN            | 207                       |
| Sparsity (fraction of zeros) | 25.7%                     |
| Value range                  | [0.0, 0.444] organisms/μL |

After removing all-zero rows (47) and NaN rows (160), **1906 rows** remain for training.

---

## 2. Row Sum Analysis — Is Total Abundance a Reliable Signal?

The **row sum** (sum of all 83 taxa per day) is a proxy for total community abundance.
The sensor images a fixed water volume, so in principle the row sum should be
approximately constant across valid days. In practice we observe strong variability.

![Row sum time series](https://claude.ai/chat/fig1_rowsum_timeseries.png)

**Key findings:**

The coefficient of variation is **CV = 1.29** (std > mean), indicating that row sum
variability is dominated by biology — bloom events produce order-of-magnitude spikes
over background winter values. A global threshold cannot separate instrument drift from
genuine biological variation; the seasonal signal is too strong.

A proposal to flag days where row sum falls anomalously low *given the season* was
considered but rejected at this stage: characterising what "expected for the season"
means requires a seasonal baseline, which requires first understanding the dominant
temporal structure (see Section 3).

---

## 3. Temporal Structure — Lomb-Scargle Periodogram

Standard FFT requires uniformly sampled data. After dropping NaN rows the time series
has structured gaps, so **Lomb-Scargle** was used instead — it fits sinusoids directly
and handles gaps correctly.

![Lomb-Scargle periodogram](https://claude.ai/chat/fig2_lombscargle.png)

**Key findings:**

The dominant period is **~365 days** (annual cycle), confirmed by harmonics at ~245d
and ~482d. The strong peak at **~1740 days** (nearly the full dataset length) is a red
flag: a peak at the total span typically indicates a long-term trend or extreme events
masquerading as a periodic signal, not a real multi-year cycle.

---

## 4. Is There a Multi-Year Trend?

To test whether the 1740-day peak reflects a real trend, annual medians and the
seasonal shape per year were examined.

![Annual trend and seasonal shape](https://claude.ai/chat/fig3_annual_seasonal.png)

**Key findings:**

The **annual median is stable** across 2019-2024 — no monotonic trend. The annual
mean is inflated in 2022-2023 by extreme events, not by a directional shift.
The 1740-day Lomb-Scargle peak is therefore an artifact of those extreme events,
not a real multi-year cycle.

A clear anomaly is visible in **January-February 2023**: monthly median row sum
is ~0.43 organisms/μL, roughly two orders of magnitude above the same months in
all other years (~0.005-0.017). This is flagged as unresolved — it could be a
genuine bloom event or an instrument artifact.

---

## 5. NaN Structure — Not Random, Not Instrument Drift

Visual inspection of the seasonal plot (Fig. 3) revealed an entire month missing
from one year. Investigation showed October 2022 has all 31 days with exactly the
same 31 taxa NaN — the other 52 taxa are valid.

![NaN structure](https://claude.ai/chat/fig5_nan_structure.png)

**Key findings:**

NaNs are **structured block outages**, not random scatter. The same 31 taxa are
missing for entire month-long periods, concentrated in ~3 blocks across the
5-year dataset. This pattern is consistent with ML classifier retraining events:
during model update transitions, some taxonomic groups produce no predictions.

This has direct consequences for the NaN handling strategy:

| Strategy                       | Problem                                    |
| ------------------------------ | ------------------------------------------ |
| Replace NaN with 0             | Ecologically wrong — unclassified ≠ absent |
| Replace NaN with seasonal mean | Introduces imputation assumptions          |
| Drop NaN rows for training     | Clean, no assumptions — 1906 rows remain   |

**Decision**: drop NaN rows for training. Use NaN rows as a **structured test set** post-training: clamp non-missing taxa as visible input, reconstruct the missing ones,
score reconstruction quality on the non-missing taxa as a proxy for imputation accuracy.

---

## 6. Marginal Distributions — Why Gaussian Alone Is Insufficient

Four representative taxa were selected spanning the full abundance range and examined
at three stages: raw counts, log-transformed, and log + z-scored.

![Marginal distributions](https://claude.ai/chat/fig4_distributions.png)

**Key findings:**

Every taxon — regardless of abundance level — shows the same pathological shape:
a large spike at zero and a heavy right tail from bloom events.

| Taxon       | Type         | Frac. zeros | Skewness (raw) | Skewness (log+z) |
| ----------- | ------------ | ----------- | -------------- | ---------------- |
| aulacoseira | dominant     | 3%          | 4.5            | ~0.5–1.5         |
| cryptophyte | common       | 0%          | 4.8            | ~0.5–1.5         |
| rotifer     | intermediate | 0%          | 4.2            | ~0.5–1.5         |
| snowella    | rare         | 93%         | 6.5            | ~1–2             |

A plain z-score does not make these distributions Gaussian — the zero-spike and
right tail persist. **Log-transform v ← log(v + ε)** substantially reduces skewness
and brings the distributions much closer to Gaussian before z-scoring. The residual
deviation (particularly for heavily zero-inflated taxa like snowella) is acknowledged
but is substantially better than untransformed data.

---

## 7. Conclusions and Architecture Implications

### Preprocessing pipeline (confirmed)

```
raw CSV
  → drop all-zero rows    (instrument downtime: consecutive multi-day runs)
  → drop NaN rows         (ML classifier outage blocks)
  → log-transform: v ← log(v + ε)
  → per-taxon z-score     (fit on train split only)
  → chronological train/val split (no random shuffling)
```

Row normalisation (sum rows to 1) is **explicitly rejected** for the RBM:
it introduces a deterministic simplex constraint the model cannot represent,
discards the total-biomass signal shown to be ecologically real, and provides
no technical benefit once z-scoring is applied.

### Visible unit type

The distributions support **Gaussian visible units with log-transform** as the
primary architecture for this group. The log-transform makes the Gaussian
assumption defensible across the majority of taxa.

An alternative path — **Beta visible units with row-normalisation** — is being
explored by a parallel group. Beta is the natural distribution for compositional
data in (0,1) and directly models relative abundance. The two approaches encode
different scientific assumptions:

|                   | Gaussian + log-transform        | Beta + row-normalisation             |
| ----------------- | ------------------------------- | ------------------------------------ |
| Models            | Absolute abundance dynamics     | Relative community composition       |
| Preserves         | Total biomass signal            | Compositional structure              |
| Zero handling     | log(0+ε) maps to large negative | Requires zero-inflated Beta or shift |
| Row normalisation | Not required                    | Required                             |

### Hidden unit type

Both groups will test **Bernoulli hidden** (discrete community states) and **Gaussian hidden** (continuous latent trajectory). The choice is orthogonal to
the visible unit type. The seasonal time series suggests smooth transitions,
which favours continuous hidden units — but this will be verified empirically
by examining h(t) spread after training.

### Open questions before training

- Value of ε for log-transform (global minimum non-zero, or fixed constant?)
- Whether to z-score after log-transform or use raw log values
- Train/val split fraction (currently 85/15 — to confirm with professor)
- January-February 2023 anomaly: bloom or artifact?
- n_hidden sweep: start at 5 (archetype reference), then 3, 7, 10

---

*For the complete decision history see `DECISIONS.md` and `CHANGELOG.md`.*
