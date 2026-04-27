# CHANGELOG.md

> Pure chronological record of what happened and what was found.
> No decisions here — those live in DECISIONS.md.

---

## 2026-04-18

**Initial project setup**

- Dataset inspected: 2113 rows × 84 cols, daily resolution, 2019-03-21 to 2024-12-31
- 47 all-zero rows identified
- NaN present in all 83 taxa columns (60–207 NaNs per column)
- Sparsity: 25.7% of all values are zero
- Value range: [0.0, 0.444] organisms/μL
- Row sums NOT normalised to 1 in the CSV — this was a preprocessing step
  in the upstream archetype analysis, not a property of the raw data
- Base code `rbm_train_export.py` reviewed: Bernoulli-Bernoulli RBM for MNIST/FPGA.
  CD-k engine, RMSprop, PLL monitor reusable. FPGA export, MNIST loader, binary
  visible unit math to be replaced.
- Architecture decided: Gaussian-Bernoulli RBM, n_visible=83, n_hidden=5
- `preprocess.py` written with switchable NAN_STRATEGY ("zero" / "drop")
- `project_log.md` created (later superseded by this three-file structure)

---

## 2026-04-19 – 2026-04-25

**Row sum analysis**

- Plotted row sum time series with 30-day rolling median
- Found: strong annual seasonality, CV=1.29 (std > mean) → row sum variability
  is dominated by biology (bloom cycles), not instrument drift
- Student-proposed threshold-based QC rejected: biological variation is
  non-negligible and inseparable from instrument noise via a global threshold
- Lomb-Scargle periodogram run on row sum (used over FFT because of gaps
  in the time series after dropping NaN rows)
- Dominant period confirmed: ~365 days
- 1740-day peak identified as artifact of 2022-2023 extreme events, not a
  real multi-year cycle
- Annual medians computed: no monotonic trend 2019-2024; mean inflated
  by extreme events in 2022-2023
- January-February 2023 anomaly found: monthly median ~0.43 vs ~0.005-0.017
  in same months of all other years — flagged as unresolved

**NaN structure investigation (prompted by missing October 2022 in seasonal plot)**

- October 2022: all 31 days have exactly 31 taxa NaN, same 31 taxa every day
- These 31 taxa each have exactly 90 NaNs across the full dataset
  (or 207 for a subset), concentrated in ~3 month-long blocks
- Interpretation: ML classifier retraining/update events — during transition
  periods some taxonomic groups had no predictions
- Consequence: "drop any NaN row" silently removes entire months of otherwise
  valid data. "Replace with 0" is ecologically wrong (unclassified ≠ absent)
- Decision: drop NaN rows for training; use them as structured test set post-training

**Row normalisation rejected for RBM**

- Originally proposed to match archetype analysis preprocessing
- Rejected: Gaussian visible units need z-score for scale, not row normalisation;
  row normalisation introduces a sum=1 constraint the RBM cannot model;
  total biomass signal is ecologically real and should not be discarded

**All-zero rows: justification for dropping verified**

- Zero rows occur in consecutive multi-day runs (5, 7, 11 days)
- Cluster in winter months and at documented instrument gap boundaries
- No ecological mechanism produces all 83 taxa absent for 11 consecutive days
- Confirmed: instrument downtime artifacts

**Marginal distribution analysis**

- Plotted raw and z-scored distributions for 4 representative taxa
  (common, intermediate, rare, dominant)
- Finding: all taxa are zero-inflated and heavily right-skewed (skew 3–11)
  regardless of abundance level
- z-score alone does not produce Gaussian-like distributions
- Decision: log-transform v ← log(v + ε) before z-score
- This makes the Gaussian visible unit assumption substantially more defensible

**Architecture discussion**

- Gaussian-Bernoulli vs alternatives examined
- Gaussian-Gaussian raised: continuous hidden units more natural for smooth
  seasonal oscillations; log-transform even more important for training stability
  in this case (both layers unbounded)
- Beta visible raised: natural for compositional data in (0,1); requires
  row-normalisation (reconsidered in this context); zero-inflation is harder
  to handle (outside Beta support)
- Key insight: visible unit type (data likelihood) and hidden unit type
  (latent structure) are orthogonal choices — 4 combinations possible
- Discrete vs continuous hidden units: binary hidden forces sharp community
  state transitions; seasonal data suggests smooth transitions → Gaussian
  hidden may be more appropriate; test post-training by examining h(t) spread

**Inter-group coordination proposal**

- Two groups working on project → proposed split:
  Group A (this group): Gaussian visible + log-transform, test Bernoulli and Gaussian hidden
  Group B: Beta visible + row-normalisation, test same hidden unit split
- Scientific value: two groups encode different assumptions (absolute abundance
  vs relative composition) — divergence between results would itself be informative
- Flag for Group B: Beta requires (0,1) strictly; zero-inflation needs explicit
  handling before starting

---

## 2026-04-26

- project_log.md replaced by three-file structure:
  DECISIONS.md / CHANGELOG.md / ARCHITECTURE.md
- preprocess.py updated: row normalisation removed, NaN strategy set to "drop"
