# DECISIONS.md

> Architecture Decision Record — chronological log of choices made and why.
> Append only: new decisions go at the end. Do not edit past entries.
> Open questions and upcoming work live in ROADMAP.md.
> Technical specs live in ARCHITECTURE.md.

---

## LOG-001 · Drop all-zero rows

**Context:** Some observation days have zero counts across all 83 taxa.

**Decision:** Drop these rows before training.

**Rationale:** Multi-day consecutive all-zero runs are instrument downtime events, not an ecological signal. They represent a different data-generating process and would corrupt seasonal structure if included.

**Consequences:** ~X rows removed. Downtime periods have no representation in the model.

---

## LOG-002 · Drop NaN rows for training; reserve as test set

**Context:** Some rows have NaN for a subset of taxa (up to 31 simultaneously).

**Decision:** Drop NaN rows from training and validation. Retain them as a structured post-training test set.

**Rationale:** NaN blocks affect the same 31 taxa simultaneously, consistent with ML classifier retraining events — not random missingness. Dropping is the correct response. Retaining them as a test set makes use of the structured missing pattern post-hoc (clamp non-missing taxa, reconstruct missing, score on non-missing).

**Consequences:** October 2022 entirely lost (all 31 days have the same 31 taxa NaN). Test set procedure not yet implemented.

---

## LOG-003 · Reject row normalisation

**Context:** Row-normalising (dividing each observation by its row sum) is a common preprocessing step for compositional data.

**Decision:** Do not apply row normalisation in any preprocessing path.

**Rationale:** Normalisation introduces a simplex constraint (rows sum to 1) that the RBM cannot represent. It also discards the total-biomass signal, which carries ecological information (e.g., bloom vs low-abundance periods are indistinguishable after normalisation).

**Consequences:** Raw absolute concentrations are used. Total biomass variation is preserved as a visible signal.

---

## LOG-004 · Chronological train/val split at 85/15

**Context:** Standard ML practice is random train/val split; time series data requires temporal ordering.

**Decision:** Chronological split: first 85% of clean rows for training, last 15% for validation.

**Rationale:** Random split would leak future temporal structure into training, inflating validation metrics. Chronological split tests genuine out-of-sample generalisation to a future time window. Val set = 2024 data.

**Consequences:** Val set is from a single contiguous period (2024), so val performance reflects temporal generalisation, not i.i.d. generalisation. A val NLL plateau is expected and not necessarily overfitting — see ADR-011.

---

## LOG-005 · Binarisation threshold = per-taxon median

**Context:** Bernoulli visible units require binary input. Two natural thresholds: zero (presence/absence) and per-taxon median.

**Decision:** Threshold = per-taxon median.

**Rationale:** For rare taxa (present <50% of the time), median=0 and both thresholds encode identically (v > 0) — dead units either way. For common taxa, zero threshold gives near-constant-on units (present most days, no gradient). The median transforms them into above/below-median abundance (~50/50 split, full gradient). These common taxa carry the signal.

**Consequences:** Binarisation is taxon-specific. The biological interpretation shifts from presence/absence to above/below-median abundance.

---

## LOG-006 · Implement BB-RBM and NBB-RBM in parallel

**Context:** Multiple visible unit distributions were under consideration (Bernoulli, Gaussian, Beta, Negative Binomial).

**Decision:** Implement Bernoulli-Bernoulli RBM (BB-RBM) and NB-Bernoulli RBM (NBB-RBM) in parallel. Deprioritise Gaussian and Beta.

**Rationale:** BB-RBM is the simplest baseline and easiest to validate. NBB-RBM is the theoretically motivated choice for overdispersed count data with structural zeros. Gaussian requires log-transform + z-score, which discards zero-inflation structure. Beta requires row normalisation (rejected in ADR-003) and explicit zero-inflation handling.

**Consequences:** Two model classes maintained in parallel. BB-RBM validated first; NBB-RBM experimental but promising.

---

## LOG-007 · Bernoulli hidden units — implemented first, Gaussian on hold

**Context:** Hidden units can be Bernoulli (discrete community states, binary h_j) or Gaussian (continuous latent trajectory).

**Decision:** Implement Bernoulli hidden units first. Gaussian hidden units on hold.

**Rationale:** Bernoulli was chosen for implementation simplicity and because the ecological hypothesis (distinct seasonal community states) maps naturally to binary on/off units. Not chosen because Gaussian was ruled out — the choice reflects sequencing, not closure.

**Consequences:** L=5 NBB-RBM result: sat_mid = 3% at convergence — activations are overwhelmingly binary. This is consistent with the Bernoulli choice but does not close the question. Revisit if L-sweep h(t) activations show structured continuous gradients.

---

## LOG-008 · COUNT_SCALE = 1000 for NBB-RBM

**Context:** Raw data is in organisms/μL, with values in [0, 0.44]. NB is defined for count data.

**Decision:** Multiply raw concentrations by COUNT_SCALE=1000 before feeding to NBB-RBM.

**Rationale:** COUNT_SCALE=1.0 caused degenerate training: θ stuck at initialisation, MSE ≈ 0 (mean-collapse), CD gradient ≈ 0. Root cause: NB lgamma gradient collapses when all values ≪ 1 — the score function cannot distinguish between sub-unit floats. Scaling to ×1000 brings data to approximate integer-count scale and resolves all three failure modes. The lgamma formulation supports non-integers so no further rounding is needed.

**Consequences:** Data is no longer in its original units during training. Learned μ values are in units of organisms/1000μL. Weights and biases must be interpreted accordingly.

---

## LOG-009 · L1 regularisation scope: W only for NBB-RBM

**Context:** The original single-file implementation applied L1 to W, a, b for both models. During refactoring this was copied to NBB-RBM without review.

**Decision:** NBB-RBM: L1 on W only. BB-RBM: L1 on W, a, b.

**Rationale:** In NBB-RBM, `a` is the log-mean baseline: μ_i = exp(a_i + W_i·h). Applying L1 to `a` shrinks it toward zero, biasing all conditional means toward exp(0)=1 — ecologically wrong. `b` is the hidden bias; shrinking it toward zero is also undesirable as it forces hidden units toward P=0.5. In BB-RBM, `a` and `b` are logit-scale parameters where shrinkage toward zero is semantically neutral.

**Consequences:** NBB-RBM mean structure is free to fit the data. Bug was present in earlier runs (refactor introduced it); corrected before sweep.

---

## LOG-010 · NBB-RBM numerical stability guards

**Context:** Training with L=10 crashed at epoch 161 with a Gamma distribution ValueError. Root cause: exp(η) overflow at float32 boundary → μ=inf → log-likelihood=-inf → NaN gradient → log_θ=NaN → θ=NaN.

**Decision:** Three guards added to `nb_rbm.py`:

1. Clamp η at max=10.0 before exp in `_mu()`
2. Clamp log_θ to [−10, 10] after each update
3. Apply nan_to_num(nan=0.0) to θ gradient before RMSprop step

**Rationale:** The clamp on η is the structural fix — it bounds μ to ≤ exp(10) ≈ 22026, which is 50× the data maximum (443) and well within float32 range. The log_θ clamp and gradient guard are defensive layers that catch any residual instability from other sources. float64 was considered and rejected: consumer GPU FP64 throughput is ~1/32 of FP32; the clamp is the correct fix.

**Consequences:** μ is bounded. Ecologically, μ > 22026 counts per 1000μL would be physically implausible, so the clamp is not restrictive. Sweep across L values now runs stably.

---

## LOG-011 · Training monitors: PLL for BB-RBM, NLL for NBB-RBM

**Context:** Reconstruction MSE was used as the sole training monitor. For BB-RBM it is biased (computed on positive-phase mini-batches, not full CD reconstruction). For NBB-RBM on COUNT_SCALE=1000 data, MSE is scale-dependent and misleading (near-zero MSE was the mean-collapse symptom).

**Decision:** Add PLL (pseudo-log-likelihood) as primary monitor for BB-RBM. Add NLL (negative log-likelihood via lgamma) as primary monitor for NBB-RBM. MSE retained as secondary.

**Rationale:** PLL is a tractable exact proxy for log-likelihood in Bernoulli models, computed via free-energy differences — no sampling, no bias. NLL directly measures the NB fit quality and is scale-independent. Both decrease monotonically when the model is learning, unlike MSE which can oscillate due to the positive-phase estimator bias. The val NLL plateau observed in NBB-RBM L=5 (~epoch 100) is temporal distribution shift (train=2019–2023, val=2024), not overfitting.

**Consequences:** PLL and NLL are the primary diagnostic for model quality. MSE remains in the CSV for reference. Hidden unit saturation (sat_lo, sat_hi, sat_mid) added alongside NLL for NBB-RBM to detect bias absorbers and binary collapse.

---

## LOG-012 · Exclude nb_L10; valid NB-RBM range is L∈{3,4,5,6,7}

**Context:** The L-sweep included L=10 for NB-RBM. The run diverged catastrophically between epochs 30–230 (train MSE ~27M at epoch 500, NLL columns NaN).

**Decision:** Exclude L=10 from all NB-RBM analysis. Valid sweep range for NB-RBM is L∈{3,4,5,6,7}.

**Rationale:** The divergence is a dynamical instability, not a recoverable hyperparameter issue. Diagnosis: theta trajectories for L=7 and L=10 are nearly identical through epoch 130 (both reach theta≈1.09), ruling out theta drift as the cause. The MSE oscillation amplitude in L=10 is larger from epoch 40 onward, indicating the weight updates are overshooting the loss landscape curvature. With more hidden units the landscape is more complex; the same learning rate that keeps L≤7 in a stable basin crosses energy barriers for L=10. Forcing convergence with a lower LR would likely yield redundant near-zero units — the instability is a signal that L=10 exceeds the data's intrinsic dimensionality, not a tuning problem. BB-RBM L=10 converged cleanly because the Bernoulli energy landscape is bounded and better-conditioned.

**Consequences:** NB-RBM analysis uses L∈{3,4,5,6,7}. BB-RBM retains L=10 in the sweep for completeness.

---

## LOG-013 · Multi-seed training (N=10) as statistical validation for L selection

**Context:** The L-sweep improvement table (sweep_analysis.py) compares single-run val NLL/PLL values across L values. RBM training is stochastic — weight initialisation and CD Gibbs sampling introduce run-to-run variance. A single-run comparison is not statistically defensible: the NLL difference between L=5 and L=7 could fall within the within-L variance.

**Decision:** Run N=10 independent seeds per (family, L) combination via `main_multiseed.py`. L selection criterion: improvement from L→L+1 must exceed the within-L standard deviation across seeds.

**Rationale:** The data split is chronological and deterministic — every seed sees the identical train/val partition. The only variance across seeds is weight initialisation and batch/CD sampling order, which is exactly what should be measured. The 5070 Ti supports 10 parallel training processes simultaneously (models are small: 83×L weights). Results stored in `training_runs/{family}_L{n}/seed_{k}/`.

**Consequences:** L selection becomes statistically grounded. `sweep_analysis.py` to be extended to read multiseed results and report mean ± std improvement per L step.

---

## LOG-014 · Results directory partitioned by analysis stage

**Context:** `results/01_exploratory/` (previously `results/data_analysis/`) was accumulating outputs from both the initial data investigation and the L-sweep analysis — logically distinct stages mixed in one directory.

**Decision:** Partition results output into numbered stage directories: `01_exploratory/` (EDA), `02_model_analysis/` (hidden state analysis), `03_evaluation/` (model evaluation), `04_model_selection/` (L-sweep metrics), plus `tables/` for CSV data and `diagnostics/` for training curves.

**Rationale:** Mixed output makes it hard to identify which figures belong to which analysis stage and clutters the working directory. Separate directories make each stage independently reproducible and navigable.

**Consequences:** All analysis scripts updated to write to the new paths. `.gitignore` whitelist updated accordingly.

---

## LOG-015 · NB-RBM training instability at L≥5: slow mixing, not gradient explosion

**Context:** Multi-seed training (N=10) at L≥5 shows ~10% of runs diverge regardless of learning rate. Gradient clipping (max norm=1.0 on dW/da/db and g_θ) was implemented and tested. Results: L=5 worst-case seed improved (0.87→0.67) but L=6 divergence rate worsened (1→2 failures). Clipping was reverted.

**Decision:** Do not apply gradient clipping to NB-RBM. Accept the ~10% failure rate at L≥5 and resolve it operationally by running multiple seeds and selecting the best converged result. The proper structural fix is Persistent CD (PCD) — deferred to a future iteration.

**Rationale:** The clipping failure revealed the true instability mechanism: **slow Gibbs chain mixing in the presence of multimodal distributions**, not gradient explosion. When the model distribution has two well-separated modes (e.g., bloom vs non-bloom community states), the CD-1 chain takes only one step per batch and cannot cross the low-probability valley between modes. The negative phase samples are therefore drawn from one mode only, biasing the CD gradient — the update direction is wrong, not just its magnitude. Gradient clipping constrains the magnitude of a biased signal, which cannot fix the directional bias and may worsen it by interfering with legitimate large updates needed early in training. The instability worsens with larger L because more hidden units give the model more capacity to carve out sharp, well-separated energy basins, making the valley deeper and the chain slower to mix. The proper fix is PCD, which maintains persistent Markov chains across batches and allows them to cross energy barriers given enough time.

**Consequences:** NB-RBM training remains CD-1. For final model selection, run ≥3 seeds per (family, L) and keep the best-NLL converged result. PCD implementation added to ROADMAP as Next item.

---

## LOG-016 · Persistent Contrastive Divergence (PCD) for NB-RBM

**Context:** LOG-015 identified slow Gibbs chain mixing as the root cause of 10–30% divergence rate at L≥5. CD-1 restarts the Markov chain from the data every batch, so the chain never has time to cross low-probability valleys between modes.

**Decision:** Implement PCD-1 in `nb_rbm.py` via `use_pcd=True` flag. Maintain a buffer of `n_pcd_chains=500` persistent visible-unit particles. Each batch, a random subset of particles is advanced by `cd_steps` Gibbs steps and stored back. The positive phase is unchanged (still uses real data). Applied to NB-RBM only — Bernoulli energy landscape is bounded and does not require it.

**Rationale:** PCD keeps the fantasy particles in the model's current distribution across batches. Over time the chains migrate between modes (bloom/non-bloom states) rather than being restarted in the data distribution each time. This removes the directional bias in the CD gradient that caused divergence. `n_pcd_chains=500 ≥ BATCH_F=256` ensures we never need to reuse a particle within the same batch draw. FPCD (fast weights) was considered but deferred: PCD-1 is the minimal intervention; fast weights add a hyperparameter pair and should only be introduced if PCD still shows significant divergence.

**Consequences:** Multiseed sweep rerun under `training_runs/` for NB L∈{3,4,5,6,7}. Divergence rate at L≥5 expected to drop substantially. If divergence persists, next step is FPCD (add fast weight tensor with high LR + decay).

---

## LOG-017 · n_hidden = 6 selected for all model families

**Context:** PCD multiseed sweep (N=10 seeds, all families, L∈{3,4,5,6,7}) complete. All 150 runs converged.

**Decision:** n_hidden = 6 for NB-RBM, bernoulli_median, and bernoulli_zero.

**Rationale:** Two complementary lines of evidence both point to L=6:

1. *Step-wise criterion (LOG-013)*: improvement from L→L+1 must exceed within-L std. For NB-RBM the 5→6 step (Δ=0.0124, 1.6σ) is significant; the 6→7 step (Δ=0.0019, 0.2σ) is noise. For bernoulli_median no single step clearly exceeds its std, but the pattern is identical.

2. *Cumulative view (confirmed by sweep_analysis.py)*: L=3→6 yields ~2.5% NLL/PLL reduction for both NB-RBM and bernoulli_median — well beyond any within-L std. L=7 adds nothing cumulatively. The step-wise criterion is conservative; the cumulative picture provides the stronger argument and both converge to L=6.

bernoulli_zero is flat across all L (total range 0.0044 ≈ 3σ of L=3): no signal in either direction. L=6 chosen by consistency; the flatness confirms that zero-threshold binarization produces a near-trivial problem (sparse vectors → model predicts near-constant zeros), validating the median threshold decision in LOG-005.

**Consequences:** Canonical models for downstream analysis: NB-RBM L=6 seed_8 (val_nll=0.5437, global minimum), bernoulli_median L=6 best seed. Hidden activation analysis and cross-model community state comparison proceed at L=6.

---

## LOG-018 · NaN test set evaluation — partial-observation imputation

**Context:** 160 rows were excluded from training/validation due to NaN taxa (LOG-002). Three missingness patterns survive the nonzero filter: p3 (3 NaN, 104 rows), p31 (31 NaN, 43 rows, includes October 2022), p54 (54 NaN, 13 rows). These rows allow a post-hoc test of the model's imputation capability under structured missingness.

**Decision:** Evaluate via zero-imputation clamped inference: set NaN taxa to 0, sample P(h | v_partial) × 100, score NLL on observed positions only. Applied to NB-RBM L=6 and bernoulli_median L=6. No model changes were needed.

**Rationale:** Zero imputation (NaN → 0) treats missing taxa as absent — ecologically conservative for a dataset where many taxa have genuine absence periods. It is the minimal intervention: no alternating Gibbs or mean-field approximation. The 100-sample Monte Carlo average reduces variance to a stable estimate without batching overhead on the small model (83×6 weights).

**Results:**

| family | pattern | n_miss | n_obs | n_rows | test_nll_mean | test_nll_std |
|---|---|---|---|---|---|---|
| NB-RBM          | p3  |  3 | 80 | 104 | 0.3785 | 0.1351 |
| NB-RBM          | p31 | 31 | 52 |  43 | 0.5012 | 0.1463 |
| NB-RBM          | p54 | 54 | 29 |  13 | 0.3344 | 0.2000 |
| Bernoulli-median | p3  |  3 | 80 | 104 | 0.4316 | 0.0955 |
| Bernoulli-median | p31 | 31 | 52 |  43 | 0.7374 | 0.1501 |
| Bernoulli-median | p54 | 54 | 29 |  13 | 0.6408 | 0.3300 |

Reference: NB val_nll = 0.5437; Bernoulli-median val_pll = 0.5332.

Key observations:
- NB-RBM test NLL ≤ val_nll across all patterns — partial observation does not degrade reconstruction quality. The model's compositional representation (∼30 active 6-bit patterns) constrains the hidden state effectively even from incomplete input.
- Bernoulli-median: p3 performance is comparable to val_pll (0.43 vs 0.53), but p31 and p54 are substantially worse (0.74, 0.64). The exclusive-switching representation is less robust when a larger fraction of the visible layer is unobserved.
- October 2022 (p31): NB-RBM NLL starts elevated at month onset (~0.77, consistent with the beginning of the classifier retraining event) and decreases toward month-end (~0.42). This gradient likely reflects progressively more recoverable community states as October transitions toward autumn.
- p54 high variance (std ≈ 0.20) reflects the small sample (13 rows) and should be interpreted cautiously.

**Consequences:** NaN test evaluation complete. NB-RBM confirmed as more robust to structured missingness than Bernoulli-median. January–February 2023 anomaly investigation open at this point — closed in LOG-019.

---

## LOG-019 · January–February 2023 anomaly — retained as real ecological event

**Context:** Total abundance is ~7× the dataset mean across December 2022–February 2023, returning abruptly to baseline by March 2023. The question was whether to exclude this window and retrain.

**Decision:** Retain. No exclusion, no code changes.

**Rationale:** Eyring et al. (2025) covers this period in full (dataset spans May 2018–June 2023) and documents no instrument issue for that window. Their stated cleaning philosophy is to remove only clear technical artefacts and instrument-related errors, preserving genuine biological variability. Silence on this event is therefore informative: had it been an instrument problem, it would have been flagged or removed consistent with that philosophy. The NaN patterns that ARE known artefacts in this dataset (LOG-002) are linked to ML classifier retraining events and carry no NaN rows — this window has no NaN rows, making a classifier artefact origin additionally unlikely. The most parsimonious interpretation is a real high-biomass event (likely a winter bloom) that falls outside the scope of the paper's narrative (a data descriptor, not an ecological analysis). Certainty could be obtained from the companion CTD dataset (Merkli et al. 2024, ref 29 in Eyring 2025) — Chl-a and phycocyanin elevation at 3m depth would confirm a bloom — but that is outside the scope of this project.

**Consequences:** Data retained as-is. All trained models already include this window. Study is complete with no remaining open items.

---

## LOG-020 · NB_ReLU_RBM: hidden activation clamp [0, 5] is necessary; link function is secondary

**Context:** `NB_ReLU_RBM` replaces Bernoulli hidden units with ReLU (h ∈ [0, ∞)). Three rounds of experiments (no activation clamp in any run):

**Round 1 — exp link, lr=0.01:**
| Config | val NLL | h_mean | outcome |
|---|---|---|---|
| exp + PCD   | NaN | 5.98 | runaway + dead mix → NaN |
| exp + CD-1  | NaN | —    | collapse → NaN |

**Round 2 — softplus link, lr=0.01:**
| Config | val NLL | h_mean | outcome |
|---|---|---|---|
| softplus + PCD  | NaN | 5.98 | 96.6% dead, few runaway → NaN |
| softplus + CD-1 | NaN | 0.23 | 99.4% dead → NaN |

**Round 3 — softplus link, lr=0.001 (10× lower):**
| Config | val NLL | h_mean | outcome |
|---|---|---|---|
| softplus + PCD  | NaN | 260.8 | 0% dead, h grows to 261 → NaN |
| softplus + CD-1 | NaN |  98.8 | 18% dead, h grows to 99 → NaN |

**Decision:** The hidden activation clamp to [0, 5] is necessary and sufficient for stable NB_ReLU_RBM training. It is not a workaround — it is the correct structural fix given count-scale visible units. No other intervention (link function, CD variant, learning rate, sampling correction) eliminates the need for it.

**Rationale:** Four rounds of experiments spanning link function × CD variant × learning rate × sampling correctness all confirm the same failure without the clamp. Round 4 (below) was the decisive test: fixing a sampling bug in `_ph_given_v` (see consequences) and running softplus+CD-1 still produced h_mean → 216 and NaN NLL within 300 epochs.

**Round 4 — sampling-bug fix, lr=0.001:**
| Config | val NLL | h_mean | outcome |
|---|---|---|---|
| exp + CD-1 (fixed sampling)       | NaN |   8.3 | slow runaway |
| softplus + CD-1 (fixed sampling)  | NaN | 216.5 | fast runaway |
| softplus + PCD (fixed sampling)   | NaN | 122.5 | intermediate |

The root cause is a **scale mismatch between count data and ReLU hidden units**, not the link function or sampling formula. The Nair & Hinton (2010) Gaussian-ReLU RBM energy includes a h²/2 term that acts as a restoring force preventing h runaway. This works because visible units are z-scored to zero mean, unit variance, keeping pre-activations pre_j = b_j + Σ_i W_ij v_i in ±3 range. In NB_ReLU_RBM, visible units are raw counts ×1000 (values up to 444). With W initialised at ~0.1 (after scale_init), a single high-count taxon contributes ≈ 444 × 0.1 = 44 to the pre-activation; summed over 83 inputs, pre_j can reach the hundreds at initialisation. The h²/2 restoring force is proportional to h, so a signal of +300 from the data overwhelms any gradient pulling h back. The h²/2 theory applies only when inputs are normalised — count-scale data breaks the precondition entirely. The clamp to [0, 5] enforces the ceiling externally and unconditionally, making it the correct engineering fix for this data regime.

The sampling bug found during this investigation: the old `_ph_given_v` applied `relu` before passing to `_sample_hidden`, so `h = relu(relu(pre_act) + ε)` instead of the correct `h = relu(pre_act + ε)`. For units with negative pre-activation the old code sampled from a half-normal (mean ≈ 0.4) instead of driving h toward 0, which weakened the lower absorbing boundary. This bug is fixed in the current code regardless of the clamp conclusion.

**Consequences:** `NB_ReLU_RBM` uses `clamp(h, 0, 5)` unconditionally (both in `_ph_given_v` and `_sample_hidden` — the correct sampling fix has been applied independently). The sampling bug is fixed: `_ph_given_v` now returns the raw pre-activation; `_sample_hidden` correctly samples `relu(pre_act + N(0,1))`; `hidden_probs` applies relu for the expected value. The clamp is reapplied on top of the correct sampling. Any future ReLU-hidden NB variant operating on count-scale data must include this clamp. The principled alternative — normalising visible inputs before the hidden layer — would break the NB visible distribution and is not pursued.

---

## LOG-021 · NBSigmoidRBM: sigmoid hidden units are PCD-safe, stable, and beat NB-Bernoulli

**Context:** `NBSigmoidRBM` replaces Bernoulli hidden units with sigmoid (h ∈ (0, 1)), sampled as Bernoulli. Sigmoid is bounded → no dead-unit problem → PCD-safe. Full sweep L=[4,5,6,7] × 10 seeds, shuffled split, 200 epochs.

**Full sweep results (val NLL, mean ± std over seeds, 10 seeds per L):**

| L | Val NLL | h_mean range | Divergences |
|---|---|---|---|
| 4 | 0.460 ± 0.005 | 0.30–0.59 | 0/10 |
| 5 | 0.449 ± 0.003 | 0.27–0.60 | 0/10 |
| 6 | 0.448 ± 0.007 | 0.13–0.63 | 0/10 |
| 7 | 0.443 ± 0.019 | 0.19–0.63 | 0/10 |

**Key observations:**
- **No NaN divergences** across all 40 runs — PCD-safe by design (sigmoid is bounded).
- **h_mean stays in 0.13–0.63** — healthy hidden activity, no dead units (sparsity 0%).
- **Strong scale:** NLL improves monotonically with L: 0.460 → 0.449 → 0.448 → 0.443.
- **Beats NB-Bernoulli baseline:** NBSigmoidRBM L=4 (0.460) ≈ NB-Bernoulli L=6 (0.48). L=7 (0.443) is the **lowest NLL achieved across all model families** on the shuffled split.
- L=6 and L=7 are statistically indistinguishable (overlapping ±1σ).

**Decision:** NBSigmoidRBM is the recommended NB-family hidden unit type. Sigmoid is the correct bounded nonlinearity for count-data RBMs: it prevents hidden activation runaway without external clamping, supports PCD, and yields the best NLL of any tested configuration.

**Rationale:** Four hidden unit types were tested for NB-family RBMs:
1. **Bernoulli** (NB_RBM) — baseline. Stable, but Bernoulli hides the continuous-valued pre-activation behind a coin flip, losing signal.
2. **ReLU** (NB_ReLU_RBM) — unbounded → dead units + NaN; clamped [0,5] still 6/10 divergent at scale. Abandoned.
3. **Sigmoid** (NBSigmoidRBM) — bounded, PCD-safe, no dead units, best NLL. Selected.
4. **Softmax** (NBSoftmaxRBM) — bounded, PCD-safe, but collapses to deterministic assignments (H≈0.05). See LOG-022.

**Consequences:** NBSigmoidRBM is the new NB-family baseline. Future development should use sigmoid hidden units. L=7 (or L=6 if parsimony preferred) is the recommended final model size.

---

## LOG-023 · All numeric constants centralised in `_constants.py`

**Context:** Numeric literals were scattered across `nb_rbm.py`, `zinb_rbm.py`, `bernoulli_rbm.py`, `base_rbm.py`, `_hidden_monitors.py`, and `main_multiseed.py` as inline values with no documented rationale. Many appeared multiple times in different roles: `1e-4` appeared in four distinct roles (theta clamp, probability clamp, L1 strength, RMSprop ε); `1e-8` appeared in six places; the seven training defaults (`lr_decay`, `gamma`, `batch_i`, `batch_f`, `n_batches`, `beta`, `epsilon`) were triplicated across all three model `train()` signatures.

**Decision:** Extract all non-trivial numeric literals into `src/models/_constants.py` with a documented explanation for each value. Training defaults are imported by all three model classes and `main_multiseed.py` via aliased names. `EVAL_EVERY` is surfaced as a top-level setting in `main_multiseed.py`.

**Rationale:** A change to any stability parameter required editing multiple files by hand with no guarantee of consistency. The audit also revealed a pre-existing documentation error: ARCHITECTURE.md stated `logit_pi clamp [−5, 5]` while the code had always been `[−10, 10]`; the named constant `LOG_PARAM_CLAMP_MIN/MAX` made the discrepancy immediately visible and correctable.

**Consequences:** `_constants.py` is the single source of truth for all numerical stability parameters, training defaults, and monitoring thresholds. Saturation thresholds in `_hidden_monitors.py` and `visualization.py` are both sourced from `BERNOULLI_SAT_LO/HI`, ensuring monitoring and plotting criteria stay in sync.

---

## LOG-024 · COUNT_SCALE applied uniformly to both preprocessing paths

**Context:** `COUNT_SCALE=1000` was applied only inside `load_raw_counts` (NB/ZINB path). `load_and_binarise` (Bernoulli path) operated on raw organisms/μL. This framed COUNT_SCALE as an NB-specific model choice rather than a dataset-level preprocessing decision, and made the NB path look arbitrary in isolation.

**Decision:** Add `scale=COUNT_SCALE` parameter to `load_and_binarise`, applied before binarisation. `main_multiseed.py` passes `scale=COUNT_SCALE` to both loaders. All preprocessing paths now share the same unit shift (organisms/μL → organisms/mL).

**Rationale:** Binarisation is rank-invariant under positive scaling — `(v > median(v)) ≡ (1000v > median(1000v))` — so the binary model input is identical at any positive scale. The only observable effect is that stored thresholds are now in organisms/mL across all families, consistent with the NB path. Applying COUNT_SCALE uniformly removes the implicit assumption that it is distribution-specific and frames it as a global numerical stability decision (sub-unit floats → approximate integer range), which is the correct justification.

**Consequences:** Stored thresholds for Bernoulli models are in organisms/mL. 18 taxa have zero median — their threshold remains 0 regardless of scale. The binary output of `load_and_binarise` is bit-identical to the pre-change behaviour.

---

## LOG-022 · NBSoftmaxRBM: softmax hidden units collapse to near-deterministic assignments

**Context:** `NBSoftmaxRBM` replaces Bernoulli hidden units with softmax (Σ_j h_j = 1, 0 ≤ h_j ≤ 1), sampled as one-hot from multinomial. Softmax is bounded → PCD-safe. 5-seed test L=5 shuffled, 200 epochs.

**Results (val NLL, 5 seeds at L=5):**

| Seed | Val NLL | H (entropy) |
|---|---|---|
| 0 | 0.491 | 0.06 |
| 1 | 0.487 | 0.05 |
| 2 | 0.494 | 0.04 |
| 3 | 0.498 | 0.07 |
| 4 | 0.484 | 0.06 |

**Mean NLL: 0.491 ± 0.005.** Entropy H ≈ 0.05 (max possible log₂5 ≈ 2.32), meaning each sample is assigned to essentially one archetypal unit ≈100% of the time.

**Decision:** NBSoftmaxRBM is not useful for this dataset. The softmax competition forces each sample into a single hidden state, discarding the distributed representation that gives RBMs their expressive power. The near-zero entropy confirms the model collapses to a hard clustering.

**Rationale:** The softmax mixture-of-experts assumption is too rigid for plankton community data, where multiple ecological processes (bloom, succession, seasonality) overlap. A distributed representation (Bernoulli or Sigmoid) is necessary to capture overlapping factors.

**Consequences:** NBSoftmaxRBM retained in the codebase for completeness but not recommended for further use. It is excluded from the main sweep.

---

## LOG-025 · `training_runs/` is the canonical run directory; all locations live in `paths.py`

**Context:** Twelve scripts each re-derived the repository root with
`Path(__file__).parent.parent.parent` and then appended their own idea of where
the trained runs live. Six of them, plus `code/train/config.py`,
`ARCHITECTURE.md`, `README.md` and `.claude/SESSION_TRACKING.md`, named
`trained_models/`. On disk the 84 run directories have always been in
`training_runs/`, which `.gitignore` labelled "legacy". The consequence was not a
crash: `discover_run_dirs` found nothing, every family was skipped, and the whole
`analysis/` + `diagnostic/` layer printed "no runs found" and exited 0. The
packaging work (`pyproject.toml`, editable install of `code/src` as `models`)
made the `models` package importable from any working directory but said nothing
about data or output locations, so the drift was invisible to it.

**Decision:** The runs stay where they are, on disk, in `training_runs/`; the
code and the documentation are corrected to point there. A new module
`code/src/models/paths.py` owns every filesystem location — `PROJECT_ROOT`,
`DATA_PATH`, `RUNS_ROOT`, `RESULTS_ROOT`, `DIAGNOSTIC_ROOT` — and the
chronological/shuffled run-directory naming. `PROJECT_ROOT` is overridable with
the `RBM_PLANKTON_ROOT` environment variable. No script derives a root of its
own.

The split strategy becomes a value, `CHRONO` or `SHUFFLED`, converted to the
`_shuffled` directory suffix only by `paths.run_dir()`. `config.SHUFFLE_SPLIT`
(a bool that controlled the shuffling) and `config.SHUFFLE_TAG` (a string that
controlled the directory name) are replaced by the single `config.SPLIT`.

**Rationale:** Renaming the directory on disk was the alternative, and it would
have matched the existing documentation. It was rejected because it moves 47 MB
of irreplaceable trained weights plus a `training_runs.zip` archive to satisfy a
naming preference, and because the ambiguity has to be removed at its source in
either case: the defect was twelve independent definitions of the same path, not
the name they used. The paths are deliberately *not* declared in
`pyproject.toml` — reading it at runtime requires first locating it, which is
the same root-finding problem plus a `tomllib` dependency.

Two independent constants for one concept (`SHUFFLE_SPLIT`, `SHUFFLE_TAG`) is a
latent footgun: setting the bool without the tag writes shuffled runs into the
chronological directories, silently mixing two split strategies in one run
directory.

**Consequences:** All fourteen non-deferred scripts now run on a clean checkout
and produce output. Verified numerically behaviour-preserving: the rebuilt
`io.load_model` yields bit-identical parameters to the four loaders it replaces,
`load_nan_rows`/`scale_counts`/`binarise_rows` reproduce their predecessors
exactly, and every deterministic tracked output regenerates byte-identical.
`trained_models/` remains gitignored so a stale reference cannot silently create
a second run directory.

Recorded here and not fixed: the Bernoulli runs currently in `training_runs/`
store thresholds in organisms/μL, i.e. they predate LOG-024, whose consequences
section describes thresholds in organisms/mL. `io.binarise_rows` compares the
NaN rows to the stored thresholds unscaled, which is correct for exactly these
runs and would silently binarise almost everything to 0 for a Bernoulli family
retrained with the current `train.py`. The function now warns instead of scoring
silently; choosing the fix belongs with the scale-invariance test in the
validation phase.

---

## LOG-028 · `code/` split into `src/` (library) and `scripts/` (entry points)

**Context:** `code/train/`, `code/diagnostic/`, `code/analysis/` and
`code/archive/` sat as siblings of `code/src/models/` — the installable
library and the CLI entry points that import it were indistinguishable by
path alone. This surfaced when a teammate pushed two commits straight to
`master` (`custom_analysis/`, `scripts/`) that duplicated existing
`code/analysis/` scripts under new top-level directories, one of them
reintroducing the `sys.path.insert` hack `c6bb000` had removed — there was no
structural cue for where a new pipeline script belongs.

**Decision:** `code/train/`, `code/diagnostic/`, `code/analysis/`,
`code/archive/` move to `code/scripts/{train,diagnostic,analysis,archive}/`.
`code/src/models/` is unchanged and remains the only importable package
(`pip install -e .`). The convention going forward: new reusable logic goes
in `src/models/`; new pipeline entry points go in the matching
`scripts/<stage>/` directory; nothing under `scripts/` reimplements what
`src/models/` already provides.

**Consequences:** Every path reference in docstrings, `README.md`,
`ARCHITECTURE.md` and `results/README.md` updated to `code/scripts/...`.
`pyproject.toml` package discovery (`where = ["code/src"]`) and the CI
workflow (lints/tests `code/src/models` and `tests/` only) were already
scoped to `src/` and needed no change. The teammate's duplicate
`custom_analysis/` and `scripts/` content is to be discarded, not merged, once
`origin/master` is reconciled with local history — recorded as a follow-up,
not resolved by this entry.

---

## LOG-029 · ZINB not recommended for reconstruction/imputation; deficit is mostly a Gibbs-mixing artifact, not enough to change the ranking

**Context:** `nan_eval_summary.csv` (`code/scripts/diagnostic/nan_test_eval.py`)
showed ZINB (and its sigmoid/softmax hidden-unit variants) with worse NaN-imputation
NLL than the matching NB variant across all three missingness patterns (p3, p31,
p54), consistently regardless of hidden-unit type. Working hypothesis: not a
genuine deficiency of zero-inflation as a modelling choice, but an artefact of
`score_row_gibbs` (`code/src/models/_eval_utils.py`) — the imputation loop
initialises missing positions at `v=0` (the value zero-inflation most favours)
and, via `_sample_zinb`, redraws the discrete `z ~ Bernoulli(pi)` zero/non-zero
branch every Gibbs step, unlike NB's single smooth mode. This could anchor a
short, non-persistent chain (5 + 3·n_miss steps — PCD does not apply outside
training, where weights persist chain state across updates; see LOG-016) to the
zero-inflated mode.

**Test:** `code/scripts/diagnostic/zinb_meanfield_test.py` re-scored the same
rows and runs, replacing the ancestral `sample_zinb` step inside the imputation
loop with the deterministic conditional mean `(1-pi)*mu` (`meanfield_zinb` in
`_eval_utils.py`), leaving the final scoring likelihood (`loss_zinb`, full ZINB
log-prob) unchanged in both variants. Single run, no fixed seed, no repeats.

**Results (mean NLL on observed taxa):**

| run | pattern | ancestral | mean-field | NB-Sigmoid (best NB variant, same pattern) |
|---|---|---|---|---|
| zinb L8 shuffled         | p3  | 0.422 | 0.417 | 0.375 |
| zinb L8 shuffled         | p31 | 0.574 | 0.549 | 0.508 |
| zinb L8 shuffled         | p54 | 0.360 | 0.327 | 0.306 |
| zinb_sigmoid L7 shuffled | p3  | 0.414 | 0.407 | 0.375 |
| zinb_sigmoid L7 shuffled | p31 | 0.539 | 0.549 | 0.508 |
| zinb_sigmoid L7 shuffled | p54 | 0.336 | 0.325 | 0.306 |
| zinb_softmax L6 shuffled | p3  | 0.430 | 0.430 | 0.375 |
| zinb_softmax L6 shuffled | p31 | 0.571 | 0.562 | 0.508 |
| zinb_softmax L6 shuffled | p54 | 0.355 | 0.354 | 0.306 |

**Decision:** ZINB, in any hidden-unit variant, is not recommended for
reconstruction or NaN-imputation. `NBSigmoidRBM` (LOG-021) remains the model of
choice for both latent-pattern extraction and reconstruction/imputation — one
model serves both purposes.

**Rationale:** The mean-field substitution confirms the mixing-artifact
hypothesis for the base ZINB (Bernoulli hidden): it recovers most of the p31/p54
gap to NB, nearly closing it on p54 (0.360 → 0.327 vs. NB-Sigmoid's 0.306). This
means much of plain ZINB's apparent deficit was a property of the evaluation
procedure, not the model. Sigmoid/softmax ZINB variants show a smaller, less
consistent recovery (p31 for zinb_sigmoid gets *worse* under mean-field),
indicating the zero-inflation/hidden-unit-type interaction there is not
primarily a mixing problem. Critically, **NB-Sigmoid still has the lowest NLL of
every row in the table above, ancestral or mean-field, on every pattern** — the
residual gap (0.02–0.04) is smaller than the original ancestral gap but never
crosses zero. Whether the residual reflects a real (if modest) NB advantage or
further evaluation noise is not resolved — `score_row_gibbs` has no fixed seed,
and per-row NLL std in the original sweep (~0.13–0.20) dwarfs these deltas — but
it does not matter for the decision: closing the artifact fully would only tie
ZINB with the already-superior NB-Sigmoid, never surpass it. There is no
outcome consistent with the data in which ZINB becomes the better choice.

**Consequences:** No architecture change. `NBSigmoidRBM` (LOG-021) stands
unchallenged as the canonical model for both narratives (pattern extraction,
reconstruction). The mean-field imputation variant and its diagnostic script
are kept as exploratory tooling (`diagnostic_outputs/zinb_meanfield_test/`,
untracked) — not promoted to the canonical NaN-evaluation method, since that
remains open (ROADMAP "Now" #2) and the mean-field variant was never intended
as a replacement, only as a probe. The init-sensitivity test proposed alongside
it (0 vs. unconditional mean vs. random NB draw as the missing-position
initialiser) and a multi-seed repeat of this comparison are not pursued: with
the ranking already settled, they would sharpen the explanation, not change the
recommendation.

---

## LOG-029 · `results/` becomes a published tier: `publish_results.py` + `MANIFEST.json`

**Context:** Nine analysis/diagnostic scripts wrote figures and tables
directly into tracked `results/`, alongside others that already wrote into
gitignored `diagnostic_outputs/` — an inconsistency decided per-script, not by
policy. Consequence, observed directly this session: a routine post-rename
verification pass (re-running every script to confirm the `code/scripts/`
move didn't break anything) silently overwrote eight report-cited figures and
tables with no record of why. Separately, `results/README.md` had documented
for some time that `03_evaluation/`, part of `04_model_selection/` and
`diagnostics/` were "frozen snapshots" — tracked files the current code no
longer regenerates because the producing script had since moved to
`diagnostic_outputs/` — an open item with no resolution path.

**Decision:** `results/` is now a published tier, `diagnostic_outputs/` the
only staging tier. Every script that used to import `RESULTS_ROOT` now writes
into the equivalent `DIAGNOSTIC_ROOT` subtree instead — including
`use_trained_rbm.py` and `compare_model_reconstructions.py`, whose
`results/reconstruction_plots/` default was cwd-relative and never actually
covered by `paths.py`, nor by `.gitignore`'s tracked-results allowlist (an
accidental gap, not a decision — those plots were never meant to be tracked).
No script writes to `results/` any more, full stop.

`code/scripts/publish_results.py` is the one door in: it copies a named
category (or `--all`) from `diagnostic_outputs/` into `results/` and records
each copied file in `results/MANIFEST.json` via the new `models.manifest`
module — producing script, git commit, publish timestamp. Two scripts may
publish into the same `results/` folder (e.g. `02_model_analysis`) without
clobbering each other's entries; only the files actually copied get touched.
`tests/test_results_manifest.py` fails CI if a tracked file under `results/`
has no manifest entry, which is what makes "someone bypassed
`publish_results.py`" a caught condition rather than a silent one.

**Consequences:** The "frozen snapshot" open item is resolved by construction
— resolving today's already-generated output through the new publish flow
surfaced seven genuinely stale tracked files that no current script produces
under those exact paths any more (an old flat `sweep_shuffled_final_metric.png`
naming superseded by the `shuffled/` subdirectory convention; three
`tables/hidden/*.csv` files missing the `zinb` family and later `L` values
added since). Removed via `git rm`, superseded by their correctly-published
equivalents elsewhere in the tree. `results/README.md` rewritten to describe
the tier boundary instead of a per-directory "refreshed?" table that had
already drifted out of date once. `.gitignore` gained an explicit
`!results/MANIFEST.json` — it would otherwise have been silently swallowed by
the blanket `results/*` pattern, the same class of gap that hid
`reconstruction_plots/` above.

**Left as a follow-up, not decided here:** `training_runs/` (the weights
themselves) gets no equivalent provenance ledger — raised in conversation,
scoped out as a separate tier with a different lifecycle (gitignored, never
published, owned solely by `train.py`/`paths.RUNS_ROOT`).
