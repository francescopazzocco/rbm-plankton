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

**Rationale:** Zero threshold produces constant units for rare taxa: a taxon absent 90% of the time is almost always 0 after binarisation, giving the corresponding visible unit no gradient signal. Median threshold ensures each taxon is active approximately 50% of the time by construction, maximising information content per unit.

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

**Rationale:** The data split is chronological and deterministic — every seed sees the identical train/val partition. The only variance across seeds is weight initialisation and batch/CD sampling order, which is exactly what should be measured. The 5070 Ti supports 10 parallel training processes simultaneously (models are small: 83×L weights). Results stored in `results/multiseed/{family}_L{n}/seed_{k}/`.

**Consequences:** L selection becomes statistically grounded. `sweep_analysis.py` to be extended to read multiseed results and report mean ± std improvement per L step.

---

## LOG-014 · Results directory partitioned by analysis stage

**Context:** `results/data_analysis/` was accumulating outputs from both the initial data investigation and the L-sweep analysis — logically distinct stages mixed in one directory.

**Decision:** Partition results output into three directories: `results/data_analysis/` (initial data investigation, untouched), `results/sweep/` (sweep_analysis.py outputs), `results/hidden/` (hidden activation analysis scripts).

**Rationale:** Mixed output makes it hard to identify which figures belong to which analysis stage and clutters the working directory. Separate directories make each stage independently reproducible and navigable.

**Consequences:** `sweep_analysis.py` updated to write to `results/sweep/`. New hidden analysis scripts write to `results/hidden/`. Existing `results/data_analysis/` content unchanged.

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

**Consequences:** Multiseed sweep rerun under `results/multiseed_pcd/` for NB L∈{3,4,5,6,7}. Divergence rate at L≥5 expected to drop substantially. If divergence persists, next step is FPCD (add fast weight tensor with high LR + decay).

---

## LOG-017 · n_hidden = 6 selected for all model families

**Context:** PCD multiseed sweep (N=10 seeds, all families, L∈{3,4,5,6,7}) complete. All 150 runs converged.

**Decision:** n_hidden = 6 for NB-RBM, bernoulli_median, and bernoulli_zero.

**Rationale:** Two complementary lines of evidence both point to L=6:

1. *Step-wise criterion (LOG-013)*: improvement from L→L+1 must exceed within-L std. For NB-RBM the 5→6 step (Δ=0.0124, 1.6σ) is significant; the 6→7 step (Δ=0.0019, 0.2σ) is noise. For bernoulli_median no single step clearly exceeds its std, but the pattern is identical.

2. *Cumulative view (confirmed by sweep_analysis.py)*: L=3→6 yields ~2.5% NLL/PLL reduction for both NB-RBM and bernoulli_median — well beyond any within-L std. L=7 adds nothing cumulatively. The step-wise criterion is conservative; the cumulative picture provides the stronger argument and both converge to L=6.

bernoulli_zero is flat across all L (total range 0.0044 ≈ 3σ of L=3): no signal in either direction. L=6 chosen by consistency; the flatness confirms that zero-threshold binarization produces a near-trivial problem (sparse vectors → model predicts near-constant zeros), validating the median threshold decision in LOG-005.

**Consequences:** Canonical models for downstream analysis: NB-RBM L=6 seed_8 (val_nll=0.5437, global minimum), bernoulli_median L=6 best seed. Hidden activation analysis and cross-model community state comparison proceed at L=6.
