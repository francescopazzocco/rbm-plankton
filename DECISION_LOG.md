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
