# ROADMAP.md

> Impact-driven roadmap. Living document — update as priorities shift.
> Decisions needed to unblock work live at the bottom.
> Rationale for past choices lives in DECISION_LOG.md.

---

## Now

| Task | Success criterion |
|---|---|
| NaN test set evaluation | Clamp non-missing taxa → reconstruct → score NLL on non-missing positions only (October 2022 structured missingness block) |

---

## Next

| Task | Depends on |
|---|---|
| January–February 2023 anomaly investigation | Taxon-level zoom; decide whether to exclude or retain |

---

## Future work

| Task | Notes |
|---|---|
| Gaussian hidden units | sat_mid < 15% across all runs — Bernoulli assumption well supported. Revisit only if a follow-up dataset shows structured continuous gradients in h(t). |
| Interpret val NLL plateau (NBB-RBM) | Temporal distribution shift (train=2019–2023, val=2024) vs model limitation |

---

## Decisions needed

| Decision | Blocking | What is needed to close it |
|---|---|---|
| Hidden unit type (Bernoulli vs Gaussian) | Gaussian path | Closed as future work — Bernoulli confirmed sufficient |
| Train/val split fraction (85/15) | Nothing currently | Professor confirmation |
| January–February 2023: bloom or artifact? | Potential data exclusion | Taxon-level time series inspection |

---

## Closed

| Item | Resolution |
|---|---|
| L-sweep [3,4,5,6,7,10] — BB-RBM and NBB-RBM | Complete. nb_L10 diverged (LOG-012); excluded from NB analysis. |
| Bias absorber at L=5 (h1 always-on) | Was a first-run training artifact. All hidden units active across all current runs. |
| NLL/PLL plateau qualitative confirmation | Confirmed by `sweep_analysis.py` — diminishing returns beyond L=5–7. |
| NB-RBM slow mixing / divergence at L≥5 | Fixed by PCD-1 (LOG-016). 10/10 convergence across all L after PCD. |
| n_hidden final value | L=6 for all families (LOG-017). Substantial cumulative gain L=3→6; no gain L=6→7. |
| Multi-seed training N=10 | Complete in `results/multiseed_pcd/` for all 3 families × L∈{3,4,5,6,7}. |
| Implement PCD for NB-RBM | Done (LOG-016). `use_pcd=True, n_pcd_chains=500` in `NBRBM.train()`. |
| Cross-model comparison NB vs BB-median L=6 | Done. Both models independently recover summer/winter community axes. NB uses compositional representation (~30 patterns/64, consistent across seeds). BB uses exclusive switching. Core structure agreed. |
