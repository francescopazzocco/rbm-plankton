# ROADMAP.md

> Impact-driven roadmap. Living document — update as priorities shift.
> Decisions needed to unblock work live at the bottom.
> Rationale for past choices lives in DECISION_LOG.md.

---

## Now

| Task | Success criterion |
|---|---|
| Multi-seed training N=10 via `main_multiseed.py` | All (family, L, seed) runs complete in `results/multiseed/` |
| Update `sweep_analysis.py` for mean ± std reporting | Improvement table shows mean ± std; L selection criterion applied |
| Hidden activation analysis: weight profiles + state timelines | Scripts run and figures interpretable for all families and L values |

---

## Next

| Task | Depends on |
|---|---|
| Choose final n_hidden (statistically grounded) | Multi-seed results + hidden activation analysis |
| Cross-model comparison: do NB-RBM and BB-RBM agree on community structure? | `hidden_cross_model.py` (not yet written) |
| NaN test set evaluation: clamp non-missing taxa, reconstruct, score on non-missing | Final n_hidden |

---

## Later

| Task | Notes |
|---|---|
| Gaussian hidden units | Only if h(t) activations show structured continuous gradients — current evidence (sat_mid <15% across all runs) suggests Bernoulli is sufficient |
| January–February 2023 anomaly investigation | Taxon-level zoom; decide whether to exclude or retain |
| Interpret val NLL plateau (NBB-RBM) | Temporal distribution shift (train=2019–2023, val=2024) vs model limitation |

---

## Decisions needed

| Decision | Blocking | What is needed to close it |
|---|---|---|
| n_hidden final value | All downstream analysis | Multi-seed mean ± std improvement table + hidden activation agreement across L |
| Hidden unit type (Bernoulli vs Gaussian) | Gaussian hidden unit path | h(t) activation shape — current Bernoulli evidence strong but not closed |
| Train/val split fraction (85/15) | Nothing currently | Professor confirmation |
| January–February 2023: bloom or artifact? | Potential data exclusion | Taxon-level time series inspection |

---

## Closed

| Item | Resolution |
|---|---|
| L-sweep [3,4,5,6,7,10] — BB-RBM and NBB-RBM | Complete. nb_L10 diverged (LOG-012); excluded from NB analysis. |
| Bias absorber at L=5 (h1 always-on) | Was a first-run training artifact. All hidden units active across all current runs. |
| NLL/PLL plateau qualitative confirmation | Confirmed by `sweep_analysis.py` — diminishing returns beyond L=5–7. |
