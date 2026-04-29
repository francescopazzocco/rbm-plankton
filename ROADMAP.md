# ROADMAP.md

> Impact-driven roadmap. Living document — update as priorities shift.
> Decisions needed to unblock work live at the bottom.
> Rationale for past choices lives in DECISION_LOG.md.

---

## Now

| Task | Success criterion |
|---|---|
| L-sweep [3, 5, 7, 10] — BB-RBM and NBB-RBM | All runs complete, results in `results/{model}_L{n}/` |
| Cross-validate recovered states against archetype analysis | Same community states appear consistently across L values |

---

## Next

| Task | Depends on |
|---|---|
| Choose final n_hidden | L-sweep results |
| Characterise bias absorber across L (is it always the same unit? does it disappear at higher L?) | L-sweep results |
| Build analysis script: seasonal patterns, taxa groupings per hidden unit, comparison across models | Final n_hidden |
| NaN test set evaluation: clamp non-missing taxa, reconstruct, score on non-missing | Final n_hidden |

---

## Later

| Task | Notes |
|---|---|
| Gaussian hidden units | Only if L-sweep h(t) activations show structured continuous gradients rather than binary switching |
| January–February 2023 anomaly investigation | Taxon-level zoom; decide whether to exclude or retain |
| Interpret val NLL plateau (NBB-RBM) | Temporal distribution shift (train=2019–2023, val=2024) vs model limitation |

---

## Decisions needed

| Decision | Blocking | What is needed to close it |
|---|---|---|
| Hidden unit type (Bernoulli vs Gaussian) | Gaussian hidden unit path | L-sweep h(t) activation shape analysis |
| n_hidden final value | Analysis + interpretation | L-sweep cross-validation vs archetype |
| Train/val split fraction (85/15) | Nothing currently | Professor confirmation |
| January–February 2023: bloom or artifact? | Potential data exclusion | Taxon-level time series inspection |
