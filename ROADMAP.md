# ROADMAP.md

> Impact-driven roadmap. Living document — update as priorities shift.
> Decisions needed to unblock work live at the bottom.
> Rationale for past choices lives in DECISION_LOG.md.

---

## Complete model architecture

```
BaseRBM                         base_rbm.py
├── BernoulliRBM                bernoulli_rbm.py
│
├── NB branch                   nb_rbm.py
│   └── NB_RBM(BernoulliHiddenMonitor, BaseRBM)
│       ├── NB_ReLU_RBM(ReLUHiddenMonitor, NB_RBM)    ❌ abandoned
│       ├── NBSigmoidRBM(SigmoidHiddenMonitor, NB_RBM) ✓ recommended
│       └── NBSoftmaxRBM(SoftmaxHiddenMonitor, NB_RBM) ❌ low entropy
│
└── ZINB branch                 zinb_rbm.py
    └── ZINB_RBM(BernoulliHiddenMonitor, BaseRBM)
        └── ZINB_ReLU_RBM(ReLUHiddenMonitor, ZINB_RBM)
```

**Recommendation:** Use `NBSigmoidRBM` for all NB-family work. Sigmoid hidden units are PCD-safe, produce the lowest NLL (0.443 ± 0.019 at L=7), and show healthy hidden activity (h_mean 0.13–0.63). L=6–7 are statistically indistinguishable.

---

## Now

1. **Validation phase (§B of `.claude/REORG_AND_VALIDATION.md`)** — tests that check
   mathematics rather than shapes, then fix the defects that survived the
   reorganisation. Highest-value first: `BernoulliRBM.pll` against brute force
   (it drives Bernoulli model selection and has no test), NB/ZINB log-prob
   against `scipy.stats.nbinom`, the clamping invariant of `score_row_gibbs`.
2. **Settle the canonical NaN-evaluation method** — LOG-018's numbers came from a
   method no longer in the tree; `nan_test_eval` samples, `split_comparison`
   injects means. Closes with a new ADR, never an edit to LOG-018.

---

## Next

1. **Read and repair `compare_model_reconstructions.py` and the four archetype
   scripts** — deliberately untouched in the reorganisation phase pending review
   by their authors. Dispositions proposed in §A-3 of the reorganisation report.
2. **Decide where the diagnostic figures belong** — tracked `results/` (cited by
   the report) or untracked `diagnostic_outputs/` (regeneratable). Today several
   tracked figures are frozen snapshots that a re-run does not refresh; see
   `results/README.md`.
3. **Bernoulli threshold units** — runs on disk store thresholds in organisms/μL
   and predate LOG-024; a retrained Bernoulli family would break
   `io.binarise_rows` silently. Guarded by a warning, not fixed.

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
| January–February 2023: bloom or artifact? | ~~Potential data exclusion~~ | Closed — retained as real ecological event (LOG-019) |

---

## Closed

| Item | Resolution |
|---|---|
| NBSigmoidRBM / NBSoftmaxRBM shuffled sweeps | Complete and written up (LOG-021, LOG-022). |
| Codebase reorganisation (§A of the reorganisation report) | Complete (LOG-025). One run root, one loader, one preprocessing path; every non-deferred script runs. |
| L-sweep [3,4,5,6,7,10] — BB-RBM and NBB-RBM | Complete. nb_L10 diverged (LOG-012); excluded from NB analysis. |
| Bias absorber at L=5 (h1 always-on) | Was a first-run training artifact. All hidden units active across all current runs. |
| NLL/PLL plateau qualitative confirmation | Confirmed by `sweep_analysis.py` — diminishing returns beyond L=5–7. |
| NB-RBM slow mixing / divergence at L≥5 | Fixed by PCD-1 (LOG-016). 10/10 convergence across all L after PCD. |
| n_hidden final value | L=6 for all families (LOG-017). Substantial cumulative gain L=3→6; no gain L=6→7. |
| Multi-seed training N=10 | Complete in `training_runs/` for all families × L∈{3,4,5,6,7} + sigmoid/softmax sweeps. |
| Implement PCD for NB-RBM | Done (LOG-016). `use_pcd=True, n_pcd_chains=500` in `NB_RBM.train()`. |
| Cross-model comparison NB vs BB-median L=6 | Done. Both models independently recover summer/winter community axes. NB uses compositional representation (~30 patterns/64, consistent across seeds). BB uses exclusive switching. Core structure agreed. |
| NaN test set evaluation | Done (LOG-018). 160 rows, 3 missingness patterns. NB-RBM test_nll ≤ val_nll across all patterns; more robust than Bernoulli-median. October 2022 NLL decreases over the month. |
| January–February 2023 anomaly | Closed (LOG-019). Total abundance ~7× mean Dec 2022–Feb 2023. Eyring 2025 covers the period, documents no instrument issue, and states their philosophy is to preserve genuine biological variability. Retained as probable real ecological event; no exclusion. |
| NB_ReLU_RBM viability | Abandoned (LOG-020). Clamp [0,5] required but 6/10 seeds still divergent at scale. ReLU is fundamentally mismatched with count-scale visible units. |
| Hidden monitoring mixins | Complete. `BernoulliHiddenMonitor`, `ReLUHiddenMonitor`, `SigmoidHiddenMonitor`, `SoftmaxHiddenMonitor` in `_hidden_monitors.py`. |
| ZINB underperformance on NaN-imputation: mixing artifact or genuine? | Closed (LOG-029). Mean-field imputation (`zinb_meanfield_test.py`) confirms mixing artifact explains most of plain ZINB's gap to NB, but NB-Sigmoid still has the lowest NLL on every pattern regardless. ZINB not recommended for reconstruction in any hidden-unit variant; `NBSigmoidRBM` unchanged as canonical model. |
