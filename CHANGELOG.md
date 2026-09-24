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

---

## 2026-07-26 (later)

**Reorganisation phase A — one home for the runs, one loader, one preprocessing path**

Executed from `.claude/REORG_AND_VALIDATION.md` §A. The validation phase (§B) is
a separate later session; nothing mathematical was changed here.

- Nothing in `analysis/` or `diagnostic/` had been running. Twelve scripts each
  re-derived the repo root; six of them plus `config.py` and the docs pointed at
  `trained_models/`, while the 84 run directories are in `training_runs/`.
  `discover_run_dirs` returned nothing and every family was silently skipped —
  exit code 0, no output, no error. Resolved as LOG-025.
- New `code/src/models/paths.py` owns `PROJECT_ROOT`, `DATA_PATH`, `RUNS_ROOT`,
  `RESULTS_ROOT`, `DIAGNOSTIC_ROOT` and the run-directory naming.
  `RBM_PLANKTON_ROOT` overrides the root.
- Chronological/shuffled promoted from a string suffix to a `CHRONO`/`SHUFFLED`
  value. `config.SHUFFLE_SPLIT` + `config.SHUFFLE_TAG` collapsed into
  `config.SPLIT`; `sweep_analysis`, `hidden_mean_activation`,
  `hidden_dominant_state`, `hidden_cross_model`, `hidden_pattern_analysis` and
  `rbm_hidden_stackplot` gained `--split` instead of a module constant that had
  to be edited. Shuffled results write to a `shuffled/` subdirectory, so the two
  splits no longer overwrite each other's figures.
- Deduplication: four "weights → model" loaders became `io.load_model` (reads
  `npz["visible_model"]`, verified present in all 840 run directories); three
  copies of the NaN-row preparation became `io.load_nan_rows` + `scale_counts` +
  `binarise_rows`; the raw-CSV filter became `io.partition_rows`, now also used
  by `dataset_analysis.load_clean`; five readers of
  `rbm_hidden_activations.csv` collapsed onto `io.load_hidden_activations`; the
  0.5-threshold pattern logic became `visualization.hidden_binary` +
  `pattern_frequency`, shared by `hidden_cross_model` and
  `hidden_pattern_analysis`.
- `hidden_coactivation.py` → `hidden_dominant_state.py`: it computes weight
  profiles and argmax state assignment, no coactivation quantity.
- Dead default paths fixed: `hidden_pattern_analysis` and `rbm_hidden_stackplot`
  defaulted to `weights/NB_RBM/L6_chrono` and `analysis/results/…`, none of which
  exist; `plot_visible_by_hidden` defaulted to `analysis/results/visible_by_hidden`.
  All three now resolve a run through `run_dir` + `best_seed_dir`.
- `sys.path.insert` hacks removed from the three `code/archive/` scripts (missed
  by commit c6bb000, which cleared them from the tracked pipeline scripts).
- `nan_test_eval` labels every bar `{family} L={n} ({split})`. Previously the
  `zinb*` labels omitted "shuffled" while mixing L=6, L=7 and L=8 with both
  splits in one chart, so a capacity or split difference read as a family
  difference. Also added the missing empty-input guard before `pd.concat`.
- Verified: 14/14 pytest pass, `ruff check code/src/models tests/` clean, and
  every non-deferred script was executed against the real 84-run tree and
  produced output. `io.load_model` returns bit-identical parameters to the four
  loaders it replaced; the row-preparation helpers reproduce their predecessors
  exactly; deterministic tracked outputs regenerate byte-identical.

**Found during the work, not in the audit report:**

- Bernoulli thresholds stored in `training_runs/` are in organisms/μL, so those
  runs predate LOG-024 (which describes organisms/mL). `binarise_rows` compares
  unscaled rows to them, correct for these runs, silently wrong for any
  Bernoulli family retrained with today's `train.py`. Now warns; fix deferred to
  the validation phase.
- Several tracked files under `results/` are frozen snapshots: the diagnostic
  scripts write their regenerated versions into the gitignored
  `diagnostic_outputs/`, so re-running the pipeline does not refresh the figures
  the report cites. Listed in `results/README.md`; where they belong is unsettled.
- `README.md` documented `--single-run` / `--no-single-run` / `--shuffled` CLI
  flags that `train.py` has never had.
- `ARCHITECTURE.md` had the `train/` block duplicated in its source layout.

**Left untouched by request:** `compare_model_reconstructions.py` and the four
archetype scripts, pending review by their authors. `use_trained_rbm.py` keeps
its own loader because the first of those imports it.

---

## 2026-09-22

**Origin history diverged, and `code/` split into library vs. scripts**

- `git fetch` found `origin/master` 2 commits ahead of local `HEAD`, both
  pushed today by a teammate (Mattia-Ponchio) directly to `master`: `custom
  analysis` (new top-level `custom_analysis/`, duplicating three
  `code/analysis` scripts with variant behaviour, plus ~40 result files) and
  `comapre model reconstruction` (new top-level `scripts/`, duplicating
  `use_trained_rbm.py` and `compare_model_reconstructions.py` with the
  `sys.path.insert` hack `c6bb000` had removed, plus a new `get_test_set.py`).
  Local `HEAD` had 3 unpushed commits of its own (editable packaging, pytest
  suite, CI workflow) branching from the same parent. Not yet merged.
- Found in the local working tree, staged but never committed: a full copy of
  Phase A's file edits (`LOG-025`) and a copy of the teammate's
  `custom_analysis/` content byte-identical to what they later pushed —
  evidence of a prior, uncommitted local session.
- `code/train/`, `code/diagnostic/`, `code/analysis/`, `code/archive/` moved
  to `code/scripts/{train,diagnostic,analysis,archive}/`, so the repo now
  reads as library (`code/src/models/`) vs. entry points grouped by pipeline
  stage (`code/scripts/*/`), matching common src-layout convention. All
  docstring usage examples, `README.md`, `ARCHITECTURE.md` and
  `results/README.md` updated to the new paths. No import changes needed:
  scripts already resolve `models` through the editable install, not through
  path hacks.
- The locally-staged duplicate `custom_analysis/` deleted (unstaged + removed)
  as redundant with `code/scripts/analysis/`. The teammate's pushed
  `custom_analysis/` and `scripts/` will need the same treatment once
  `origin/master` is merged — not done yet, tracked as follow-up.
- Verified after the move: `py_compile` on every moved script, `import
  models` from an unrelated cwd, and `pytest tests/` (14/14) all still pass.

---

## 2026-09-22 (later)

**`results/` becomes a published tier — `publish_results.py`, `MANIFEST.json`**

- Found while verifying the `code/scripts/` rename: routine script re-runs had
  silently overwritten eight report-cited figures/tables in tracked
  `results/`, with no record of what changed or why. Root cause: nine
  scripts wrote into `results/` directly (`RESULTS_ROOT`), inconsistent with
  the rest that already staged into gitignored `diagnostic_outputs/`.
- All script defaults switched from `RESULTS_ROOT` to `DIAGNOSTIC_ROOT`,
  including `use_trained_rbm.py` / `compare_model_reconstructions.py`, whose
  `results/reconstruction_plots/` output was cwd-relative and had never
  actually gone through `paths.py` or the tracked-results `.gitignore`
  allowlist.
- New `models.manifest` (`publish()`, `load()`) and
  `code/scripts/publish_results.py`: the only path from `diagnostic_outputs/`
  into `results/`, recording producing script + git commit + timestamp per
  file in `results/MANIFEST.json`. `.gitignore` explicitly un-ignores it
  (it was silently caught by the blanket `results/*` pattern otherwise).
- `tests/test_results_manifest.py` added: fails if a tracked `results/` file
  has no manifest entry. `tests/models/test_manifest.py` covers `publish()`
  directly (copy + record, pattern filtering, no cross-producer clobbering,
  missing-input error).
- Reset today's direct-write modifications to the committed state,
  regenerated every affected script into `diagnostic_outputs/`, then
  published everything through the new tool — 95 manifest entries.
- Publishing surfaced seven tracked files no current script produces any
  more (an old flat `*_shuffled_*` naming superseded by the `shuffled/`
  subdirectory convention; three `tables/hidden/*.csv` snapshots missing the
  `zinb` family and later `L` values). Removed; superseded copies already
  exist at their correct current path. This resolves the "frozen snapshots —
  open item" `results/README.md` had carried since Phase A.
- Recorded as `LOG-025`. `results/README.md` and `ARCHITECTURE.md`
  ("Output tiers") rewritten to describe the tier boundary.

---

## 2026-09-22 (part 3)

**`code/scripts/analysis/` reorganised into `hidden/`/`archetype/`/`reconstruction/`; ambiguous output filenames fixed; three broken archetype scripts repaired**

- Raised directly: the 13 flat `analysis/` scripts were "sparsi a caso"
  (scattered randomly), and separately, files under
  `results/02_model_analysis/` couldn't be identified by name alone — e.g.
  `pattern_frequency_threshold.csv` gives no way to tell whether it came from
  `nb_sigmoid_L6` or any other run.
- Scripts split into `hidden/` (six scripts), `archetype/` (four scripts),
  `reconstruction/` (`use_trained_rbm.py` + `compare_model_reconstructions.py`,
  moved together to keep their sibling import working). `results/` and
  `diagnostic_outputs/` under `02_model_analysis/` gained matching `hidden/`
  and `archetype/` subdirectories, per explicit request to mirror the split.
- Filenames that depended on family/L/split/mode but didn't encode it now do:
  `pattern_frequency_{family}_L{L}_{mode}.csv`,
  `visible_by_hidden_{family}_L{n}[_shuffled]_seed_{k}_{mode}.csv` (tag
  derived from the `--weights` path), `cross_model_correlation_L{L}.png`,
  `nb_pattern_frequency_L{L}.png`, `seasonal_profiles_L{L}.png`.
  `nb_pattern_frequency.png` also separately redesigned this session as a
  Pareto/cumulative-coverage plot, and `seasonal_profiles.png` restyled to
  match fig3's marker+faint-dashed-connector convention (both unrelated to
  the reorg itself).
- Found while investigating: three of the four `archetype*` scripts
  (`distance_archetypes_rbm.py`, `overlap_archetypes_rbm.py`,
  `archetype_closest_rbm_scatter.py`) pointed at a pre-`training_runs/`
  layout (`weights/*.npz`, `Cheng/Data/*.csv`) that no longer exists, and two
  imported `seaborn`, not a project dependency and not installed — all three
  would have crashed immediately. Repaired: default `--weights` now resolves
  via `models.io.best_seed_dir` from `--family`/`--L`/`--split`, `--archetypes`
  defaults to `prof/archetypes_k5_profiles.csv`, and the two `seaborn.heatmap`
  calls were rewritten with plain `matplotlib.imshow` (no new dependency).
  `archetype_rbm_comparison.py`'s hardcoded `VBH` paths (pointing at a
  `results/nb_chrono_vbh/` directory that doesn't exist) now resolve the same
  way instead.
- `results/02_model_analysis/` emptied and regenerated from the moved
  scripts end to end (95 manifest entries); `MANIFEST.json` pruned of stale
  entries for paths no longer produced, scoped to `02_model_analysis/` only.
  `README.md`, `ARCHITECTURE.md`, `results/README.md` and
  `publish_results.py`'s category doc-string updated to the new paths.
- Recorded as `LOG-025`. Not done: `.claude/REORG_AND_VALIDATION.md` §A-3's
  recommendation to consolidate the three near-identical archetype scripts
  into one with a `--mode` flag, and its recommended rewrite of
  `compare_model_reconstructions.py` against the NB/ZINB-only comparison
  perimeter — both left as open follow-ups, not silently done as part of
  this pass.

---

## 2026-09-23

**nb_softmax L9/L10 sweep, clean family names in every figure, final-model figures**

- Trained `nb_softmax` at L=9 and L=10 (10 seeds each, shuffled split, same
  hyperparameters as the rest of the sweep); 20/20 runs converged, no NaN.
  Final val NLL (mean over seeds): L8 0.468, L9 0.462, L10 0.460. Only 5–7
  hidden units carry more than 2% of the days as argmax winner at any L >= 7:
  the extra softmax units stay dead, so the capacity saturates at ~6 archetypes.
- Every plot title/legend/tick now uses one family-name map
  (`FAMILY_DISPLAY_NAME` / `display_name()` in `visualization.py`, e.g.
  `nb` -> NB-Bernoulli, `nb_sigmoid` -> NB-Sigmoid, `bernoulli_median` ->
  BB-median) instead of the raw directory name. Replaces `FAMILY_SHORT_LABEL`
  and the diverging local `FAMILY_LABELS` tables in `nan_test_eval.py` and
  `split_comparison.py`. Archetype heatmaps/scatter now state which model they
  show (previously no model name, or `Weights: weights.npz`).
- Pattern-coverage plot: bold title plus a normal-weight subtitle stating the
  binarisation rule as $p(h_j=1 \mid v) \geq 0.5$; x-axis label no longer
  hardcodes `h0...h5`.
- `rbm_hidden_stackplot.py`: sampling gaps longer than 7 days are left blank
  instead of bridged by flat bands that read as a stable hidden state.
- `plot_visible_by_hidden.py`: titles name the plotted quantity (`expected
  count` / `P(v=1 | h)`) instead of the internal `zinb` mode key, which also
  applies to NB models. Filenames unchanged (read by `archetype_rbm_comparison.py`).
- Generated the full single-model figure set for the selected final model
  (NB-Sigmoid L=6, shuffled, seed_9): stackplot, pattern histogram/timeline,
  visible-by-hidden, the three archetype comparisons, plus the existing
  cross-model/pattern-coverage/seasonal figures. All `02_model_analysis/`,
  `04_model_selection/` and `diagnostics/` figures regenerated with the new
  titles and republished to `results/`.
- Not republished: `03_evaluation/` and the NaN/split evaluation tables. Their
  scripts score by unseeded Gibbs sampling, so a rerun shifts the published
  NLLs by ~1e-3 for a label-only change; the committed numbers were kept and
  the new labels will appear on their next deliberate rerun.

**Presentation material re-synced with current results**

- `materiale_presentazione/` audited file by file (hash match against `results/`).
  Slides 24–30 switched from `nb` (and a chrono fallback stackplot) to the final
  model NB-Sigmoid L=6 shuffled seed_9; superseded softmax / overview / training-curve
  figures replaced with the L≤10 versions; `INDEX.md` rewritten with the
  slide → source mapping and the skeleton claims the current data contradict
  (slide 17 "Sigmoid beats NB-Bernoulli": 0.4435 vs 0.4443 val NLL at L=6,
  within 1/10 of the seed spread; slide 19 L=10 exclusion; slide 29 report
  still on chrono nb/zinb).
- New `code/scripts/diagnostic/plot_nll_curves.py`: train/val NLL curves of
  several families at one L, replacing the archived `plot_*_nll.py` scripts
  that read the pre-`artifacts/` layout.
- `compare_model_reconstructions.py`: figure legend listed every model two or
  three times (collected from all panels); now built from the first panel only.
- Reconstruction plot for the final model (`use_trained_rbm.py`,
  `nb_sigmoid_L6_shuffled`, seed_9, shuffled test set) generated in
  `results/reconstruction_plots/` (gitignored, local only).

## 2026-09-23 (diagnostics reorganisation)

- `results/diagnostics/` restructured by figure type (LOG-025): `training_curves/{all_families_by_L,single_family_by_L,family_comparison_fixed_L}/{split}/` and `nb_zinb_parameters/{split}/`; all figures regenerated and republished with fresh manifest entries.
- New `code/scripts/diagnostic/plot_train_nll_curves.py` (one family across L); `plot_sweep_curves` now draws a 2x5 grid instead of one row of ten panels.
- `materiale_presentazione/`: figures 16/17/21 and INDEX paths refreshed to the republished versions.
