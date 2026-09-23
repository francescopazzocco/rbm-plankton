# Results

`results/` is a **published** tier: every file in it arrived through
`code/scripts/publish_results.py`, never through a script writing here
directly. No script imports `RESULTS_ROOT` any more — they all write into the
matching subtree of the gitignored `diagnostic_outputs/` (the staging tier),
and publishing is a separate, explicit step. See `ARCHITECTURE.md` and
`DECISION_LOG` LOG-029 for why.

## Provenance — `MANIFEST.json`

Every tracked file here (except this README and the manifest itself) has an
entry in `MANIFEST.json`: which script produced it, the git commit the
working tree was at when it was published, and the publish timestamp.
`tests/test_results_manifest.py` fails CI if a tracked file has no entry —
that is what stops a plain `git add results/...` from bypassing this.

To look up what produced a file: `results/MANIFEST.json` is keyed by path
relative to `results/`. To publish fresh output, regenerate it into
`diagnostic_outputs/` by running the producing script (below), then:

```
python code/scripts/publish_results.py <category>   # e.g. 02_model_analysis
python code/scripts/publish_results.py --all         # everything at once
```

## Figures

| Directory | Content | Produced by |
|---|---|---|
| `01_exploratory/` | Dataset EDA: row sums, periodogram, seasonal patterns, marginal distributions, NaN structure | `code/scripts/train/dataset_analysis.py` |
| `02_model_analysis/hidden/<kind>/{chrono,shuffled}/` | Learned representations, one subfolder per analysis kind (`weight_profiles`, `state_timeline`, `state_frequency`, `dominant_state`, `mean_activation`, `cross_model_correlation`, `nb_pattern_frequency`, `seasonal_profiles`, `hidden_stackplot`, `visible_by_hidden`) | the `code/scripts/analysis/hidden/` scripts (LOG-033) |
| `02_model_analysis/hidden/patterns/{chrono,shuffled}/` | Binary hidden patterns of a single run | `hidden_pattern_analysis.py` |
| `02_model_analysis/archetype/<kind>/{chrono,shuffled}/` | RBM vs. Cheng's k=5 archetypes, one subfolder per kind (`distance_heatmap`, `overlap_heatmap`, `archetype_closest_rbm`) | the `code/scripts/analysis/archetype/` scripts (LOG-033) |
| `03_evaluation/` | NaN imputation test, chronological vs shuffled split comparison | `code/scripts/diagnostic/nan_test_eval.py`, `split_comparison.py` |
| `04_model_selection/{chrono,shuffled}/` | Final validation metrics vs L | `code/scripts/diagnostic/sweep_analysis.py`, `code/scripts/archive/plot_final_metric_nb.py` |
| `diagnostics/training_curves/all_families_by_L/{split}/` | Val metric vs epoch, every family, one line per L | `code/scripts/diagnostic/sweep_analysis.py` |
| `diagnostics/training_curves/single_family_by_L/{split}/` | Train NLL vs epoch of one family, one line per L (mean ± 1σ over seeds) | `code/scripts/diagnostic/plot_train_nll_curves.py` |
| `diagnostics/training_curves/family_comparison_fixed_L/{split}/` | Train/val NLL of several families at one L (mean ± 1σ over seeds) | `code/scripts/diagnostic/plot_nll_curves.py` |
| `diagnostics/nb_zinb_parameters/{split}/` | NB/ZINB val NLL with θ (and π) trajectories by L | `code/scripts/diagnostic/sweep_analysis.py` |

Every split-aware directory holds a `chrono/` and a `shuffled/` subdirectory,
symmetrically — neither split gets an unlabelled default location.

## Tables

| Path | Content | Produced by |
|---|---|---|
| `tables/hidden/{chrono,shuffled}/` | Hidden unit activation analysis CSVs (cross-model correlation, matched pairs, seasonal profiles, pattern frequency) | `hidden_cross_model.py` |
| `tables/split_comparison.csv` | Head-to-head chronological vs shuffled split NLL comparison | `code/scripts/diagnostic/split_comparison.py` |
| `tables/nan_eval_rows.csv`, `tables/nan_eval_summary.csv` | Per-row and aggregated NaN imputation NLL | `code/scripts/diagnostic/nan_test_eval.py` |

## Not published here

`diagnostic_outputs/training_curves/`, `reconstruction_plots/`,
`nan_eval_extended/` (its non-csv/png remainder) and `zinb_meanfield_test/`
are scratch-only by design — exploratory or per-run inspection output with no
report-citation role. They stay in `diagnostic_outputs/`, gitignored,
regenerated on demand, never published.

`training_runs/` (the trained weights themselves) is a separate tier again:
gitignored, never committed, owned by `paths.RUNS_ROOT`. Not covered by this
manifest.
