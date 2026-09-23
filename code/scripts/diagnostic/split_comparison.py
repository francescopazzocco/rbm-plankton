"""
split_comparison.py - Head-to-head NaN inference test: chronological vs shuffled split.

NaN rows are removed before the train/val split in both strategies (io._base_load),
so all 160 rows are genuinely unseen for every model here.

Models:
  NB-RBM          chrono L=6   vs  shuffled L=8
  Bernoulli-med   chrono L=6   vs  shuffled L=8

Metric: NLL on observed (non-NaN) positions only.

Inference: observed positions stay fixed while the missing ones are inferred
before the final scoring step.  Note that this script injects the *means*
(_mu / _pv_given_h) over N_GIBBS flat steps, i.e. mean-field, while
nan_test_eval.py injects samples over 5 + 3*n_miss steps.  The two scripts'
numbers are therefore not comparable with each other, and neither reproduces
the method recorded in LOG-018.  Which one becomes canonical is open decision B
in .claude/REORG_AND_VALIDATION.md, settled in Phase B.

Output:
  results/tables/split_comparison.csv
  diagnostic_outputs/03_evaluation/split_comparison.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models._eval_utils import score_row_gibbs, loss_nb, loss_bern
from models.io import (
    CHRONO, COUNT_SCALE, METRIC_COL, SHUFFLED, best_seed_dir, binarise_rows,
    load_model, load_nan_rows, model_dir, scale_counts,
)
from models.palette import get_palette
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.utils import get_device

OUT_DIR     = DIAGNOSTIC_ROOT / "tables"
FIG_DIR     = DIAGNOSTIC_ROOT / "03_evaluation"
N_SAMPLES   = 100
N_GIBBS     = 5

# NOTE: chrono L=6 vs shuffled L=8 confounds the split effect with capacity
# (item 12 in .claude/REORG_AND_VALIDATION.md).  Left as-is here: choosing the
# comparable capacities is a Phase B call, not a path fix.
CONFIGS = [
    dict(family="nb",               split=CHRONO,   L=6),
    dict(family="nb",               split=SHUFFLED, L=8),
    dict(family="bernoulli_median", split=CHRONO,   L=6),
    dict(family="bernoulli_median", split=SHUFFLED, L=8),
]

PATTERN_LABELS = {
    "p3_3miss":   "3 missing\n(n=104)",
    "p31_31miss": "31 missing\n(n=43)",
    "p54_54miss": "54 missing\n(n=13)",
}
PATTERNS = ["p3_3miss", "p31_31miss", "p54_54miss"]


# Each family already gets its own subplot (see plot_comparison), so color
# only needs to carry the split, not the family; this is the canonical
# chrono/shuffled color pair -- reuse SPLIT_COLORS (or this same [CHRONO,
# SHUFFLED] -> get_palette(2) mapping) wherever else the two splits need to
# be told apart, so "chrono" and "shuffled" mean the same color project-wide.
SPLIT_COLORS = dict(zip([CHRONO, SHUFFLED], get_palette(2)))
FAMILY_COLORS = {
    "nb":               SPLIT_COLORS,
    "bernoulli_median": SPLIT_COLORS,
}
FAMILY_LABELS = {"nb": "NB-RBM", "bernoulli_median": "Bernoulli-med"}


def plot_comparison(summary: pd.DataFrame, out: Path):
    n_patterns = len(PATTERNS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, family in zip(axes, ["nb", "bernoulli_median"]):
        sub = summary[summary["family"] == family]
        x   = np.arange(n_patterns)
        w   = 0.35
        for i, split in enumerate([CHRONO, SHUFFLED]):
            s = sub[sub["split"] == split].set_index("pattern")
            means = [s.loc[p, "nll_mean"] if p in s.index else np.nan for p in PATTERNS]
            stds  = [s.loc[p, "nll_std"]  if p in s.index else 0      for p in PATTERNS]
            color = FAMILY_COLORS[family][split]
            L_val = s["L"].iloc[0] if not s.empty else "?"
            label = f"{split}  L={L_val}"
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                   color=color, alpha=0.88, label=label,
                   error_kw={"linewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels([PATTERN_LABELS[p] for p in PATTERNS])
        ax.set_xlabel("Missingness pattern")
        ax.set_ylabel("Test NLL (observed taxa only)")
        ax.set_title(FAMILY_LABELS[family])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle("Split strategy comparison — NaN inference test", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device   = get_device()
    all_dfs  = []
    base_rows, taxa = load_nan_rows()

    for cfg in CONFIGS:
        family, split, L = cfg["family"], cfg["split"], cfg["L"]
        tag = f"{family}  {split}  L={L}"
        print(f"\n=== {tag} ===")

        rdir = model_dir(family, L, split, MODELS_ROOT)
        if not rdir.exists():
            print(f"  ! directory not found: {rdir}  — skipping")
            continue

        seed_dir = best_seed_dir(rdir, METRIC_COL[family])
        if seed_dir is None:
            print(f"  ! no valid seed found in {rdir}  — skipping")
            continue

        metric_col = METRIC_COL[family]
        df_curves  = pd.read_csv(seed_dir / "rbm_training_curves.csv")
        val_metric = df_curves[metric_col].dropna().iloc[-1]
        print(f"  seed: {seed_dir.name}   {metric_col}={val_metric:.4f}")

        model, npz = load_model(seed_dir, device)

        if family == "nb":
            nan_df   = scale_counts(base_rows, taxa, COUNT_SCALE)
            score_fn = lambda v, d: score_row_gibbs(
                model, v, d, n_samples=N_SAMPLES, impute_base=N_GIBBS, impute_per_nan=0,
                sample_hidden=model._sample_bernoulli,
                sample_visible=lambda r, h: r._mu(h),
                compute_loss=loss_nb)
        else:
            nan_df   = binarise_rows(base_rows, taxa, npz["thresholds"])
            score_fn = lambda v, d: score_row_gibbs(
                model, v, d, n_samples=N_SAMPLES, impute_base=N_GIBBS, impute_per_nan=0,
                sample_hidden=model._sample,
                sample_visible=lambda r, h: r._pv_given_h(h),
                compute_loss=loss_bern)

        records = []
        for _, row in nan_df.iterrows():
            v = row[taxa].values.astype(np.float32)
            records.append({
                "date": row["date"],
                "n_obs": int((~np.isnan(v)).sum()),
                "n_miss": int(np.isnan(v).sum()),
                "nll": score_fn(v, device),
            })
        df = pd.DataFrame(records)
        df["family"] = family
        df["split"]  = split
        df["L"]      = L
        all_dfs.append(df)

    if not all_dfs:
        print("No results — check directory names.")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    full["pattern"] = full["n_miss"].map({3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"})

    csv_out = OUT_DIR / "split_comparison.csv"
    full.to_csv(csv_out, index=False)

    summary = (full.groupby(["family", "split", "L", "pattern", "n_miss", "n_obs"])
               .agg(n_rows=("nll", "count"),
                    nll_mean=("nll", "mean"),
                    nll_std=("nll", "std"))
               .reset_index()
               .sort_values(["family", "split", "n_miss"]))

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- Split strategy comparison — NaN inference summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved: {csv_out}")

    plot_comparison(summary, FIG_DIR / "split_comparison.png")


if __name__ == "__main__":
    main()
