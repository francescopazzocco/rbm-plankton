"""
nan_test_eval.py - NaN imputation inference with validated RBMs (L=6).

For each row with missing taxa: Gibbs-impute missing values -> score NLL
on observed positions only.

Missingness patterns (after nonzero filter):
  p3:   3 NaN taxa  (104 rows, 80 observed)
  p31: 31 NaN taxa   (43 rows, 52 observed)
  p54: 54 NaN taxa   (13 rows, 29 observed)

Metrics:
  NB / ZINB families (all hidden types): NLL on raw-count observed taxa
  Bernoulli families (median / zero):    BCE on binarised observed taxa

  These are two different quantities.  They currently share one y-axis in
  nan_eval_bars.png, which is item 11 in .claude/REORG_AND_VALIDATION.md and is
  settled in Phase B, not here.

Outputs (all in diagnostic_outputs/nan_eval_extended/):
  nan_eval_rows.csv      per-row NLL, date, missingness pattern
  nan_eval_summary.csv   per-(run, pattern) mean +/- std
  nan_eval_bars.png      grouped bar chart -- all runs
  nan_eval_timeseries.png  p31 NLL time series -- all runs

Each evaluated run is identified as {family}_L{n}_{split}: the specs use the
optimal L per family, so capacity and split vary between bars.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models._eval_utils import (
    score_row_gibbs, loss_nb, loss_zinb, loss_bern,
    sample_nb, sample_zinb, sample_bern,
)
from models.io import (
    CHRONO, COUNT_SCALE, METRIC_COL, SHUFFLED, best_seed_dir, binarise_rows,
    load_model, load_nan_rows, model_dir, scale_counts,
)
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.utils import get_device
from models.visualization import COLORS, display_name


# -- Config ------------------------------------------------------------------

@dataclass
class EvalConfig:
    models_root:   Path   = MODELS_ROOT
    out_root:    Path   = field(default_factory=lambda:
                                DIAGNOSTIC_ROOT / "nan_eval_extended")
    n_samples:   int    = 100
    impute_base: int    = 5
    impute_per_nan: int = 3
    count_scale: float  = COUNT_SCALE


def read_val(seed_dir: Path, col: str) -> float:
    return pd.read_csv(seed_dir / "rbm_training_curves.csv")[col].dropna().iloc[-1]


# -- Scoring wrappers ---------------------------------------------------------

def score_nb_family(rbm, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row_gibbs(
        rbm, v_raw, device,
        n_samples=config.n_samples,
        impute_base=config.impute_base,
        impute_per_nan=config.impute_per_nan,
        sample_hidden=rbm._sample_bernoulli,
        sample_visible=sample_nb,
        compute_loss=loss_nb)


def score_zinb_family(rbm, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row_gibbs(
        rbm, v_raw, device,
        n_samples=config.n_samples,
        impute_base=config.impute_base,
        impute_per_nan=config.impute_per_nan,
        sample_hidden=rbm._sample_bernoulli,
        sample_visible=sample_zinb,
        compute_loss=loss_zinb)


def score_bern(rbm, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row_gibbs(
        rbm, v_raw, device,
        n_samples=config.n_samples,
        impute_base=config.impute_base,
        impute_per_nan=config.impute_per_nan,
        sample_hidden=rbm._sample,
        sample_visible=sample_bern,
        compute_loss=loss_bern)


# -- Evaluation loop ---------------------------------------------------------

def evaluate(rbm, rows_df: pd.DataFrame, taxa: list[str],
             config: EvalConfig, device, score_fn: Callable) -> pd.DataFrame:
    records = []
    for _, row in rows_df.iterrows():
        v = row[taxa].values.astype(np.float32)
        n_miss = int(np.isnan(v).sum())
        records.append({
            "date":   row["date"],
            "n_obs":  len(v) - n_miss,
            "n_miss": n_miss,
            "nll":    score_fn(rbm, v, config, device),
        })
    return pd.DataFrame(records)


# -- Aggregation -------------------------------------------------------------

PATTERN_MAP = {3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["run", "pattern", "n_miss", "n_obs"])
            .agg(n_rows=("nll", "count"), nll_mean=("nll", "mean"), nll_std=("nll", "std"))
            .reset_index().sort_values(["run", "n_miss"]))


# -- Figures -----------------------------------------------------------------

PATTERN_LABELS = {
    "p3_3miss":   "3 missing\n(n=104)",
    "p31_31miss": "31 missing\n(n=43)",
    "p54_54miss": "54 missing\n(n=13)",
}
PATTERNS = ["p3_3miss", "p31_31miss", "p54_54miss"]

# Project-wide canonical family->color map (models.visualization.COLORS), so
# a family gets the same color here as in plot_final_metric, split_comparison,
# and compare_model_reconstructions -- not a locally recomputed Brewer scheme.
FAMILY_COLORS = COLORS

def run_key(family: str, n_hidden: int, split: str) -> str:
    """Identity of one evaluated run.

    Capacity and split are part of the key, not implied by the family name: the
    specs below mix L values and both split strategies, and a label that hides
    that invites reading a family difference where there is a capacity or split
    difference.
    """
    return f"{family}_L{n_hidden}_{split}"


def run_label(family: str, n_hidden: int, split: str) -> str:
    return f"{display_name(family)} L={n_hidden} ({split})"


def plot_bars(summary: pd.DataFrame, out: Path, run_meta: dict):
    runs = [r for r in run_meta if r in summary["run"].unique()]
    n_runs = len(runs)
    w = 0.75 / n_runs
    x = np.arange(len(PATTERNS))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, run in enumerate(runs):
        cfg = run_meta[run]
        sub = summary[summary["run"] == run].set_index("pattern")
        means = [sub.loc[p, "nll_mean"] if p in sub.index else np.nan for p in PATTERNS]
        stds  = [sub.loc[p, "nll_std"]  if p in sub.index else 0.0    for p in PATTERNS]
        offset = (i - (n_runs - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=stds, capsize=3,
               color=cfg["color"], alpha=0.85, label=cfg["label"],
               error_kw={"linewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS[p] for p in PATTERNS])
    ax.set_xlabel("Missingness pattern")
    ax.set_ylabel("NLL on observed taxa\n(NB/ZINB: count NLL  |  Bernoulli: BCE)")
    ax.set_title("NaN test set evaluation - all model families (optimal L per family)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out: Path, run_meta: dict):
    p31 = df[df["n_miss"] == 31].copy()
    p31["date"] = pd.to_datetime(p31["date"])
    p31 = p31.sort_values("date")
    runs = [r for r in run_meta if r in p31["run"].unique()]

    fig, ax = plt.subplots(figsize=(12, 4))
    for run in runs:
        cfg = run_meta[run]
        sub = p31[p31["run"] == run]
        ax.plot(sub["date"], sub["nll"], "o-", color=cfg["color"],
                markersize=3, linewidth=1, label=cfg["label"], alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("NLL / BCE (52 observed taxa)")
    ax.set_title("p31 pattern - per-day test NLL, all model families (optimal L per family)")
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -- Main --------------------------------------------------------------------

# (family, L, split) — the optimal L per family, so the specs mix capacities and
# both split strategies.  run_key/run_label keep that visible in every output.
_SPECS = [
    ("nb",               6, CHRONO),
    ("nb",               8, SHUFFLED),
    ("nb_sigmoid",       7, SHUFFLED),
    ("nb_softmax",       7, SHUFFLED),
    ("zinb",             8, SHUFFLED),
    ("zinb_sigmoid",     7, SHUFFLED),
    ("zinb_softmax",     6, SHUFFLED),
    ("bernoulli_median", 6, CHRONO),
    ("bernoulli_zero",   6, CHRONO),
]

_SCORE_FN = {"bernoulli": score_bern, "zinb": score_zinb_family, "nb": score_nb_family}


def score_kind(family: str) -> str:
    """Which loss the family is scored with: BCE for Bernoulli, count NLL otherwise."""
    if family.startswith("bernoulli"):
        return "bernoulli"
    return "zinb" if family.startswith("zinb") else "nb"


def main():
    config = EvalConfig()
    config.out_root.mkdir(parents=True, exist_ok=True)
    device = get_device()

    nan_df, taxa = load_nan_rows()
    count_rows = scale_counts(nan_df, taxa, config.count_scale)

    all_dfs = []
    run_meta: dict[str, dict] = {}
    for family, n_hidden, split in _SPECS:
        key = run_key(family, n_hidden, split)
        fam_dir = model_dir(family, n_hidden, split, config.models_root)
        if not fam_dir.exists():
            print(f"\n[SKIP] {fam_dir.name} not found")
            continue

        seed_dir = best_seed_dir(fam_dir, METRIC_COL[family])
        if seed_dir is None:
            print(f"\n[SKIP] no valid seed in {fam_dir.name}")
            continue

        val_col = METRIC_COL[family]
        print(f"\n=== {run_label(family, n_hidden, split)} ===")
        print(f"  dir: {fam_dir.name}  seed: {seed_dir.name}  "
              f"({val_col} = {read_val(seed_dir, val_col):.4f})")

        rbm, npz = load_model(seed_dir, device)
        kind = score_kind(family)
        if kind == "bernoulli":
            rows_df = binarise_rows(nan_df, taxa, npz["thresholds"])
        else:
            rows_df = count_rows

        df_run = evaluate(rbm, rows_df, taxa, config, device, _SCORE_FN[kind])
        df_run["run"] = key
        all_dfs.append(df_run)
        run_meta[key] = {"color": FAMILY_COLORS[family],
                         "label": run_label(family, n_hidden, split)}
        print(f"  evaluated {len(df_run)} rows")

    if not all_dfs:
        print(f"\nNo runs evaluated — nothing found under {config.models_root}.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df["pattern"] = df["n_miss"].map(PATTERN_MAP)

    rows_out    = config.out_root / "nan_eval_rows.csv"
    summary     = summarise(df)
    summary_out = config.out_root / "nan_eval_summary.csv"
    df.to_csv(rows_out, index=False)
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- NaN test set evaluation summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved:    {rows_out}")
    print(f"Summary saved: {summary_out}")

    plot_bars(summary, config.out_root / "nan_eval_bars.png", run_meta)
    plot_timeseries(df, config.out_root / "nan_eval_timeseries.png", run_meta)
    print(f"Figures saved: {config.out_root}/nan_eval_bars.png  &  nan_eval_timeseries.png")


if __name__ == "__main__":
    main()
