"""
rbm_hidden_stackplot.py - Normalised stackplot of hidden activations over time.

Shows the share of total hidden activation held by each unit on each date, which
makes the seasonal handover between units visible.

Usage:
    python code/scripts/analysis/rbm_hidden_stackplot.py
    python code/scripts/analysis/rbm_hidden_stackplot.py --family nb_sigmoid --L 7 --split shuffled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.io import (
    CHRONO, METRIC_COL, SPLITS, best_seed_dir, load_hidden_activations, run_dir,
)
from models.palette import get_palette
from models.paths import DIAGNOSTIC_ROOT, RUNS_ROOT


def resolve_seed_dir(family: str, n_hidden: int, split: str, runs_root: Path) -> Path:
    """Best-converged seed directory of one run, by the family's val metric."""
    family_l_dir = run_dir(family, n_hidden, split, runs_root)
    seed_dir = best_seed_dir(family_l_dir, METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(
            f"No converged seed for {family} L={n_hidden} ({split}) in {family_l_dir}")
    return seed_dir


def normalize_rows(hidden_df: pd.DataFrame) -> pd.DataFrame:
    values = hidden_df.to_numpy(dtype=float)
    row_sums = values.sum(axis=1)
    normalized = values.copy()
    nonzero = row_sums > 0
    normalized[nonzero] = normalized[nonzero] / row_sums[nonzero, None]
    normalized[~nonzero] = 0.0
    return pd.DataFrame(normalized, columns=hidden_df.columns, index=hidden_df.index)


def plot_stackplot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = df["date"]
    hidden_cols = [col for col in df.columns if col.startswith("h")]
    normalized = normalize_rows(df[hidden_cols])

    n_hidden = len(hidden_cols)
    # get_palette repeats past 8 categories (no marker equivalent for a filled
    # stackplot band); acceptable here since adjacent bands are still split by
    # a visible boundary line, unlike overlapping scatter/line series.
    colors = get_palette(n_hidden)
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.stackplot(
        dates,
        *[normalized[col].to_numpy() for col in hidden_cols],
        labels=hidden_cols,
        colors=colors,
        alpha=1.0,
        linewidth=0.0,
    )

    for year in range(dates.dt.year.min(), dates.dt.year.max() + 1):
        ax.axvspan(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-09-01"), color="orange", alpha=0.08, lw=0)

    ax.set_title(title)
    ax.set_ylabel("Normalized hidden activation share")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    # Legend linear and below plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(10, n_hidden), frameon=False)
    # Every 2 months for easier month identification
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a normalised stackplot of RBM hidden activations over time.")
    parser.add_argument("--family", default="nb", help="Model family (default: nb)")
    parser.add_argument("--L", type=int, default=6, help="Hidden unit count (default: 6)")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Split strategy of the run (default: chrono)")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT,
                        help="Directory holding the {family}_L{n} run directories")
    parser.add_argument("--seed-dir", type=Path, default=None,
                        help="Use this seed directory directly, ignoring --family/--L/--split")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save the stackplot "
                             "(default: results/02_model_analysis/hidden_stackplot_{family}_L{n}.png)")
    parser.add_argument("--title", default=None, help="Figure title")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    seed_dir = args.seed_dir or resolve_seed_dir(
        args.family, args.L, args.split, args.runs_root)
    print(f"Reading {seed_dir}")

    output = args.output or (
        DIAGNOSTIC_ROOT / "02_model_analysis"
        / f"hidden_stackplot_{args.family}_L{args.L}.png")
    title = args.title or (
        f"{args.family} L={args.L} ({args.split}) — "
        f"hidden activation composition over time")

    df = load_hidden_activations(seed_dir / "rbm_hidden_activations.csv", indexed=False)
    plot_stackplot(df, output, title)
    print(f"Saved stackplot to {output}")


if __name__ == "__main__":
    main()
