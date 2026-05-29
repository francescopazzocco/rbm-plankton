from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("weights/NB_RBM/L6_chrono")
DEFAULT_OUTPUT = Path("analysis/results/nb_rbm_hidden_stackplot.png")


def load_hidden_activations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    hidden_cols = [col for col in df.columns if col.startswith("h")]
    if not hidden_cols:
        raise ValueError(f"No hidden-unit columns found in {csv_path}")
    return df[["date", *hidden_cols]].sort_values("date").reset_index(drop=True)


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
    cmap = matplotlib.colormaps["gist_ncar"].resampled(n_hidden)
    colors = [cmap(x) for x in np.linspace(0, 1, n_hidden)]
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
    parser = argparse.ArgumentParser(description="Plot a normalized stackplot of NB-RBM hidden activations over time.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing rbm_hidden_activations.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the stackplot image")
    parser.add_argument("--title", default="NB-RBM hidden activation composition over time", help="Figure title")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_dir / "rbm_hidden_activations.csv"
    df = load_hidden_activations(input_path)
    plot_stackplot(df, args.output, args.title)
    print(f"Saved stackplot to {args.output}")


if __name__ == "__main__":
    main()
