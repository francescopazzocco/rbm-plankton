"""
plot_training_runs.py — Plot all trained models from trained_models/
=============================================================================

Reads the CSV files saved in each seed folder and generates plots:
- Training curves (MSE, PLL/NLL)
- Weight heatmaps
- Hidden activation time series

Saves figures to /diagnostic_outputs/training_curves/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from models.visualization import plot_training_curves, plot_weight_heatmap


def load_history_from_csv(csv_path: Path) -> dict:
    """Load training history from rbm_training_curves.csv"""
    df = pd.read_csv(csv_path)
    history = {
        "epoch": df["epoch"].tolist(),
        "train_mse": df["train_mse"].tolist(),
    }
    if "val_mse" in df.columns:
        history["val_mse"] = df["val_mse"].tolist()
    if "train_pll" in df.columns:
        history["train_pll"] = df["train_pll"].tolist()
    if "val_pll" in df.columns:
        history["val_pll"] = df["val_pll"].tolist()
    if "train_nll" in df.columns:
        history["train_nll"] = df["train_nll"].tolist()
    if "val_nll" in df.columns:
        history["val_nll"] = df["val_nll"].tolist()
    return history


def load_weights_from_csv(csv_path: Path) -> tuple[np.ndarray, list]:
    """Load weight matrix and taxa columns from rbm_weights.csv"""
    df = pd.read_csv(csv_path, index_col=0)
    W = df.values
    taxa_cols = df.index.tolist()
    return W, taxa_cols


def plot_hidden_activations_from_csv(activations_csv: Path, out_dir: Path):
    """Plot hidden activations time series from saved CSV"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(activations_csv, index_col=0, parse_dates=True)
    dates_all = df.index
    H_all = df.values
    n_hidden = H_all.shape[1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(n_hidden, 1, figsize=(13, 2.5 * n_hidden), sharex=True)
    if n_hidden == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        vals = H_all[:, j]
        near_0 = (vals < 0.1).mean()
        near_1 = (vals > 0.9).mean()
        mid = 1 - near_0 - near_1
        ax.plot(dates_all, vals, lw=0.5, color=colors[j % len(colors)], alpha=0.6)
        ax.plot(dates_all, pd.Series(vals).rolling(14, center=True).mean(),
                lw=1.8, color=colors[j % len(colors)])
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(h=1|v)", fontsize=8)
        ax.set_title(
            f"h{j}  |  <0.1: {near_0:.0%}   >0.9: {near_1:.0%}   "
            f"middle: {mid:.0%}  ->  {'binary' if mid < 0.15 else 'continuous'}",
            fontsize=8, loc="left"
        )
        ax.grid(True, alpha=0.25)
        for year in range(2019, 2025):
            ax.axvspan(pd.Timestamp(f"{year}-06-01"),
                       pd.Timestamp(f"{year}-09-01"),
                       alpha=0.07, color="orange")

    axes[-1].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.suptitle("Hidden unit activations h(t)  |  orange = summer", fontsize=11)
    plt.tight_layout()
    path = out_dir / "hidden_activations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")


def process_training_run(run_dir: Path, figures_dir: Path):
    """Process all seeds in a training run directory"""
    run_name = run_dir.name
    print(f"\nProcessing: {run_name}")

    for seed_dir in sorted(run_dir.glob("seed_*")):
        seed_num = seed_dir.name
        print(f"  Seed: {seed_num}")

        # Create output directory for this seed
        out_dir = figures_dir / run_name / seed_num
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plot training curves
        curves_csv = seed_dir / "rbm_training_curves.csv"
        if curves_csv.exists():
            history = load_history_from_csv(curves_csv)
            plot_training_curves(history, str(out_dir))

        # Plot weight heatmap
        weights_csv = seed_dir / "rbm_weights.csv"
        if weights_csv.exists():
            W, taxa_cols = load_weights_from_csv(weights_csv)
            plot_weight_heatmap(W, taxa_cols, str(out_dir))

        # Plot hidden activations
        activations_csv = seed_dir / "rbm_hidden_activations.csv"
        if activations_csv.exists():
            plot_hidden_activations_from_csv(activations_csv, out_dir)


def main():
    base_dir = Path(__file__).parent.parent.parent
    training_runs_dir = base_dir / "trained_models"
    figures_dir = base_dir / "diagnostic_outputs" / "training_curves"

    figures_dir.mkdir(parents=True, exist_ok=True)

    if not training_runs_dir.exists():
        print(f"Error: {training_runs_dir} does not exist")
        return

    # Process all training runs
    for run_dir in sorted(training_runs_dir.iterdir()):
        if run_dir.is_dir():
            process_training_run(run_dir, figures_dir)

    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
