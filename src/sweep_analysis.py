"""
sweep_analysis.py — NLL/PLL vs L sweep analysis across all trained models.

Reads results/{family}_L{n}/rbm_training_curves.csv for all available runs and
produces three figures:
  1. Final val metric vs L per model family
  2. Val metric training curves overlaid per family
  3. NB-specific: NLL and theta_mean trajectories per L (instability diagnostic)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "sweep"

# Metric column and direction per family (higher/lower is better)
FAMILY_META = {
    "bernoulli_median": dict(col="val_pll", label="PLL (↑)",   better="higher"),
    "bernoulli_zero":   dict(col="val_pll", label="PLL (↑)",   better="higher"),
    "nb":               dict(col="val_nll", label="NLL (↓)",   better="lower"),
}

COLORS = {
    "bernoulli_median": "#1f77b4",
    "bernoulli_zero":   "#ff7f0e",
    "nb":               "#2ca02c",
}


def discover_runs(results_dir: Path) -> dict[str, dict[int, Path]]:
    """Return {family: {L: csv_path}} for all valid result directories."""
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, Path]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        csv = d / "rbm_training_curves.csv"
        if csv.exists():
            runs.setdefault(family, {})[l_val] = csv
    return runs


def load_curves(csv_path: Path, col: str) -> pd.Series | None:
    """Load a metric column from a training curves CSV; return None if all NaN."""
    df = pd.read_csv(csv_path)
    if col not in df.columns or df[col].isna().all():
        return None
    return df.set_index("epoch")[col]


def plot_final_metric(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(14, 4), sharey=False)
    fig.suptitle("Final val metric vs L (last epoch)", fontsize=13)

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        xs, ys = [], []
        for l_val in sorted(family_runs):
            curve = load_curves(family_runs[l_val], col)
            if curve is None:
                continue
            xs.append(l_val)
            ys.append(curve.iloc[-1])

        color = COLORS[family]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=7)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

        ax.set_title(family)
        ax.set_xlabel("L (hidden units)")
        ax.set_ylabel(meta["label"])
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_final_metric.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_training_curves(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(15, 4), sharey=False)
    fig.suptitle("Val metric training curves by L", fontsize=13)

    cmap = plt.colormaps["viridis"]

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        l_values = sorted(family_runs)
        n = len(l_values)

        for i, l_val in enumerate(l_values):
            curve = load_curves(family_runs[l_val], col)
            if curve is None:
                ax.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i*0.07),
                            xycoords="axes fraction", fontsize=8, color="red")
                continue
            color = cmap(i / max(n - 1, 1))
            ax.plot(curve.index, curve.values, color=color,
                    linewidth=1.5, label=f"L={l_val}")

        ax.set_title(family)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(meta["label"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_nb_diagnostics(runs, figures_dir: Path):
    """Two-panel figure: NB val_nll and theta_mean trajectories per L."""
    nb_runs = runs.get("nb", {})
    l_values = sorted(nb_runs)
    n = len(l_values)
    cmap = plt.colormaps["viridis"]

    fig, (ax_nll, ax_theta) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NB-RBM: NLL and θ trajectories by L", fontsize=13)

    for i, l_val in enumerate(l_values):
        csv = nb_runs[l_val]
        color = cmap(i / max(n - 1, 1))
        label = f"L={l_val}"

        nll = load_curves(csv, "val_nll")
        theta = load_curves(csv, "theta_mean")

        if nll is not None:
            ax_nll.plot(nll.index, nll.values, color=color, linewidth=1.5, label=label)
        else:
            ax_nll.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i * 0.07),
                            xycoords="axes fraction", fontsize=8, color="red")

        if theta is not None:
            ax_theta.plot(theta.index, theta.values, color=color, linewidth=1.5, label=label)

    ax_nll.set_title("Val NLL (↓)")
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.legend(fontsize=8)
    ax_nll.grid(True, alpha=0.3)

    ax_theta.set_title("θ mean (dispersion)")
    ax_theta.set_xlabel("Epoch")
    ax_theta.set_ylabel("θ")
    ax_theta.legend(fontsize=8)
    ax_theta.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_nb_diagnostics.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def print_improvement_table(runs):
    """Print relative val metric improvement per L step for each family."""
    print("\n── Relative improvement per L step ──")
    print(f"{'family':<22} {'L→L+1':<10} {'val metric':<12} {'Δ abs':<10} {'Δ %'}")
    print("-" * 65)

    for family, meta in FAMILY_META.items():
        col = meta["col"]
        better = meta["better"]
        family_runs = runs.get(family, {})

        pairs: list[tuple[int, float]] = []
        for l_val in sorted(family_runs):
            curve = load_curves(family_runs[l_val], col)
            if curve is not None:
                pairs.append((l_val, curve.iloc[-1]))

        for i in range(len(pairs) - 1):
            l_a, v_a = pairs[i]
            l_b, v_b = pairs[i + 1]
            delta = v_b - v_a
            rel   = delta / abs(v_a) * 100
            # for PLL higher is better (positive delta = improvement)
            # for NLL lower is better (negative delta = improvement)
            marker = ""
            if better == "higher" and delta > 0:
                marker = "↑"
            elif better == "lower" and delta < 0:
                marker = "↓"
            print(f"{family:<22} {l_a}→{l_b:<8} {v_a:<12.4f} {delta:+.4f}   {rel:+.2f}% {marker}")
        print()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    print("Discovered runs:")
    for family, ls in sorted(runs.items()):
        print(f"  {family}: L={sorted(ls.keys())}")

    print_improvement_table(runs)
    plot_final_metric(runs, FIGURES_DIR)
    plot_training_curves(runs, FIGURES_DIR)
    plot_nb_diagnostics(runs, FIGURES_DIR)


if __name__ == "__main__":
    main()
